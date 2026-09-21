//! Admission, in-flight coalescing, per-model queue, batcher and worker threads.
use crate::cache::{Cache, Key};
use crate::model::postprocess::Raw;
use crate::model::sequence::Encoded;
use futures::future::{FutureExt, Shared, WeakShared};
use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::time::{Duration, Instant};

/// A model runtime. Implemented by `OrtBackend` (real) and `fake::Fake` (tests, stress).
pub trait Backend: Send {
    fn run(&mut self, batch: &Batch) -> Result<Vec<Raw>, String>;
}

/// Builds one backend per worker thread (and again after a worker panic).
pub type BackendFactory = Arc<dyn Fn() -> Result<Box<dyn Backend>, String> + Send + Sync>;

pub struct Batch {
    pub items: Vec<Arc<Encoded>>,
    pub pad_id: u32,
}

pub struct ModelSpec {
    pub name: String,
    pub workers: usize,
    pub max_pending: usize,
    pub max_batch_items: usize,
    pub max_batch_tokens: usize,
    pub max_wait: Duration,
    pub pad_id: u32,
    pub factory: BackendFactory,
}

/// One unit of work: a cache-missed question, already tokenized.
#[derive(Clone)]
pub struct Job {
    pub key: Key,
    pub model: usize,
    pub enc: Arc<Encoded>,
}

#[derive(Debug, PartialEq)]
pub enum SchedError {
    Overloaded,
    Deadline,
    Backend(String),
}

/// Rounds of "joined a dying entry" re-admission before giving up (bounds a hot retry loop).
const MAX_READMITS: u32 = 3;

type Outcome = Result<Arc<Raw>, String>;
type Sub = Shared<oneshot::Receiver<Outcome>>;
type Map = Arc<Mutex<HashMap<Key, Entry>>>;
type Dispatch = (Vec<WorkItem>, OwnedSemaphorePermit);

struct Entry {
    generation: u64,
    weak: WeakShared<oneshot::Receiver<Outcome>>,
    deadline: Arc<Mutex<Instant>>,
}

/// Owns the permit and the result sender. Every exit path (completion, error, skip, panic,
/// shutdown) ends by dropping it, which removes its map entry (if still its own) and frees the permit.
/// Never drop one while holding the map lock (`Drop` takes it).
pub struct WorkItem {
    key: Key,
    generation: u64,
    enc: Arc<Encoded>,
    tx: Option<oneshot::Sender<Outcome>>,
    deadline: Arc<Mutex<Instant>>,
    map: Map,
    _permit: OwnedSemaphorePermit,
    /// `enc.ids.len()`, counted in `queued` until this item drops.
    tokens: usize,
    queued: Arc<AtomicUsize>,
}

impl WorkItem {
    fn live(&self, now: Instant) -> bool {
        self.tx.as_ref().is_some_and(|t| !t.is_closed()) && now < *self.deadline.lock().unwrap()
    }

    /// Returns false when nobody was listening (orphaned inference).
    fn finish(&mut self, out: Outcome) -> bool {
        self.tx.take().is_some_and(|t| t.send(out).is_ok())
    }
}

impl Drop for WorkItem {
    fn drop(&mut self) {
        self.queued.fetch_sub(self.tokens, Ordering::Relaxed);
        let mut map = self.map.lock().unwrap();
        if map.get(&self.key).is_some_and(|e| e.generation == self.generation) {
            map.remove(&self.key);
        }
    }
}

struct ModelQueue {
    name: String,
    max_pending: usize,
    permits: Arc<Semaphore>,
    tx: mpsc::UnboundedSender<WorkItem>,
    workers: usize,
    live: Arc<AtomicUsize>,
    /// EMA of inference seconds per real token (absorbs average padding), as f64 bits; 0.0 = no estimate yet.
    secs_per_token: Arc<AtomicU64>,
    /// Tokens of admitted items not yet finished or dropped (queued or in a running batch).
    queued_tokens: Arc<AtomicUsize>,
}

#[derive(Clone)]
pub struct Scheduler {
    map: Map,
    models: Arc<Vec<ModelQueue>>,
    next_gen: Arc<AtomicU64>,
}

impl Scheduler {
    /// Spawns worker threads (each builds its backend; any failure aborts startup) and one batcher
    /// task per model. Must be called inside a tokio runtime.
    pub fn start(specs: Vec<ModelSpec>, cache: Arc<Cache>) -> Result<Self, String> {
        let rt = tokio::runtime::Handle::current();
        let map: Map = Arc::default();
        let mut models = Vec::new();
        for spec in specs {
            let permits = Arc::new(Semaphore::new(spec.max_pending));
            let (tx, rx) = mpsc::unbounded_channel();
            let (work_tx, work_rx) = std_mpsc::sync_channel::<Dispatch>(spec.workers);
            let work_rx = Arc::new(Mutex::new(work_rx));
            let (ready_tx, ready_rx) = std_mpsc::channel();
            let idle = Arc::new(Semaphore::new(spec.workers));
            let live = Arc::new(AtomicUsize::new(spec.workers));
            let secs_per_token = Arc::new(AtomicU64::new(0));
            for w in 0..spec.workers {
                let ctx = WorkerCtx {
                    name: spec.name.clone(),
                    factory: spec.factory.clone(),
                    pad_id: spec.pad_id,
                    rx: work_rx.clone(),
                    cache: cache.clone(),
                    rt: rt.clone(),
                    permits: permits.clone(),
                    max_pending: spec.max_pending,
                    idle: idle.clone(),
                    live: live.clone(),
                    secs_per_token: secs_per_token.clone(),
                };
                let ready = ready_tx.clone();
                std::thread::Builder::new()
                    .name(format!("infer-{}-{w}", spec.name))
                    .spawn(move || worker(ctx, ready))
                    .map_err(|e| e.to_string())?;
            }
            for _ in 0..spec.workers {
                ready_rx.recv().map_err(|_| format!("model {}: worker exited during startup", spec.name))??;
            }
            let limits = (spec.max_batch_items, spec.max_batch_tokens, spec.max_wait);
            rt.spawn(batcher(rx, limits, idle, work_tx));
            models.push(ModelQueue {
                name: spec.name,
                max_pending: spec.max_pending,
                permits,
                tx,
                workers: spec.workers,
                live,
                secs_per_token,
                queued_tokens: Arc::default(),
            });
        }
        Ok(Self { map, models: Arc::new(models), next_gen: Arc::default() })
    }

    /// Every model has at least one live worker.
    pub fn ready(&self) -> bool {
        self.models.iter().all(|q| q.live.load(Ordering::Relaxed) > 0)
    }

    /// Runs every job (unique keys) to completion, coalescing with identical in-flight work.
    pub async fn run_jobs(&self, jobs: Vec<Job>, deadline: Instant) -> Result<HashMap<Key, Arc<Raw>>, SchedError> {
        let mut todo: HashMap<Key, Job> = jobs.into_iter().map(|j| (j.key, j)).collect();
        if todo.values().any(|j| j.model >= self.models.len()) {
            return Err(SchedError::Backend("unknown model index".into())); // before the map lock
        }
        let mut done = HashMap::with_capacity(todo.len());
        let mut readmits = 0;
        while !todo.is_empty() {
            if Instant::now() >= deadline {
                return Err(SchedError::Deadline);
            }
            let subs = self.admit(todo.values(), deadline)?;
            let waits = futures::future::join_all(subs.into_iter().map(|(k, s)| async move { (k, s.await) }));
            let results = tokio::time::timeout_at(deadline, waits).await.map_err(|_| SchedError::Deadline)?;
            let mut dropped = false;
            for (k, r) in results {
                match r {
                    Ok(Ok(raw)) => {
                        todo.remove(&k);
                        done.insert(k, raw);
                    }
                    Ok(Err(e)) => return Err(SchedError::Backend(e)),
                    Err(_) => dropped = true, // item dropped without a result (joined a dying entry): admit again
                }
            }
            if dropped {
                readmits += 1;
                if readmits > MAX_READMITS {
                    return Err(SchedError::Backend("work items dropped without a result".into()));
                }
            }
        }
        Ok(done)
    }

    /// Step 4: one critical section. Joins live entries (no permit), else takes permits for all new
    /// items of every model (all-or-nothing, fixed model order) and registers them.
    fn admit<'a>(&self, jobs: impl Iterator<Item = &'a Job>, deadline: Instant) -> Result<Vec<(Key, Sub)>, SchedError> {
        let mut subs = Vec::new();
        let mut parts = Vec::new();
        {
            let mut map = self.map.lock().unwrap();
            let mut fresh: Vec<&Job> = Vec::new();
            for job in jobs {
                if let Some(s) = map.get(&job.key).and_then(|e| e.weak.upgrade().map(|s| (s, e.deadline.clone()))) {
                    let mut d = s.1.lock().unwrap();
                    *d = (*d).max(deadline);
                    subs.push((job.key, s.0));
                    metrics::counter!("rsdecider_coalesced_total").increment(1);
                } else {
                    fresh.push(job);
                }
            }
            let mut counts = vec![0u32; self.models.len()];
            fresh.iter().for_each(|j| counts[j.model] += 1);
            let mut permits: Vec<Option<OwnedSemaphorePermit>> = Vec::with_capacity(counts.len());
            for (m, &n) in counts.iter().enumerate() {
                if n == 0 {
                    permits.push(None);
                    continue;
                }
                let q = &self.models[m];
                // Shed now if the work already queued ahead plus this request cannot finish before the
                // deadline (fast 529, not a late 504). An idle queue always admits, so a stale high
                // estimate can't shed large requests forever.
                let spt = f64::from_bits(q.secs_per_token.load(Ordering::Relaxed));
                let ahead = q.queued_tokens.load(Ordering::Relaxed);
                let own: usize = fresh.iter().filter(|j| j.model == m).map(|j| j.enc.ids.len()).sum();
                let left = deadline.saturating_duration_since(Instant::now()).as_secs_f64();
                if spt > 0.0 && ahead > 0 && (ahead + own) as f64 * spt / q.workers as f64 > left {
                    metrics::counter!("rsdecider_shed_total", "model" => q.name.clone()).increment(1);
                    return Err(SchedError::Overloaded);
                }
                match q.permits.clone().try_acquire_many_owned(n) {
                    Ok(p) => permits.push(Some(p)),
                    Err(_) => {
                        metrics::counter!("rsdecider_shed_total", "model" => q.name.clone()).increment(1);
                        return Err(SchedError::Overloaded);
                    }
                }
                metrics::gauge!("rsdecider_pending", "model" => q.name.clone())
                    .set((q.max_pending - q.permits.available_permits()) as f64);
            }
            for job in fresh {
                let (tx, rx) = oneshot::channel();
                let shared = rx.shared();
                let generation = self.next_gen.fetch_add(1, Ordering::Relaxed);
                let dl = Arc::new(Mutex::new(deadline));
                let weak = shared.downgrade().expect("fresh future is pending");
                map.insert(job.key, Entry { generation, weak, deadline: dl.clone() });
                // Counted inside the lock so the next admit sees it; this job's WorkItem subtracts it on drop.
                self.models[job.model].queued_tokens.fetch_add(job.enc.ids.len(), Ordering::Relaxed);
                let permit = permits[job.model].as_mut().unwrap().split(1).unwrap();
                parts.push((job.clone(), generation, tx, dl, permit));
                subs.push((job.key, shared));
            }
        }
        // Lock released: build WorkItems and enqueue, with no await in between.
        for (job, generation, tx, deadline, permit) in parts {
            let q = &self.models[job.model];
            let tokens = job.enc.ids.len();
            let item = WorkItem {
                key: job.key,
                generation,
                enc: job.enc,
                tx: Some(tx),
                deadline,
                map: self.map.clone(),
                _permit: permit,
                tokens,
                queued: q.queued_tokens.clone(),
            };
            // A send error means the batcher is gone (shutdown, or every worker retired): fail
            // subscribers now instead of letting them re-admit until the deadline.
            if let Err(mpsc::error::SendError(mut item)) = q.tx.send(item) {
                item.finish(Err("model workers unavailable".into()));
            }
        }
        Ok(subs)
    }
}

/// Step 6: once a worker is idle, collect up to max items / max wait (plus everything already queued),
/// then dispatch the oldest item with the buffered items closest to its length (less padding).
/// Surplus work stays buffered where it can still be skipped.
async fn batcher(
    mut rx: mpsc::UnboundedReceiver<WorkItem>,
    (max_items, max_tokens, max_wait): (usize, usize, Duration),
    idle: Arc<Semaphore>,
    work: std_mpsc::SyncSender<Dispatch>,
) {
    // Bounded by `max_pending`: every buffered item holds a permit.
    let mut buf: VecDeque<WorkItem> = VecDeque::new();
    loop {
        let Ok(idle_permit) = idle.clone().acquire_owned().await else { return };
        if buf.is_empty() {
            match rx.recv().await {
                Some(i) => buf.push_back(i),
                None => return,
            }
        }
        let close = Instant::now() + max_wait;
        // Stop early once the buffer can already fill a batch (real tokens >= cap implies padded >= cap).
        while buf.len() < max_items && buf.iter().map(|i| i.enc.ids.len()).sum::<usize>() < max_tokens {
            let Ok(Some(item)) = tokio::time::timeout_at(close, rx.recv()).await else { break };
            buf.push_back(item);
        }
        while let Ok(item) = rx.try_recv() {
            buf.push_back(item); // everything already queued is a candidate
        }
        let now = Instant::now();
        buf.retain(|i| i.live(now)); // dropped here: entry removed, permit released
        let Some(first) = buf.pop_front() else { continue };
        let batch = pick(first, &mut buf, max_items, max_tokens);
        // Never blocks: at most `workers` idle permits exist and the channel holds `workers`.
        if work.send((batch, idle_permit)).is_err() {
            return;
        }
    }
}

/// The oldest item, then the buffered items closest to its length, under the padded-token cap
/// (ORT allocates batch × longest).
fn pick(first: WorkItem, buf: &mut VecDeque<WorkItem>, max_items: usize, max_tokens: usize) -> Vec<WorkItem> {
    let target = first.enc.ids.len();
    let mut order: Vec<usize> = (0..buf.len()).collect();
    order.sort_by_key(|&i| (buf[i].enc.ids.len().abs_diff(target), i));
    let (mut longest, mut chosen) = (target, Vec::new());
    for i in order {
        if chosen.len() + 1 >= max_items {
            break;
        }
        let len = longest.max(buf[i].enc.ids.len());
        if (chosen.len() + 2) * len <= max_tokens {
            longest = len;
            chosen.push(i);
        }
    }
    // Highest index first, so earlier removals don't shift the ones still to come.
    chosen.sort_unstable_by(|a, b| b.cmp(a));
    let mut batch = vec![first];
    batch.extend(chosen.into_iter().map(|i| buf.remove(i).expect("index in range")));
    batch
}

struct WorkerCtx {
    name: String,
    factory: BackendFactory,
    pad_id: u32,
    rx: Arc<Mutex<std_mpsc::Receiver<Dispatch>>>,
    cache: Arc<Cache>,
    rt: tokio::runtime::Handle,
    permits: Arc<Semaphore>,
    max_pending: usize,
    idle: Arc<Semaphore>,
    live: Arc<AtomicUsize>,
    secs_per_token: Arc<AtomicU64>,
}

fn worker(ctx: WorkerCtx, ready: std_mpsc::Sender<Result<(), String>>) {
    let mut backend = match (ctx.factory)() {
        Ok(b) => {
            let _ = ready.send(Ok(()));
            b
        }
        Err(e) => {
            let _ = ready.send(Err(format!("model {}: {e}", ctx.name)));
            return;
        }
    };
    loop {
        let msg = ctx.rx.lock().unwrap().recv();
        let Ok((items, idle_permit)) = msg else { return };
        let batch = Batch { items: items.iter().map(|i| i.enc.clone()).collect(), pad_id: ctx.pad_id };
        let started = std::time::Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| backend.run(&batch)));
        let batch_secs = started.elapsed().as_secs_f64();
        metrics::histogram!("rsdecider_inference_seconds", "model" => ctx.name.clone()).record(batch_secs);
        metrics::histogram!("rsdecider_batch_size", "model" => ctx.name.clone()).record(items.len() as f64);
        let mut panicked = false;
        let failure = match result {
            Ok(Ok(raws)) if raws.len() == items.len() => {
                let x = batch_secs / items.iter().map(|i| i.enc.ids.len()).sum::<usize>().max(1) as f64;
                let _ = ctx.secs_per_token.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| {
                    let old = f64::from_bits(b);
                    Some(if old == 0.0 { x } else { 0.8 * old + 0.2 * x }.to_bits())
                });
                for (mut item, raw) in items.into_iter().zip(raws) {
                    let (key, raw) = (item.key, Arc::new(raw));
                    ctx.cache.put_l1(key, raw.clone()); // L1 before the entry disappears
                    if !item.finish(Ok(raw.clone())) {
                        metrics::counter!("rsdecider_orphaned_inference_total", "model" => ctx.name.clone())
                            .increment(1);
                    }
                    drop(item);
                    let cache = ctx.cache.clone();
                    ctx.rt.spawn(async move { cache.put_l2(key, raw).await });
                }
                metrics::gauge!("rsdecider_pending", "model" => ctx.name.clone())
                    .set((ctx.max_pending - ctx.permits.available_permits()) as f64);
                continue;
            }
            Ok(Ok(raws)) => format!("backend returned {} outputs for {} items", raws.len(), items.len()),
            Ok(Err(e)) => e,
            Err(_) => {
                panicked = true;
                "inference worker panicked".to_string()
            }
        };
        tracing::error!(model = %ctx.name, "batch failed: {failure}");
        for mut item in items {
            item.finish(Err(failure.clone()));
        }
        if panicked {
            // ponytail: a failed rebuild retires this worker; the model runs with fewer workers.
            match (ctx.factory)() {
                Ok(b) => backend = b,
                Err(e) => {
                    tracing::error!(model = %ctx.name, "worker retired, rebuild failed: {e}");
                    // One fewer idle slot, so the batcher never hands a batch to nobody; the last
                    // worker closes it, which stops the batcher and fails new work fast.
                    idle_permit.forget();
                    if ctx.live.fetch_sub(1, Ordering::Relaxed) == 1 {
                        ctx.idle.close();
                    }
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CacheCfg;
    use crate::fake::Fake;
    use crate::model::sequence::QType;
    use std::sync::atomic::Ordering::SeqCst;

    fn job(n: u8, model: usize, len: usize) -> Job {
        Job {
            key: [n; 32],
            model,
            enc: Arc::new(Encoded { ids: vec![n as u32; len], markers: vec![1, 2], qtype: QType::Choice }),
        }
    }

    fn spec(name: &str, fake: &Fake, max_pending: usize, max_items: usize, max_tokens: usize) -> ModelSpec {
        ModelSpec {
            name: name.into(),
            workers: 1,
            max_pending,
            max_batch_items: max_items,
            max_batch_tokens: max_tokens,
            max_wait: Duration::from_millis(5),
            pad_id: 0,
            factory: fake.factory(),
        }
    }

    async fn sched(specs: Vec<ModelSpec>) -> Scheduler {
        Scheduler::start(specs, Arc::new(Cache::new(&CacheCfg::default(), Duration::from_millis(250)).await.unwrap()))
            .unwrap()
    }

    fn secs(n: u64) -> Instant {
        Instant::now() + Duration::from_secs(n)
    }

    /// Polls `cond` every millisecond, failing the test after 2 s.
    async fn wait_until(cond: impl Fn() -> bool) {
        for _ in 0..2000 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("condition not met within 2 s");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn identical_concurrent_jobs_run_once() {
        let fake = Fake { delay: Duration::from_millis(50), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 256, 8, 8192)]).await;
        let tasks: Vec<_> = (0..100)
            .map(|_| {
                let s = s.clone();
                tokio::spawn(async move { s.run_jobs(vec![job(1, 0, 4)], secs(5)).await })
            })
            .collect();
        for t in tasks {
            assert!(t.await.unwrap().is_ok());
        }
        assert_eq!(fake.items.load(SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn admission_is_all_or_nothing_across_models() {
        let fake = Fake { delay: Duration::from_millis(100), ..Fake::default() };
        let s = sched(vec![spec("a", &fake, 4, 8, 8192), spec("b", &fake, 2, 8, 8192)]).await;
        let busy = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(9, 1, 4)], secs(5)).await })
        };
        wait_until(|| s.map.lock().unwrap().contains_key(&[9; 32])).await; // model b: 1 of 2 permits used
        let r = s.run_jobs(vec![job(1, 0, 4), job(2, 1, 4), job(3, 1, 4)], secs(5)).await;
        assert_eq!(r.unwrap_err(), SchedError::Overloaded);
        assert_eq!(s.models[0].permits.available_permits(), 4, "model a permits returned");
        assert_eq!(s.map.lock().unwrap().len(), 1, "only the busy entry is registered");
        busy.await.unwrap().unwrap();
        assert_eq!(s.models[1].permits.available_permits(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn joiners_take_no_permits() {
        let fake = Fake { delay: Duration::from_millis(100), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 2, 8, 8192)]).await;
        let first = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(1, 0, 4)], secs(5)).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let second = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(1, 0, 4)], secs(5)).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(s.models[0].permits.available_permits(), 1);
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        assert_eq!(s.models[0].permits.available_permits(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn backend_error_reaches_subscribers_and_cleans_up() {
        let fake = Fake { fail_with: Some("boom".into()), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 4, 8, 8192)]).await;
        assert_eq!(s.run_jobs(vec![job(1, 0, 4)], secs(5)).await.unwrap_err(), SchedError::Backend("boom".into()));
        assert!(s.map.lock().unwrap().is_empty());
        assert_eq!(s.models[0].permits.available_permits(), 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn panic_fails_only_that_batch_and_worker_recovers() {
        let fake = Fake { panic_token: Some(7), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 4, 8, 8192)]).await;
        let r = s.run_jobs(vec![job(7, 0, 4)], secs(5)).await;
        assert_eq!(r.unwrap_err(), SchedError::Backend("inference worker panicked".into()));
        assert!(s.run_jobs(vec![job(7, 0, 4)], secs(5)).await.is_ok(), "same key retried cleanly");
        assert_eq!(s.models[0].permits.available_permits(), 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn short_deadline_does_not_fail_long_deadline_joiner() {
        let fake = Fake { delay: Duration::from_millis(150), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 4, 8, 8192)]).await;
        let long = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(1, 0, 4)], secs(5)).await })
        };
        let short = s.run_jobs(vec![job(1, 0, 4)], Instant::now() + Duration::from_millis(30)).await;
        assert_eq!(short.unwrap_err(), SchedError::Deadline);
        assert!(long.await.unwrap().is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batches_respect_item_and_token_limits() {
        let fake = Fake { delay: Duration::from_millis(30), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 64, 3, 20)]).await;
        let jobs: Vec<Job> = (1..=10).map(|n| job(n, 0, 8)).collect();
        s.run_jobs(jobs, secs(5)).await.unwrap();
        let sizes = fake.batch_sizes.lock().unwrap().clone();
        assert_eq!(sizes.iter().sum::<usize>(), 10);
        assert!(sizes.iter().all(|&n| n <= 2), "8+8 fits 20 tokens, a third would not: {sizes:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn token_cap_counts_padding() {
        let fake = Fake { delay: Duration::from_millis(30), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 64, 8, 20)]).await;
        // 16 + 2 = 18 raw tokens, but padded together they are 2 × 16 = 32 > 20
        s.run_jobs(vec![job(1, 0, 16), job(2, 0, 2)], secs(5)).await.unwrap();
        assert_eq!(*fake.batch_sizes.lock().unwrap(), [1, 1]);
    }

    fn item(len: usize) -> WorkItem {
        let (tx, _) = oneshot::channel();
        WorkItem {
            key: [0; 32],
            generation: 0,
            enc: job(1, 0, len).enc,
            tx: Some(tx),
            deadline: Arc::new(Mutex::new(Instant::now())),
            map: Arc::default(),
            _permit: Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap(),
            tokens: 0,
            queued: Arc::default(),
        }
    }

    fn lens<'a>(items: impl IntoIterator<Item = &'a WorkItem>) -> Vec<usize> {
        items.into_iter().map(|i| i.enc.ids.len()).collect()
    }

    #[test]
    fn pick_groups_similar_lengths_oldest_first() {
        let mut buf: VecDeque<WorkItem> = [500, 12, 480, 11, 490].into_iter().map(item).collect();
        let batch = pick(item(10), &mut buf, 3, 8192);
        assert_eq!(lens(&batch), [10, 11, 12], "the oldest leads, then the closest lengths");
        assert_eq!(lens(&buf), [500, 480, 490], "the rest stay buffered in arrival order");
        let batch = pick(buf.pop_front().unwrap(), &mut buf, 3, 8192);
        assert_eq!(lens(&batch), [500, 490, 480]);
    }

    #[test]
    fn pick_respects_the_padded_token_cap() {
        let mut buf: VecDeque<WorkItem> = [12, 11].into_iter().map(item).collect();
        // 10 + 11 pad to 2 × 11 = 22 <= 30; adding 12 would pad to 3 × 12 = 36.
        assert_eq!(lens(&pick(item(10), &mut buf, 8, 30)), [10, 11]);
        assert_eq!(lens(&buf), [12]);
    }

    #[test]
    fn pick_with_one_item_cap_returns_only_the_first() {
        let mut buf: VecDeque<WorkItem> = [11, 12].into_iter().map(item).collect();
        assert_eq!(lens(&pick(item(10), &mut buf, 1, 8192)), [10]);
        assert_eq!(lens(&buf), [11, 12]);
    }

    #[test]
    fn pick_from_an_empty_buffer_returns_only_the_first() {
        assert_eq!(lens(&pick(item(10), &mut VecDeque::new(), 8, 8192)), [10]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn abandoned_items_are_skipped() {
        let fake = Fake { delay: Duration::from_millis(100), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 4, 8, 8192)]).await;
        let busy = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(1, 0, 4)], secs(5)).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let abandoned = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(2, 0, 4)], secs(5)).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        abandoned.abort(); // client disconnect while its item waits in the queue
        busy.await.unwrap().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(fake.items.load(SeqCst), 1);
        assert!(s.map.lock().unwrap().is_empty());
        assert_eq!(s.models[0].permits.available_permits(), 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dead_model_fails_fast_with_backend_error() {
        let fake = Fake { panic_token: Some(7), ..Fake::default() };
        let built = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let factory: BackendFactory =
            Arc::new(
                move || {
                    if built.swap(true, SeqCst) { Err("rebuild failed".into()) } else { Ok(Box::new(fake.clone())) }
                },
            );
        let s = sched(vec![ModelSpec { factory, ..spec("m", &Fake::default(), 4, 8, 8192) }]).await;
        let r = s.run_jobs(vec![job(7, 0, 4)], secs(5)).await;
        assert_eq!(r.unwrap_err(), SchedError::Backend("inference worker panicked".into()));
        assert!(!s.ready(), "a model with no live workers is not ready");
        let started = Instant::now();
        let r = s.run_jobs(vec![job(1, 0, 4)], secs(5)).await;
        assert_eq!(r.unwrap_err(), SchedError::Backend("model workers unavailable".into()));
        assert!(started.elapsed() < Duration::from_secs(1), "failed fast, not at the deadline");
        assert!(s.map.lock().unwrap().is_empty());
        assert_eq!(s.models[0].permits.available_permits(), 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unmeetable_deadline_is_shed_immediately() {
        let fake = Fake { delay: Duration::from_millis(100), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 64, 1, 8192)]).await;
        assert!(s.ready());
        s.run_jobs(vec![job(1, 0, 4)], secs(5)).await.unwrap(); // seeds the estimate: ~0.1 s/item
        let busy = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs((2..7).map(|n| job(n, 0, 4)).collect(), secs(5)).await })
        };
        wait_until(|| s.map.lock().unwrap().contains_key(&[2; 32])).await; // 5 items queued: ~0.5 s of work
        let started = Instant::now();
        let r = s.run_jobs(vec![job(9, 0, 4)], Instant::now() + Duration::from_millis(300)).await;
        assert_eq!(r.unwrap_err(), SchedError::Overloaded);
        assert!(started.elapsed() < Duration::from_millis(50), "shed at admission, not at the deadline");
        busy.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn own_work_counts_toward_the_deadline() {
        let fake = Fake { delay: Duration::from_millis(100), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 64, 1, 8192)]).await;
        s.run_jobs(vec![job(1, 0, 4)], secs(5)).await.unwrap(); // seeds the estimate: ~0.1 s per 4 tokens
        let busy = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(2, 0, 4)], secs(5)).await })
        };
        wait_until(|| s.map.lock().unwrap().contains_key(&[2; 32])).await; // 1 item ahead: ~0.1 s
        let started = Instant::now();
        // Ahead alone fits 300 ms; ahead plus these 5 items (~0.6 s) does not.
        let r = s.run_jobs((3..8).map(|n| job(n, 0, 4)).collect(), Instant::now() + Duration::from_millis(300)).await;
        assert_eq!(r.unwrap_err(), SchedError::Overloaded);
        assert!(started.elapsed() < Duration::from_millis(50), "shed at admission, not at the deadline");
        busy.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queued_tokens_return_to_zero() {
        let fake = Fake { delay: Duration::from_millis(400), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 64, 1, 8192)]).await;
        s.run_jobs(vec![job(1, 0, 400)], secs(5)).await.unwrap(); // success; low estimate: 1 ms/token
        let busy = {
            let s = s.clone();
            tokio::spawn(async move { s.run_jobs(vec![job(2, 0, 4)], secs(5)).await })
        };
        wait_until(|| s.map.lock().unwrap().contains_key(&[2; 32])).await;
        let r = s.run_jobs(vec![job(3, 0, 4)], Instant::now() + Duration::from_millis(50)).await;
        assert_eq!(r.unwrap_err(), SchedError::Deadline, "admitted, then stuck behind the busy batch");
        let r = s.run_jobs((4..9).map(|n| job(n, 0, 400)).collect(), Instant::now() + Duration::from_millis(300)).await;
        assert_eq!(r.unwrap_err(), SchedError::Overloaded);
        busy.await.unwrap().unwrap();
        // The dead item is dropped once the worker frees.
        wait_until(|| s.models[0].queued_tokens.load(Ordering::Relaxed) == 0 && s.map.lock().unwrap().is_empty()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn large_request_on_idle_queue_is_admitted_despite_high_estimate() {
        let fake = Fake { delay: Duration::from_millis(200), ..Fake::default() };
        let s = sched(vec![spec("m", &fake, 64, 8, 8192)]).await;
        s.run_jobs(vec![job(1, 0, 4)], secs(5)).await.unwrap(); // slow lone batch: ~0.2 s/item estimate
        let jobs: Vec<Job> = (2..10).map(|n| job(n, 0, 4)).collect(); // 8 × 0.2 s > 1 s, but it is one batch
        let r = s.run_jobs(jobs, Instant::now() + Duration::from_secs(1)).await;
        assert_eq!(r.map(|m| m.len()), Ok(8), "nothing queued ahead, so nothing to shed for");
    }

    #[tokio::test]
    async fn unknown_model_index_is_rejected() {
        let s = sched(vec![spec("m", &Fake::default(), 4, 8, 8192)]).await;
        assert!(matches!(s.run_jobs(vec![job(1, 5, 4)], secs(5)).await, Err(SchedError::Backend(_))));
        assert_eq!(s.run_jobs(vec![job(1, 0, 4)], secs(5)).await.map(|m| m.len()), Ok(1), "map lock not poisoned");
    }

    #[tokio::test]
    async fn stale_generation_does_not_remove_newer_entry() {
        let map: Map = Arc::default();
        let (_tx, rx) = oneshot::channel::<Outcome>();
        let shared = rx.shared();
        let dl = Arc::new(Mutex::new(Instant::now()));
        map.lock()
            .unwrap()
            .insert([1; 32], Entry { generation: 2, weak: shared.downgrade().unwrap(), deadline: dl.clone() });
        let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();
        let (tx_old, _rx_old) = oneshot::channel();
        let old = WorkItem {
            key: [1; 32],
            generation: 1,
            enc: job(1, 0, 1).enc,
            tx: Some(tx_old),
            deadline: dl,
            map: map.clone(),
            _permit: permit,
            tokens: 0,
            queued: Arc::default(),
        };
        drop(old);
        assert!(map.lock().unwrap().contains_key(&[1; 32]));
    }
}
