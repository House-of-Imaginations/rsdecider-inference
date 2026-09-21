//! Idempotency-Key records: Redis when available, in-process map otherwise (single instance only).
use crate::cache::{REDIS_TIMEOUT, redis_error};
use moka::ops::compute::Op;
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "status", rename_all = "lowercase")]
enum Record {
    Pending { body_sha: String, owner: String },
    Done { body_sha: String, response: Value },
}

pub enum Begin {
    Proceed(Guard),
    Replay(Value),
    InProgress(Duration),
    Mismatch,
}

/// Returned with `Begin::Proceed`; hand it back to `finish`.
pub struct Guard {
    rec_key: String,
    owner: String,
    body_sha: String,
    local: bool,
}

pub struct Idem {
    redis: Option<ConnectionManager>,
    local: moka::sync::Cache<String, Entry>,
    pending_ttl: Duration,
    done_ttl: Duration,
}

/// Record, its deadline, and its weight in bytes (the serialized size).
type Entry = (Record, Instant, u32);

/// Bytes the in-process map may hold across all keys.
const LOCAL_MAX_BYTES: u64 = 64 << 20;
/// Larger responses are not stored: a repeat runs again instead of pinning memory.
pub const MAX_STORED_BYTES: usize = 256 << 10;
/// Weight of a pending record (hashes + owner id).
const PENDING_BYTES: u32 = 160;

const DEL_IF_OWNER: &str = r"local v = redis.call('GET', KEYS[1])
if v and cjson.decode(v).owner == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0";

impl Idem {
    pub fn new(redis: Option<ConnectionManager>, request_timeout: Duration, done_ttl: Duration) -> Self {
        let local = moka::sync::Cache::builder()
            .max_capacity(LOCAL_MAX_BYTES)
            .weigher(|k: &String, v: &Entry| (k.len() as u32).saturating_add(v.2))
            .expire_after(ExpiresAt)
            .build();
        Self { redis, local, pending_ttl: request_timeout + Duration::from_secs(5), done_ttl }
    }

    pub async fn begin(&self, key_name: &str, idem_key: &str, body_sha: &str, owner: &str) -> Begin {
        let rec_key = format!("rsd:idem:{key_name}:{}", crate::auth::hash_key(idem_key));
        let pending = Record::Pending { body_sha: body_sha.into(), owner: owner.into() };
        let guard = |local| Guard { rec_key: rec_key.clone(), owner: owner.into(), body_sha: body_sha.into(), local };
        if let Some(con) = &self.redis {
            match self.begin_redis(con.clone(), &rec_key, &pending, body_sha).await {
                Ok(Some(b)) => return b,
                Ok(None) => return Begin::Proceed(guard(false)),
                Err(e) => redis_error(e), // fall through to the in-process map
            }
        }
        let now = Instant::now();
        let e = self.local.entry(rec_key.clone()).or_insert_with(|| (pending, now + self.pending_ttl, PENDING_BYTES));
        if e.is_fresh() {
            return Begin::Proceed(guard(true));
        }
        let (rec, exp, _) = e.into_value();
        existing(&rec, body_sha, exp.saturating_duration_since(now))
    }

    /// Ok(None) = we own a fresh pending record.
    async fn begin_redis(
        &self,
        mut con: ConnectionManager,
        k: &str,
        pending: &Record,
        body_sha: &str,
    ) -> Result<Option<Begin>, String> {
        let val = serde_json::to_string(pending).unwrap();
        for _ in 0..3 {
            let mut set = redis::cmd("SET");
            set.arg(k).arg(&val).arg("NX").arg("PX").arg(self.pending_ttl.as_millis() as u64);
            if timed(set.query_async::<Option<String>>(&mut con)).await?.is_some() {
                return Ok(None);
            }
            let mut get = redis::cmd("GET");
            get.arg(k);
            let mut pttl = redis::cmd("PTTL");
            pttl.arg(k);
            let cur = timed(get.query_async::<Option<String>>(&mut con)).await?;
            let ttl_ms = timed(pttl.query_async::<i64>(&mut con)).await?;
            if let Some(s) = cur {
                let rec: Record = serde_json::from_str(&s).map_err(|e| e.to_string())?;
                return Ok(Some(existing(&rec, body_sha, Duration::from_millis(ttl_ms.max(1000) as u64))));
            } // expired between SET and GET: try again
        }
        Err("idempotency record kept vanishing".into())
    }

    /// `response` = Some(body) on 200 (stored for replay), None on any error (record released).
    /// A response over `MAX_STORED_BYTES` is treated like an error: released, not stored.
    pub async fn finish(&self, g: Guard, response: Option<&Value>) {
        let done = response
            .map(|r| Record::Done { body_sha: g.body_sha.clone(), response: r.clone() })
            .map(|rec| {
                let json = serde_json::to_string(&rec).unwrap();
                (rec, json)
            })
            .filter(|(_, json)| json.len() <= MAX_STORED_BYTES);
        if !g.local
            && let Some(mut con) = self.redis.clone()
        {
            let res = match done {
                Some((_, json)) => {
                    let mut set = redis::cmd("SET");
                    set.arg(&g.rec_key).arg(json).arg("PX").arg(self.done_ttl.as_millis() as u64);
                    timed(set.query_async::<()>(&mut con)).await
                }
                None => {
                    let script = redis::Script::new(DEL_IF_OWNER);
                    let mut inv = script.key(&g.rec_key);
                    inv.arg(&g.owner);
                    timed(inv.invoke_async::<i64>(&mut con)).await.map(|_| ())
                }
            };
            if let Err(e) = res {
                redis_error(e);
            }
            return;
        }
        match done {
            Some((rec, json)) => {
                self.local.insert(g.rec_key, (rec, Instant::now() + self.done_ttl, json.len() as u32));
            }
            None => {
                self.local.entry(g.rec_key).and_compute_with(|cur| match cur.map(|e| e.into_value()) {
                    Some((Record::Pending { owner, .. }, _, _)) if owner == g.owner => Op::Remove,
                    _ => Op::Nop,
                });
            }
        }
    }
}

/// Per-entry expiry: each record carries its own deadline (pending vs done TTL).
struct ExpiresAt;

impl moka::Expiry<String, Entry> for ExpiresAt {
    fn expire_after_create(&self, _: &String, v: &Entry, now: Instant) -> Option<Duration> {
        Some(v.1.saturating_duration_since(now))
    }

    fn expire_after_update(&self, _: &String, v: &Entry, now: Instant, _: Option<Duration>) -> Option<Duration> {
        Some(v.1.saturating_duration_since(now))
    }
}

fn existing(rec: &Record, body_sha: &str, remaining: Duration) -> Begin {
    match rec {
        Record::Pending { body_sha: b, .. } if b != body_sha => Begin::Mismatch,
        Record::Pending { .. } => Begin::InProgress(remaining),
        Record::Done { body_sha: b, response } if b == body_sha => Begin::Replay(response.clone()),
        Record::Done { .. } => Begin::Mismatch,
    }
}

async fn timed<T>(f: impl Future<Output = redis::RedisResult<T>>) -> Result<T, String> {
    match tokio::time::timeout(REDIS_TIMEOUT, f).await {
        Ok(r) => r.map_err(|e| e.to_string()),
        Err(_) => Err("timeout".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn idem() -> Idem {
        Idem::new(None, Duration::from_secs(10), Duration::from_secs(60))
    }

    #[tokio::test]
    async fn local_flow_pending_replay_mismatch() {
        let i = idem();
        let Begin::Proceed(g) = i.begin("acme", "k1", "sha-a", "req1").await else { panic!() };
        assert!(
            matches!(i.begin("acme", "k1", "sha-a", "req2").await, Begin::InProgress(d) if d > Duration::from_secs(10))
        );
        assert!(
            matches!(i.begin("acme", "k1", "sha-b", "req2b").await, Begin::Mismatch),
            "different body while pending"
        );
        i.finish(g, Some(&json!({"ok": 1}))).await;
        assert!(matches!(i.begin("acme", "k1", "sha-a", "req3").await, Begin::Replay(v) if v == json!({"ok": 1})));
        assert!(matches!(i.begin("acme", "k1", "sha-b", "req4").await, Begin::Mismatch));
        assert!(matches!(i.begin("other", "k1", "sha-b", "req5").await, Begin::Proceed(_)), "scoped per key name");
    }

    #[tokio::test]
    async fn errors_release_the_record() {
        let i = idem();
        let Begin::Proceed(g) = i.begin("acme", "k1", "sha-a", "req1").await else { panic!() };
        i.finish(g, None).await;
        assert!(matches!(i.begin("acme", "k1", "sha-a", "req2").await, Begin::Proceed(_)));
    }

    #[tokio::test]
    async fn oversized_responses_are_released_not_stored() {
        let i = idem();
        let Begin::Proceed(g) = i.begin("acme", "k1", "sha-a", "req1").await else { panic!() };
        i.finish(g, Some(&json!("x".repeat(MAX_STORED_BYTES)))).await;
        assert!(matches!(i.begin("acme", "k1", "sha-a", "req2").await, Begin::Proceed(_)));
    }

    #[tokio::test]
    async fn local_map_is_bounded_by_bytes() {
        let i = idem();
        let big = json!("x".repeat(MAX_STORED_BYTES - 100));
        for n in 0..(2 * LOCAL_MAX_BYTES as usize / MAX_STORED_BYTES) {
            let Begin::Proceed(g) = i.begin("acme", &format!("k{n}"), "sha", "req").await else { panic!() };
            i.finish(g, Some(&big)).await;
        }
        i.local.run_pending_tasks();
        assert!(i.local.weighted_size() <= LOCAL_MAX_BYTES, "{} bytes held", i.local.weighted_size());
    }

    #[tokio::test]
    async fn pending_record_expires_on_its_own_ttl() {
        let mut i = idem();
        i.pending_ttl = Duration::from_millis(50);
        let Begin::Proceed(_abandoned) = i.begin("acme", "k1", "sha-a", "req1").await else { panic!() };
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(matches!(i.begin("acme", "k1", "sha-a", "req2").await, Begin::Proceed(_)));
    }
}
