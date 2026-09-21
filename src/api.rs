//! HTTP layer: request/response types, handlers, error mapping, app wiring.
use crate::auth::Auth;
use crate::cache::{self, Cache, Key, Tier};
use crate::config::{Config, ModelCfg};
use crate::idempotency::{Begin, Idem};
use crate::lang;
use crate::model::LoadedModel;
use crate::model::postprocess::{Raw, answer};
use crate::model::sequence::{Prepared, QuestionDef, encode, prepare, state_segment};
use crate::scheduler::{BackendFactory, Job, ModelSpec, SchedError, Scheduler};
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::Instrument;

#[derive(Debug, Deserialize)]
pub struct DecideReq {
    pub state: Value,
    #[serde(default)]
    pub model: Option<String>,
    pub questions: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct BatchReq {
    pub items: Vec<DecideReq>,
}

#[derive(Debug)]
pub enum ApiError {
    Unauthorized,
    InProgress(Duration),
    TooLarge,
    Invalid(String),
    RateLimited(Duration),
    Overloaded(Duration),
    Deadline,
    Internal(String),
}

impl ApiError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::InProgress(_) => StatusCode::CONFLICT,
            Self::TooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Invalid(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::RateLimited(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Overloaded(_) => StatusCode::from_u16(529).unwrap(),
            Self::Deadline => StatusCode::GATEWAY_TIMEOUT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (code, msg, retry) = match &self {
            Self::Unauthorized => ("unauthorized", "missing or unknown API key".to_string(), None),
            Self::InProgress(d) => {
                ("idempotency_in_progress", "a request with this Idempotency-Key is in progress".into(), Some(*d))
            }
            Self::TooLarge => ("payload_too_large", "request body too large".into(), None),
            Self::Invalid(m) => ("invalid_request", m.clone(), None),
            Self::RateLimited(d) => ("rate_limited", "rate limit exceeded".into(), Some(*d)),
            Self::Overloaded(d) => ("overloaded", "server is at capacity, retry later".into(), Some(*d)),
            Self::Deadline => ("deadline_exceeded", "request timed out".into(), None),
            Self::Internal(_) => ("internal", "internal error".into(), None),
        };
        let mut r = (self.status(), Json(json!({"error": {"code": code, "message": msg}}))).into_response();
        if let Some(d) = retry {
            let secs = d.as_secs_f64().ceil().max(1.0) as u64;
            r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        r
    }
}

pub struct AppState {
    pub cfg: Config,
    pub models: Vec<Arc<LoadedModel>>,
    pub by_name: HashMap<String, usize>,
    pub english: usize,
    pub non_english: usize,
    pub cache: Arc<Cache>,
    pub sched: Scheduler,
    pub auth: Arc<Auth>,
    pub idem: Idem,
    pub tokenize: Arc<Semaphore>,
    /// Requests tokenizing or waiting to; when empty, new work is shed with 529.
    pub tokenize_queue: Arc<Semaphore>,
}

/// Loads every model, connects the cache, starts the scheduler. `make_backend` picks ORT or Fake.
pub async fn build(
    cfg: Config,
    make_backend: &dyn Fn(&ModelCfg, &LoadedModel) -> BackendFactory,
) -> Result<Arc<AppState>, String> {
    let mut models = Vec::new();
    let mut specs = Vec::new();
    for m in &cfg.models {
        let lm = LoadedModel::load(&m.name, &m.path)?;
        if m.max_batch_tokens < lm.meta.max_len {
            return Err(format!(
                "model {:?}: max_batch_tokens {} < laya max_len {}",
                m.name, m.max_batch_tokens, lm.meta.max_len
            ));
        }
        specs.push(ModelSpec {
            name: m.name.clone(),
            workers: m.workers,
            max_pending: m.max_pending,
            max_batch_items: m.max_batch_items,
            max_batch_tokens: m.max_batch_tokens,
            max_wait: Duration::from_millis(m.max_wait_ms),
            pad_id: lm.meta.pad_id,
            factory: make_backend(m, &lm),
        });
        models.push(Arc::new(lm));
    }
    let by_name: HashMap<String, usize> = cfg.models.iter().enumerate().map(|(i, m)| (m.name.clone(), i)).collect();
    let cache = Arc::new(Cache::new(&cfg.cache, cfg.knobs.redis_timeout()).await?);
    let sched = Scheduler::start(specs, cache.clone())?;
    let idem = Idem::new(
        cache.redis(),
        Duration::from_millis(cfg.server.request_timeout_ms),
        Duration::from_secs(cfg.cache.idempotency_ttl_secs),
        &cfg.knobs,
    );
    Ok(Arc::new(AppState {
        english: by_name[&cfg.routing.english_model],
        non_english: by_name[&cfg.routing.non_english_model],
        auth: Arc::new(Auth::new(&cfg.keys)),
        tokenize: Arc::new(Semaphore::new(cfg.server.tokenize_threads.max(1))),
        // default: the scheduler queues' memory knob; requests past Σ max_pending can't be admitted anyway
        tokenize_queue: Arc::new(Semaphore::new(
            cfg.knobs.tokenize_queue.unwrap_or_else(|| cfg.models.iter().map(|m| m.max_pending).sum()),
        )),
        models,
        by_name,
        cache,
        sched,
        idem,
        cfg,
    }))
}

const OPENAPI: &str = include_str!("../openapi.yaml");
const DOCS: &str = include_str!("docs.html");

pub fn router(st: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/decide", post(decide))
        .route("/v1/decide/batch", post(decide_batch))
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(readyz))
        .route("/openapi.yaml", get(|| async { ([(header::CONTENT_TYPE, "application/yaml")], OPENAPI) }))
        .route("/docs", get(|| async { Html(DOCS) }))
        .layer(DefaultBodyLimit::max(st.cfg.limits.max_body_bytes))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(st)
}

/// Models load before the listener binds; after that, ready while every model has a live worker.
async fn readyz(State(st): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    if st.sched.ready() { (StatusCode::OK, "ready") } else { (StatusCode::SERVICE_UNAVAILABLE, "not ready") }
}

async fn decide(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Result<Bytes, BytesRejection>) -> Response {
    handle(st, headers, body, false).await
}

async fn decide_batch(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    handle(st, headers, body, true).await
}

async fn handle(st: Arc<AppState>, headers: HeaderMap, body: Result<Bytes, BytesRejection>, batch: bool) -> Response {
    let started = Instant::now();
    let id = format!("req_{}", ulid::Ulid::generate());
    let deadline = tokio::time::Instant::now() + Duration::from_millis(st.cfg.server.request_timeout_ms);
    let mut key_name = "-".to_string();
    let result = run(&st, &headers, body, batch, &id, deadline, &mut key_name)
        .instrument(tracing::info_span!("request", request_id = %id))
        .await;
    let status = result.as_ref().map_or_else(|e| e.status(), |_| StatusCode::OK);
    if let Err(ApiError::Internal(m)) = &result {
        tracing::error!(request_id = %id, "internal error: {m}");
    }
    metrics::counter!("rsdecider_requests_total", "key" => key_name, "status" => status.as_u16().to_string())
        .increment(1);
    metrics::histogram!("rsdecider_request_seconds").record(started.elapsed().as_secs_f64());
    // An idempotent replay carries the original request's id; the header must match the body.
    let rid = result.as_ref().ok().and_then(|v| v["id"].as_str()).and_then(|s| HeaderValue::from_str(s).ok());
    let rid = rid.unwrap_or_else(|| HeaderValue::from_str(&id).unwrap());
    let mut resp = match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => e.into_response(),
    };
    resp.headers_mut().insert("x-request-id", rid);
    resp
}

async fn run(
    st: &AppState,
    headers: &HeaderMap,
    body: Result<Bytes, BytesRejection>,
    batch: bool,
    id: &str,
    deadline: tokio::time::Instant,
    key_name: &mut String,
) -> Result<Value, ApiError> {
    let auth = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok());
    let key = st.auth.lookup(auth).ok_or(ApiError::Unauthorized)?;
    *key_name = key.cfg.name.clone();
    let body = body.map_err(|r| match r.status() {
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::TooLarge,
        _ => ApiError::Invalid(r.body_text()),
    })?;
    let items = if batch {
        let b: BatchReq = serde_json::from_slice(&body).map_err(|e| ApiError::Invalid(e.to_string()))?;
        if b.items.is_empty() || b.items.len() > st.cfg.limits.max_batch_states {
            return Err(ApiError::Invalid(format!("items must hold 1-{} states", st.cfg.limits.max_batch_states)));
        }
        b.items
    } else {
        vec![serde_json::from_slice(&body).map_err(|e| ApiError::Invalid(e.to_string()))?]
    };
    key.check(items.len() as u32).map_err(ApiError::RateLimited)?;

    let ik = headers.get("idempotency-key").map(|v| v.to_str()).transpose();
    let ik = ik.map_err(|_| ApiError::Invalid("Idempotency-Key must be visible ASCII".into()))?;
    let guard = match ik {
        None => None,
        Some(ik) => {
            // the route is part of the digest: one body can parse on both endpoints
            let route: &[u8] = if batch { b"batch\n" } else { b"decide\n" };
            let body_sha =
                hex::encode(<[u8; 32]>::from(Sha256::new().chain_update(route).chain_update(&body).finalize()));
            let begin = tokio::time::timeout_at(deadline, st.idem.begin(&key.cfg.name, ik, &body_sha, id));
            match begin.await.map_err(|_| ApiError::Deadline)? {
                Begin::Proceed(g) => Some(g),
                Begin::Replay(v) => return Ok(v),
                Begin::InProgress(d) => return Err(ApiError::InProgress(d)),
                Begin::Mismatch => {
                    return Err(ApiError::Invalid("Idempotency-Key reused with a different body".into()));
                }
            }
        }
    };
    let res = decide_items(st, &items, batch, deadline, &key.cfg.name).await.map(|mut results| {
        let mut out = Map::new();
        out.insert("id".into(), json!(id));
        if batch {
            out.insert("results".into(), Value::Array(results));
        } else if let Value::Object(m) = results.remove(0) {
            out.extend(m);
        }
        Value::Object(out)
    });
    if let Some(g) = guard {
        st.idem.finish(g, res.as_ref().ok()).await;
    }
    res
}

struct Q {
    item: usize,
    qid: String,
    prep: Prepared,
    key: Key,
}

async fn decide_items(
    st: &AppState,
    items: &[DecideReq],
    batch: bool,
    deadline: tokio::time::Instant,
    key_name: &str,
) -> Result<Vec<Value>, ApiError> {
    let t0 = Instant::now();
    let lim = &st.cfg.limits;
    let at = |i: usize| if batch { format!("items[{i}].") } else { String::new() };
    let invalid = |i: usize, m: String| ApiError::Invalid(format!("{}{m}", at(i)));

    // Step 1+2: validate, route, build segments, cache keys.
    let mut routes: Vec<(usize, &str)> = Vec::new();
    let mut states: Vec<String> = Vec::new();
    let mut qs: Vec<Q> = Vec::new();
    for (i, it) in items.iter().enumerate() {
        if !matches!(it.state, Value::String(_) | Value::Object(_) | Value::Array(_)) {
            return Err(invalid(i, "state must be a string, object or array".into()));
        }
        if it.questions.is_empty() || it.questions.len() > lim.max_questions_per_state {
            return Err(invalid(i, format!("questions must hold 1-{} entries", lim.max_questions_per_state)));
        }
        let (m, reason) = match &it.model {
            Some(n) => (*st.by_name.get(n).ok_or_else(|| invalid(i, format!("unknown model {n:?}")))?, "explicit"),
            None if lang::is_english(&it.state) => (st.english, "detected:english"),
            None => (st.non_english, "detected:non_english"),
        };
        let model = &st.models[m];
        let seg = state_segment(&it.state, &model.meta.mask_token);
        if seg.chars().count() > lim.max_state_chars {
            return Err(invalid(i, format!("state exceeds {} characters", lim.max_state_chars)));
        }
        for (qid, qv) in &it.questions {
            let def: QuestionDef =
                serde_json::from_value(qv.clone()).map_err(|e| invalid(i, format!("questions.{qid}: {e}")))?;
            let prep =
                prepare(&def, &model.meta.mask_token).map_err(|e| invalid(i, format!("questions.{qid}: {e}")))?;
            let mut segs = vec![seg.as_str(), prep.head.as_str()];
            segs.extend(prep.options.iter().map(String::as_str));
            let key = cache::key(&model.fingerprint, prep.qtype, &segs);
            qs.push(Q { item: i, qid: qid.clone(), prep, key });
        }
        routes.push((m, reason));
        states.push(seg);
    }

    // Step 3: cache lookup.
    let looked = futures::future::join_all(qs.iter().map(|q| st.cache.get(&q.key, q.prep.options.len()))).await;
    let mut raws: HashMap<Key, (Arc<Raw>, bool)> = HashMap::new();
    for (q, hit) in qs.iter().zip(looked) {
        let tier = match hit {
            Some((r, t)) => {
                raws.insert(q.key, (r, true));
                if t == Tier::L1 { "l1" } else { "l2" }
            }
            None => "miss",
        };
        metrics::counter!("rsdecider_cache_total", "tier" => tier).increment(1);
    }

    // Step 3b: tokenize misses (state once per item, each unique key once).
    let mut jobs = Vec::new();
    let mut seen = HashSet::new();
    for (i, seg) in states.iter().enumerate() {
        let missing: Vec<&Q> =
            qs.iter().filter(|q| q.item == i && !raws.contains_key(&q.key) && seen.insert(q.key)).collect();
        if missing.is_empty() {
            continue;
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(ApiError::Deadline);
        }
        let model = st.models[routes[i].0].clone();
        let (seg, preps) = (seg.clone(), missing.iter().map(|q| q.prep.clone()).collect::<Vec<_>>());
        // bounded wait: shed instead of holding the parsed request until its deadline
        let queued = st.tokenize_queue.clone().try_acquire_owned().map_err(|_| {
            metrics::counter!("rsdecider_tokenize_shed_total").increment(1);
            ApiError::Overloaded(st.cfg.knobs.retry_after())
        })?;
        let permit = tokio::time::timeout_at(deadline, st.tokenize.clone().acquire_owned())
            .await
            .map_err(|_| ApiError::Deadline)?
            .expect("semaphore never closed");
        // The permit moves into the closure so it is held until tokenization really ends, even if
        // the request is dropped or times out first (keeps concurrency <= tokenize_threads).
        let encoded = tokio::time::timeout_at(
            deadline,
            tokio::task::spawn_blocking(move || {
                let _permits = (permit, queued);
                let ids = model.tokenize_prefix(&seg, model.meta.max_len)?;
                Ok::<_, String>(preps.iter().map(|p| encode(&model, &ids, p)).collect::<Vec<_>>())
            }),
        )
        .await
        .map_err(|_| ApiError::Deadline)?
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .map_err(ApiError::Internal)?;
        for (q, enc) in missing.iter().zip(encoded) {
            let enc = enc.map_err(|e| invalid(i, format!("questions.{}: {e}", q.qid)))?;
            jobs.push(Job { key: q.key, model: routes[i].0, enc: Arc::new(enc) });
        }
    }
    let t_tok = t0.elapsed();

    // Steps 4-7: admission, batching, inference.
    if !jobs.is_empty() {
        let computed = st.sched.run_jobs(jobs, deadline).await.map_err(|e| match e {
            SchedError::Overloaded => ApiError::Overloaded(st.cfg.knobs.retry_after()),
            SchedError::Deadline => ApiError::Deadline,
            SchedError::Backend(m) => ApiError::Internal(m),
        })?;
        raws.extend(computed.into_iter().map(|(k, r)| (k, (r, false))));
    }
    let t_wait = t0.elapsed() - t_tok;

    let ms = |d: Duration| (d.as_secs_f64() * 1e4).round() / 10.0;
    let mut out = Vec::with_capacity(items.len());
    for (i, &(m, reason)) in routes.iter().enumerate() {
        let model = &st.models[m];
        let mut answers = Map::new();
        let (mut tokens, mut cached_tokens) = (0u64, 0u64);
        for q in qs.iter().filter(|q| q.item == i) {
            let (raw, cached) = &raws[&q.key];
            let mut a = answer(&q.prep, raw, &model.meta);
            a["cached"] = json!(cached);
            answers.insert(q.qid.clone(), a);
            tokens += raw.n_tokens as u64;
            if *cached {
                cached_tokens += raw.n_tokens as u64;
            }
        }
        metrics::counter!("rsdecider_tokens_total", "key" => key_name.to_string(), "source" => "cached")
            .increment(cached_tokens);
        metrics::counter!("rsdecider_tokens_total", "key" => key_name.to_string(), "source" => "computed")
            .increment(tokens - cached_tokens);
        out.push(json!({
            "model": model.display_name(),
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "routing": {"model": model.name, "reason": reason},
            "timing_ms": {"tokenize": ms(t_tok), "wait": ms(t_wait), "total": ms(t0.elapsed())},
        }));
    }
    Ok(out)
}
