//! Idempotency-Key records: Redis when available, in-process map otherwise (single instance only).
use crate::cache::{REDIS_TIMEOUT, redis_error};
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
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
    local: Mutex<HashMap<String, (Record, Instant)>>,
    pending_ttl: Duration,
    done_ttl: Duration,
}

const DEL_IF_OWNER: &str = r"local v = redis.call('GET', KEYS[1])
if v and cjson.decode(v).owner == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0";

impl Idem {
    pub fn new(redis: Option<ConnectionManager>, request_timeout: Duration, done_ttl: Duration) -> Self {
        Self { redis, local: Mutex::default(), pending_ttl: request_timeout + Duration::from_secs(5), done_ttl }
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
        let mut map = self.local.lock().unwrap();
        if map.len() > 100_000 {
            map.retain(|_, (_, exp)| *exp > now); // ponytail: lazy sweep instead of a janitor task
        }
        match map.get(&rec_key) {
            Some((rec, exp)) if *exp > now => existing(rec, body_sha, *exp - now),
            _ => {
                map.insert(rec_key.clone(), (pending, now + self.pending_ttl));
                Begin::Proceed(guard(true))
            }
        }
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
    pub async fn finish(&self, g: Guard, response: Option<&Value>) {
        if !g.local
            && let Some(mut con) = self.redis.clone()
        {
            let res = match response {
                Some(r) => {
                    let rec = Record::Done { body_sha: g.body_sha.clone(), response: r.clone() };
                    let mut set = redis::cmd("SET");
                    set.arg(&g.rec_key)
                        .arg(serde_json::to_string(&rec).unwrap())
                        .arg("PX")
                        .arg(self.done_ttl.as_millis() as u64);
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
        let mut map = self.local.lock().unwrap();
        match response {
            Some(r) => {
                let rec = Record::Done { body_sha: g.body_sha, response: r.clone() };
                map.insert(g.rec_key, (rec, Instant::now() + self.done_ttl));
            }
            None => {
                if matches!(map.get(&g.rec_key), Some((Record::Pending { owner, .. }, _)) if *owner == g.owner) {
                    map.remove(&g.rec_key);
                }
            }
        }
    }
}

fn existing(rec: &Record, body_sha: &str, remaining: Duration) -> Begin {
    match rec {
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
}
