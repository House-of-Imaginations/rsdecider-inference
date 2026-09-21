//! Two-tier cache of raw model outputs: L1 moka (byte-weighted) + optional L2 Redis.
use crate::config::CacheCfg;
use crate::model::postprocess::Raw;
use crate::model::sequence::QType;
use redis::aio::ConnectionManager;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Duration;

pub type Key = [u8; 32];

/// Bump whenever sequence construction or the cached value format changes.
pub const CACHE_SCHEMA: &[u8] = b"rsdecider-cache-v1";

/// sha256(schema ‖ fingerprint ‖ qtype ‖ len(seg)‖seg ...), u64-LE lengths.
pub fn key(fingerprint: &[u8; 32], qtype: QType, segments: &[&str]) -> Key {
    let mut h = Sha256::new();
    h.update(CACHE_SCHEMA);
    h.update(fingerprint);
    h.update([qtype.id() as u8]);
    for s in segments {
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    h.finalize().into()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tier {
    L1,
    L2,
}

pub struct Cache {
    l1: moka::sync::Cache<Key, Arc<Raw>>,
    redis: Option<ConnectionManager>,
    l2_ttl_secs: u64,
    timeout: Duration,
}

pub fn redis_error(e: impl std::fmt::Display) {
    tracing::warn!("redis: {e}");
    metrics::counter!("rsdecider_redis_errors_total").increment(1);
}

impl Cache {
    /// Connects to Redis when `redis_url` is set (startup fails if it is unreachable).
    pub async fn new(cfg: &CacheCfg, redis_timeout: Duration) -> Result<Self, String> {
        let redis = match &cfg.redis_url {
            Some(url) => {
                let client = redis::Client::open(url.as_str()).map_err(|e| e.to_string())?;
                let con = tokio::time::timeout(Duration::from_secs(5), ConnectionManager::new(client))
                    .await
                    .map_err(|_| format!("redis {url}: connect timeout"))?
                    .map_err(|e| format!("redis {url}: {e}"))?;
                Some(con)
            }
            None => None,
        };
        let l1 = moka::sync::Cache::builder()
            .max_capacity(cfg.l1_max_bytes)
            .weigher(|_k: &Key, v: &Arc<Raw>| (96 + v.logits.len() * 4) as u32)
            .time_to_live(Duration::from_secs(cfg.l1_ttl_secs))
            .build();
        Ok(Self { l1, redis, l2_ttl_secs: cfg.l2_ttl_secs, timeout: redis_timeout })
    }

    pub fn redis(&self) -> Option<ConnectionManager> {
        self.redis.clone()
    }

    pub fn put_l1(&self, k: Key, v: Arc<Raw>) {
        self.l1.insert(k, v);
    }

    /// L1, then L2 (promoting hits to L1). Redis errors, and values that don't decode to exactly
    /// `n_logits` logits (a malformed L2 value would panic in postprocess), count as misses.
    pub async fn get(&self, k: &Key, n_logits: usize) -> Option<(Arc<Raw>, Tier)> {
        if let Some(v) = self.l1.get(k).filter(|v| v.logits.len() == n_logits) {
            return Some((v, Tier::L1));
        }
        let mut con = self.redis.clone()?;
        let mut cmd = redis::cmd("GET");
        cmd.arg(redis_key(k));
        let s = match tokio::time::timeout(self.timeout, cmd.query_async::<Option<String>>(&mut con)).await {
            Ok(Ok(s)) => s?,
            Ok(Err(e)) => {
                redis_error(e);
                return None;
            }
            Err(_) => {
                redis_error("GET timeout");
                return None;
            }
        };
        let raw: Arc<Raw> = match serde_json::from_str::<Raw>(&s) {
            Ok(r) if r.logits.len() == n_logits => Arc::new(r),
            Ok(r) => {
                redis_error(format!("malformed cached value: {} logits, want {n_logits}", r.logits.len()));
                return None;
            }
            Err(e) => {
                redis_error(format!("malformed cached value: {e}"));
                return None;
            }
        };
        self.l1.insert(*k, raw.clone());
        Some((raw, Tier::L2))
    }

    pub async fn put_l2(&self, k: Key, v: Arc<Raw>) {
        let Some(mut con) = self.redis.clone() else { return };
        let val = serde_json::to_string(&*v).expect("Raw serializes");
        let mut cmd = redis::cmd("SET");
        cmd.arg(redis_key(&k)).arg(val).arg("EX").arg(self.l2_ttl_secs);
        match tokio::time::timeout(self.timeout, cmd.query_async::<()>(&mut con)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => redis_error(e),
            Err(_) => redis_error("SET timeout"),
        }
    }
}

fn redis_key(k: &Key) -> String {
    format!("rsd:c:{}", hex::encode(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: [u8; 32] = [7; 32];

    #[test]
    fn key_covers_every_input() {
        let base = key(&FP, QType::Choice, &["state", "head", " a", " b"]);
        assert_ne!(base, key(&[8; 32], QType::Choice, &["state", "head", " a", " b"]), "fingerprint");
        assert_ne!(base, key(&FP, QType::Score, &["state", "head", " a", " b"]), "qtype");
        assert_ne!(base, key(&FP, QType::Choice, &["state2", "head", " a", " b"]), "state");
        assert_ne!(base, key(&FP, QType::Choice, &["state", "head", " b", " a"]), "option order");
        assert_eq!(base, key(&FP, QType::Choice, &["state", "head", " a", " b"]));
    }

    #[test]
    fn key_is_length_prefixed() {
        assert_ne!(key(&FP, QType::Noul, &["ab", "c"]), key(&FP, QType::Noul, &["a", "bc"]));
    }

    #[tokio::test]
    async fn l1_roundtrip_without_redis() {
        let c = Cache::new(&CacheCfg::default(), Duration::from_millis(250)).await.unwrap();
        let k = key(&FP, QType::Noul, &["x"]);
        assert!(c.get(&k, 2).await.is_none());
        c.put_l1(k, Arc::new(Raw { logits: vec![0.1, 0.2], act_prob: 0.5, n_tokens: 3 }));
        let (v, tier) = c.get(&k, 2).await.unwrap();
        assert_eq!((v.n_tokens, tier), (3, Tier::L1));
    }

    #[tokio::test]
    async fn wrong_length_raw_is_a_miss() {
        let c = Cache::new(&CacheCfg::default(), Duration::from_millis(250)).await.unwrap();
        let k = key(&FP, QType::Choice, &["x"]);
        c.put_l1(k, Arc::new(Raw { logits: vec![0.1, 0.2], act_prob: 0.5, n_tokens: 3 }));
        assert!(c.get(&k, 3).await.is_none());
    }
}
