//! Bearer-key lookup (SHA-256 of the key) and per-key rate limiters, swappable on SIGHUP.
use crate::config::KeyCfg;
use arc_swap::ArcSwap;
use governor::clock::{Clock, DefaultClock};
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

pub struct KeyRecord {
    pub cfg: KeyCfg,
    limiter: DefaultDirectRateLimiter,
}

impl KeyRecord {
    fn new(cfg: KeyCfg) -> Self {
        let q = Quota::per_second(NonZeroU32::new(cfg.rps).expect("validated rps >= 1"))
            .allow_burst(NonZeroU32::new(cfg.burst.max(1)).unwrap());
        Self { limiter: RateLimiter::direct(q), cfg }
    }

    /// Consume `n` cells; Err carries the Retry-After duration.
    pub fn check(&self, n: u32) -> Result<(), Duration> {
        match self.limiter.check_n(NonZeroU32::new(n.max(1)).unwrap()) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(not_until)) => Err(not_until.wait_time_from(DefaultClock::default().now())),
            Err(_) => Err(Duration::from_secs(1)), // n > burst; startup validation prevents this
        }
    }
}

pub fn hash_key(plain: &str) -> String {
    let d: [u8; 32] = Sha256::digest(plain.as_bytes()).into();
    hex::encode(d)
}

pub struct Auth {
    by_hash: ArcSwap<HashMap<String, Arc<KeyRecord>>>,
}

impl Auth {
    pub fn new(keys: &[KeyCfg]) -> Self {
        let a = Self { by_hash: ArcSwap::from_pointee(HashMap::new()) };
        a.reload(keys);
        a
    }

    /// Replace the key set; unchanged keys keep their limiter state.
    pub fn reload(&self, keys: &[KeyCfg]) {
        let old = self.by_hash.load();
        let next = keys
            .iter()
            .map(|k| {
                let rec = match old.get(&k.sha256) {
                    Some(r) if r.cfg == *k => r.clone(),
                    _ => Arc::new(KeyRecord::new(k.clone())),
                };
                (k.sha256.clone(), rec)
            })
            .collect();
        self.by_hash.store(Arc::new(next));
    }

    pub fn lookup(&self, authorization: Option<&str>) -> Option<Arc<KeyRecord>> {
        let key = authorization?.strip_prefix("Bearer ")?.trim();
        self.by_hash.load().get(&hash_key(key)).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(rps: u32, burst: u32) -> KeyCfg {
        KeyCfg { name: "acme".into(), sha256: hash_key("test"), rps, burst }
    }

    #[test]
    fn hash_matches_sha256sum() {
        assert_eq!(hash_key("test"), "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08");
    }

    #[test]
    fn lookup_requires_bearer_and_known_key() {
        let a = Auth::new(&[key(1, 8)]);
        assert!(a.lookup(Some("Bearer test")).is_some());
        assert!(a.lookup(Some("Bearer nope")).is_none());
        assert!(a.lookup(Some("test")).is_none());
        assert!(a.lookup(None).is_none());
    }

    #[test]
    fn limiter_rejects_after_burst_and_reload_keeps_state() {
        let a = Auth::new(&[key(1, 8)]);
        let rec = a.lookup(Some("Bearer test")).unwrap();
        assert!(rec.check(8).is_ok());
        assert!(rec.check(1).unwrap_err() > Duration::ZERO);
        a.reload(&[key(1, 8)]);
        assert!(a.lookup(Some("Bearer test")).unwrap().check(1).is_err(), "same config keeps limiter");
        a.reload(&[key(1, 9)]);
        assert!(a.lookup(Some("Bearer test")).unwrap().check(1).is_ok(), "changed config resets limiter");
        a.reload(&[]);
        assert!(a.lookup(Some("Bearer test")).is_none(), "removed key");
    }
}
