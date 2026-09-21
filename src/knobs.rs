//! Advanced limits, `[knobs]` in rsdecider.toml. The defaults suit most deployments; every field is optional.
use serde::Deserialize;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Knobs {
    /// Requests tokenizing or waiting to; full → 529. None = Σ models.max_pending.
    pub tokenize_queue: Option<usize>,
    /// `Retry-After` seconds sent with 529.
    pub overloaded_retry_after_secs: u64,
    /// Per-command Redis timeout; on expiry the cache misses and idempotency falls back to local.
    pub redis_timeout_ms: u64,
    /// Bytes the in-process idempotency store may hold (used when Redis is off or failing).
    pub idempotency_local_max_bytes: u64,
    /// Largest response stored for replay; larger ones re-run on repeat.
    pub idempotency_max_stored_bytes: usize,
    /// One ORT intra-op thread pool of this size shared by every session; per-model intra_op_threads is then ignored.
    pub ort_global_threads: Option<usize>,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            tokenize_queue: None,
            overloaded_retry_after_secs: 1,
            redis_timeout_ms: 250,
            idempotency_local_max_bytes: 64 << 20,
            idempotency_max_stored_bytes: 256 << 10,
            ort_global_threads: None,
        }
    }
}

impl Knobs {
    pub fn validate(&self) -> Result<(), String> {
        let zero = [
            ("tokenize_queue", self.tokenize_queue.unwrap_or(1) as u64),
            ("overloaded_retry_after_secs", self.overloaded_retry_after_secs),
            ("redis_timeout_ms", self.redis_timeout_ms),
            ("idempotency_local_max_bytes", self.idempotency_local_max_bytes),
            ("idempotency_max_stored_bytes", self.idempotency_max_stored_bytes as u64),
            ("ort_global_threads", self.ort_global_threads.unwrap_or(1) as u64),
        ];
        match zero.iter().find(|(_, v)| *v == 0) {
            Some((k, _)) => Err(format!("knobs.{k} must be >= 1")),
            None if self.idempotency_max_stored_bytes as u64 > self.idempotency_local_max_bytes => {
                Err("knobs.idempotency_max_stored_bytes exceeds knobs.idempotency_local_max_bytes".into())
            }
            None => Ok(()),
        }
    }

    pub fn redis_timeout(&self) -> Duration {
        Duration::from_millis(self.redis_timeout_ms)
    }

    pub fn retry_after(&self) -> Duration {
        Duration::from_secs(self.overloaded_retry_after_secs)
    }
}
