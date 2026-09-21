use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: Server,
    #[serde(default)]
    pub limits: Limits,
    pub routing: Routing,
    pub models: Vec<ModelCfg>,
    #[serde(default)]
    pub cache: CacheCfg,
    #[serde(default)]
    pub keys: Vec<KeyCfg>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Server {
    pub listen: String,
    pub metrics_listen: String,
    pub worker_threads: usize,
    pub tokenize_threads: usize,
    pub request_timeout_ms: u64,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:3000".into(),
            metrics_listen: "127.0.0.1:9000".into(),
            worker_threads: 2,
            tokenize_threads: 2,
            request_timeout_ms: 10_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Limits {
    pub max_body_bytes: usize,
    pub max_state_chars: usize,
    pub max_batch_states: usize,
    pub max_questions_per_state: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_body_bytes: 1 << 20, max_state_chars: 100_000, max_batch_states: 8, max_questions_per_state: 16 }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Routing {
    pub english_model: String,
    pub non_english_model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCfg {
    pub name: String,
    pub path: PathBuf,
    #[serde(default = "d_ep")]
    pub execution_provider: String,
    #[serde(default = "d_one")]
    pub workers: usize,
    #[serde(default = "d_intra")]
    pub intra_op_threads: usize,
    #[serde(default = "d_pending")]
    pub max_pending: usize,
    #[serde(default = "d_items")]
    pub max_batch_items: usize,
    #[serde(default = "d_tokens")]
    pub max_batch_tokens: usize,
    #[serde(default = "d_wait")]
    pub max_wait_ms: u64,
}

fn d_ep() -> String {
    "cpu".into()
}
fn d_one() -> usize {
    1
}
fn d_intra() -> usize {
    6
}
fn d_pending() -> usize {
    256
}
fn d_items() -> usize {
    8
}
fn d_tokens() -> usize {
    8192
}
fn d_wait() -> u64 {
    2
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CacheCfg {
    pub l1_max_bytes: u64,
    pub l1_ttl_secs: u64,
    pub redis_url: Option<String>,
    pub l2_ttl_secs: u64,
    pub idempotency_ttl_secs: u64,
}

impl Default for CacheCfg {
    fn default() -> Self {
        Self {
            l1_max_bytes: 256 << 20,
            l1_ttl_secs: 3600,
            redis_url: None,
            l2_ttl_secs: 86_400,
            idempotency_ttl_secs: 86_400,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct KeyCfg {
    pub name: String,
    pub sha256: String,
    pub rps: u32,
    pub burst: u32,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let cfg: Config = toml::from_str(s).map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_toml_str(&s)
    }

    /// Structural checks. The `max_batch_tokens >= max_len` check needs laya.json and runs at model load.
    pub fn validate(&self) -> Result<(), String> {
        if self.models.is_empty() {
            return Err("at least one [[models]] entry is required".into());
        }
        if self.server.worker_threads == 0 {
            return Err("server.worker_threads must be >= 1".into());
        }
        let mut names = std::collections::HashSet::new();
        for m in &self.models {
            if !names.insert(m.name.as_str()) {
                return Err(format!("duplicate model name {:?}", m.name));
            }
            if m.workers == 0 || m.max_batch_items == 0 || m.intra_op_threads == 0 {
                return Err(format!("model {:?}: workers, max_batch_items and intra_op_threads must be >= 1", m.name));
            }
            let need = self.limits.max_batch_states * self.limits.max_questions_per_state;
            if need > m.max_pending {
                return Err(format!(
                    "model {:?}: max_batch_states x max_questions_per_state = {need} exceeds max_pending = {}",
                    m.name, m.max_pending
                ));
            }
        }
        for r in [&self.routing.english_model, &self.routing.non_english_model] {
            if !names.contains(r.as_str()) {
                return Err(format!("routing names unknown model {r:?}"));
            }
        }
        if self.limits.max_batch_states == 0 || self.limits.max_questions_per_state == 0 {
            return Err("limits must be >= 1".into());
        }
        let mut key_names = std::collections::HashSet::new();
        for k in &self.keys {
            if !key_names.insert(k.name.as_str()) {
                return Err(format!("duplicate key name {:?}", k.name));
            }
            if k.sha256.len() != 64 || !k.sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
                return Err(format!("key {:?}: sha256 must be 64 lowercase hex chars", k.name));
            }
            if k.rps == 0 {
                return Err(format!("key {:?}: rps must be >= 1", k.name));
            }
            if (k.burst as usize) < self.limits.max_batch_states {
                return Err(format!(
                    "key {:?}: burst {} < max_batch_states {}",
                    k.name, k.burst, self.limits.max_batch_states
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
[routing]
english_model = "english"
non_english_model = "multilingual"

[[models]]
name = "english"
path = "models/english"

[[models]]
name = "multilingual"
path = "models/multilingual"

[[keys]]
name = "acme"
sha256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
rps = 20
burst = 40
"#;

    #[test]
    fn defaults_apply() {
        let c = Config::from_toml_str(BASE).unwrap();
        assert_eq!(c.server.listen, "0.0.0.0:3000");
        assert_eq!(c.server.metrics_listen, "127.0.0.1:9000");
        assert_eq!(c.limits.max_batch_states, 8);
        assert_eq!(c.models[0].max_pending, 256);
        assert_eq!(c.models[0].execution_provider, "cpu");
        assert_eq!(c.cache.redis_url, None);
    }

    #[test]
    fn rejects_unknown_routing_model() {
        let s = BASE.replace("non_english_model = \"multilingual\"", "non_english_model = \"nope\"");
        assert!(Config::from_toml_str(&s).unwrap_err().contains("nope"));
    }

    #[test]
    fn rejects_unadmittable_request_size() {
        let s = BASE.replace("path = \"models/english\"", "path = \"models/english\"\nmax_pending = 100");
        assert!(Config::from_toml_str(&s).unwrap_err().contains("max_pending"));
    }

    #[test]
    fn rejects_small_burst() {
        let s = BASE.replace("burst = 40", "burst = 4");
        assert!(Config::from_toml_str(&s).unwrap_err().contains("burst"));
    }

    #[test]
    fn rejects_bad_hash() {
        let s = BASE.replace("9f86d081", "XYZ");
        assert!(Config::from_toml_str(&s).unwrap_err().contains("sha256"));
    }

    #[test]
    fn rejects_zero_worker_threads() {
        let s = format!("[server]\nworker_threads = 0\n{BASE}");
        assert!(Config::from_toml_str(&s).unwrap_err().contains("worker_threads"));
    }
}
