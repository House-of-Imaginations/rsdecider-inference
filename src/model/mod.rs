pub mod postprocess;
pub mod pyjson;
pub mod sequence;
pub mod session;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// `laya.json`, written by `tools/export_onnx.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct LayaMeta {
    pub max_len: usize,
    pub head_max_len: usize,
    pub mask_token: String,
    pub mask_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub pad_id: u32,
    pub temperature: [f64; 3],
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f64>,
}

/// Everything about a model except the ORT session (sessions live on worker threads).
pub struct LoadedModel {
    pub name: String,
    pub dir: PathBuf,
    pub meta: LayaMeta,
    pub tokenizer: Tokenizer,
    pub fingerprint: [u8; 32],
}

impl LoadedModel {
    pub fn load(name: &str, dir: &Path) -> Result<Self, String> {
        let meta_path = dir.join("laya.json");
        let meta: LayaMeta = serde_json::from_str(
            &std::fs::read_to_string(&meta_path).map_err(|e| format!("{}: {e}", meta_path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", meta_path.display()))?;
        let tok_path = dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
        // Laya calls the HF tokenizer per segment with no truncation/padding.
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        let mut h = Sha256::new();
        for f in ["model.onnx", "tokenizer.json", "laya.json"] {
            let p = dir.join(f);
            let mut file = std::fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = file.read(&mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
            }
        }
        Ok(Self { name: name.into(), dir: dir.into(), meta, tokenizer, fingerprint: h.finalize().into() })
    }

    /// `laya-<name>@<first 12 hex chars of the fingerprint>`
    pub fn display_name(&self) -> String {
        format!("laya-{}@{}", self.name, &hex::encode(self.fingerprint)[..12])
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
    }
}
