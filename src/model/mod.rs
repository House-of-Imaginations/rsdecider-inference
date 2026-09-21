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

    /// See [`tokenize_prefix`].
    pub fn tokenize_prefix(&self, text: &str, need: usize) -> Result<Vec<u32>, String> {
        tokenize_prefix(&self.tokenizer, text, need)
    }
}

/// Leading tokens of `text`, enough to fill `need` positions: tokenizes a prefix cut at whitespace and keeps
/// only tokens well clear of the cut (so they equal full tokenization); falls back to the whole text.
pub fn tokenize_prefix(tok: &Tokenizer, text: &str, need: usize) -> Result<Vec<u32>, String> {
    // ponytail: 8 bytes/token covers every script seen; the fallback keeps it exact when it does not
    const MARGIN: usize = 64;
    let full = || Ok(tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec());
    let mut cut = need.saturating_mul(8);
    if cut >= text.len() {
        return full();
    }
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    if let Some(ws) = text[..cut].rfind(char::is_whitespace) {
        cut = ws;
    }
    // `encode` reports byte offsets into its input.
    let enc = tok.encode(&text[..cut], false).map_err(|e| e.to_string())?;
    let keep = enc.get_offsets().iter().take_while(|&&(_, end)| end + MARGIN <= cut).count();
    if keep >= need { Ok(enc.get_ids()[..keep].to_vec()) } else { full() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tokenizers() -> Vec<(&'static str, Tokenizer)> {
        let found: Vec<_> = ["models/english/tokenizer.json", "models/multilingual/tokenizer.json"]
            .into_iter()
            .filter(|p| Path::new(p).exists())
            .map(|p| (p, Tokenizer::from_file(p).unwrap()))
            .collect();
        if found.is_empty() {
            println!("skip: no models/*/tokenizer.json");
        }
        found
    }

    fn full(tok: &Tokenizer, t: &str) -> Vec<u32> {
        tok.encode(t, false).unwrap().get_ids().to_vec()
    }

    fn texts() -> Vec<(&'static str, String)> {
        let english = [
            "Customer reports the invoice total is wrong after the refund was applied. ",
            "They were charged twice on 2026-09-01 and want the duplicate reversed!\n",
            "Agent: I've escalated this to billing (ticket #48213); expect a reply within 24h. ",
            "Re: re: FW: can't log in — password reset link expired, tried Safari & Chrome.\n\n",
        ];
        let vi = "Khách hàng phản ánh rằng đơn hàng bị giao chậm và sản phẩm bị hư hỏng khi nhận được. ";
        let zh = "客户反映订单延迟发货并且收到的商品已经损坏希望尽快退款或者换货谢谢";
        let emoji = "ok 👍🏽 thanks!! 🎉🎉 great 👨‍👩‍👧‍👦 support 🚀✨ ";
        let obj = json!({"subject": "Refund request", "body": "Charged twice, please reverse. Thanks!", "tags": ["billing", "urgent"], "n": 3});
        let one = pyjson::dumps(&obj, false);
        let mut out = vec![
            ("english", english.iter().cycle().take(1500).copied().collect::<String>()),
            ("vietnamese", vi.repeat(30_000 / vi.len() + 1)),
            ("chinese", zh.repeat(20_000 / zh.len() + 1)),
            ("emoji", emoji.repeat(300)),
            ("json", format!("[{}]", vec![one; 200].join(", "))),
            ("whitespace", "word    \n\n\n        \t  more   ".repeat(800)),
        ];
        out.retain(|(_, t)| t.len() > 8 * 512);
        out
    }

    #[test]
    fn prefix_equals_full_tokenization_prefix() {
        for (path, tok) in tokenizers() {
            for (name, t) in texts() {
                let full = full(&tok, &t);
                let got = tokenize_prefix(&tok, &t, 512).unwrap();
                assert!(got.len() >= 512.min(full.len()), "{path} {name}: {} tokens", got.len());
                assert_eq!(got, full[..got.len()], "{path} {name}");
                assert!(got.len() < full.len(), "{path} {name}: prefix path not taken");
            }
        }
    }

    #[test]
    fn short_text_and_large_need_fall_back_to_full() {
        for (_, tok) in tokenizers() {
            let short = "a short state that fits well under the byte budget";
            assert_eq!(tokenize_prefix(&tok, short, 512).unwrap(), full(&tok, short));
            // Longer than need × 8 bytes but fewer than `need` tokens: the prefix cannot keep enough, so it falls back.
            let sparse = "-".repeat(10_000);
            assert!(full(&tok, &sparse).len() < 1_000);
            assert_eq!(tokenize_prefix(&tok, &sparse, 1_000).unwrap(), full(&tok, &sparse));
        }
    }
}
