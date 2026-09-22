//! Model folder verification (`manifest.json`, written by tools/export_onnx.py) and download.
use crate::config::ModelCfg;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Files a pre-manifest export folder must have to load.
const LEGACY: [&str; 3] = ["model.onnx", "tokenizer.json", "laya.json"];
/// The only names a manifest may list (they become paths inside the model folder).
const KNOWN: [&str; 6] = ["model.onnx", "tokenizer.json", "laya.json", "fixtures.json", "mlx.safetensors", "mlx.json"];
/// Remote manifest.json read cap.
const MAX_MANIFEST: usize = 1 << 20;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub files: BTreeMap<String, Entry>,
}

#[derive(Debug, Deserialize)]
pub struct Entry {
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, PartialEq)]
pub enum Status {
    Ok,
    /// No manifest.json, but the files needed to load are there.
    Unverified,
    Bad {
        missing: Vec<String>,
        corrupt: Vec<String>,
    },
}

impl Status {
    pub fn usable(&self) -> bool {
        !matches!(self, Status::Bad { .. })
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Status::Ok => f.write_str("ok"),
            Status::Unverified => f.write_str("ok (no manifest, not verified)"),
            Status::Bad { missing, corrupt } => {
                let mut parts = Vec::new();
                if !missing.is_empty() {
                    parts.push(format!("missing {}", missing.join(", ")));
                }
                if !corrupt.is_empty() {
                    parts.push(format!("corrupt {}", corrupt.join(", ")));
                }
                f.write_str(&parts.join("; "))
            }
        }
    }
}

fn parse_manifest(raw: &[u8]) -> Result<Manifest, String> {
    let m: Manifest = serde_json::from_slice(raw).map_err(|e| format!("manifest.json: {e}"))?;
    if let Some(name) = m.files.keys().find(|n| !KNOWN.contains(&n.as_str())) {
        return Err(format!("manifest.json: unexpected file name {name:?}"));
    }
    let has = |n: &str| m.files.contains_key(n);
    if !(has("tokenizer.json") && has("laya.json") && (has("model.onnx") || has("mlx.safetensors"))) {
        return Err("manifest.json: must list tokenizer.json, laya.json and model.onnx or mlx.safetensors".into());
    }
    Ok(m)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let (mut h, mut buf) = (Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        match f.read(&mut buf)? {
            0 => return Ok(hex::encode(h.finalize())),
            n => h.update(&buf[..n]),
        }
    }
}

#[derive(PartialEq)]
enum Fault {
    Missing,
    Corrupt,
}

/// `None` = fine. Corrupt = wrong size, or wrong sha256 when `full` (hashing GBs takes seconds).
fn fault(path: &Path, e: &Entry, full: bool) -> Option<Fault> {
    let Ok(md) = std::fs::metadata(path) else { return Some(Fault::Missing) };
    let ok = md.len() == e.size && (!full || sha256_file(path).is_ok_and(|h| h.eq_ignore_ascii_case(&e.sha256)));
    (!ok).then_some(Fault::Corrupt)
}

/// Checks a model folder against its manifest.json. `full` hashes every file; otherwise sizes only.
pub fn check(dir: &Path, full: bool) -> Status {
    let raw = match std::fs::read(dir.join("manifest.json")) {
        Ok(raw) => raw,
        Err(_) => {
            let missing: Vec<String> =
                LEGACY.iter().filter(|f| !dir.join(f).is_file()).map(|f| f.to_string()).collect();
            if missing.is_empty() {
                return Status::Unverified;
            }
            return Status::Bad { missing: [vec!["manifest.json".into()], missing].concat(), corrupt: vec![] };
        }
    };
    let Ok(m) = parse_manifest(&raw) else {
        return Status::Bad { missing: vec![], corrupt: vec!["manifest.json".into()] };
    };
    let (mut missing, mut corrupt) = (vec![], vec![]);
    for (name, e) in &m.files {
        match fault(&dir.join(name), e, full) {
            Some(Fault::Missing) => missing.push(name.clone()),
            Some(Fault::Corrupt) => corrupt.push(name.clone()),
            None => {}
        }
    }
    if missing.is_empty() && corrupt.is_empty() { Status::Ok } else { Status::Bad { missing, corrupt } }
}

pub fn export_hint(m: &ModelCfg) -> String {
    format!(
        "export it with `python tools/export_onnx.py --out {}` (add `--subfolder multilingual` for the multilingual model)",
        m.path.display()
    )
}

/// What `serve` does about a model whose folder failed the check.
#[derive(Debug, PartialEq)]
pub enum Action {
    Pull,
    Prompt,
    Fail(String),
}

pub fn decide(m: &ModelCfg, status: &Status, download_flag: bool, interactive: bool) -> Action {
    let head = format!("model {:?} not usable at {} ({status})", m.name, m.path.display());
    match (&m.download, download_flag, interactive) {
        (None, ..) => Action::Fail(format!("{head}; {}, or set `download` for it in the config", export_hint(m))),
        (Some(_), true, _) => Action::Pull,
        (Some(_), false, true) => Action::Prompt,
        (Some(url), false, false) => Action::Fail(format!(
            "{head}; download it from {url} with --download-models or RSDECIDER_DOWNLOAD_MODELS=1 \
             (or `rsdecider models pull`), or {}",
            export_hint(m)
        )),
    }
}

/// A fetched remote manifest, ready to pull from.
pub struct Remote {
    base: String,
    client: reqwest::Client,
    raw: Vec<u8>,
    pub manifest: Manifest,
}

impl Remote {
    pub async fn fetch(base: &str) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| e.to_string())?;
        let base = base.trim_end_matches('/').to_string();
        let url = format!("{base}/manifest.json");
        let mut resp = get(&client, &url).await?;
        let mut raw = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| format!("{url}: {e}"))? {
            if raw.len() + chunk.len() > MAX_MANIFEST {
                return Err(format!("{url}: larger than {MAX_MANIFEST} bytes"));
            }
            raw.extend_from_slice(&chunk);
        }
        let manifest = parse_manifest(&raw).map_err(|e| format!("{url}: {e}"))?;
        Ok(Self { base, client, raw, manifest })
    }

    pub fn total_bytes(&self) -> u64 {
        self.manifest.files.values().fold(0u64, |a, e| a.saturating_add(e.size))
    }

    /// Downloads every file that is missing or fails size+sha256, then writes manifest.json last.
    pub async fn pull(&self, label: &str, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (name, e) in &self.manifest.files {
            let dest = dir.join(name);
            if fault(&dest, e, true).is_none() {
                continue;
            }
            self.download(label, name, e, &dest).await?;
        }
        let dest = dir.join("manifest.json");
        let part = part_path(&dest);
        std::fs::write(&part, &self.raw)
            .and_then(|_| std::fs::rename(&part, &dest))
            .map_err(|e| format!("{}: {e}", dest.display()))
    }

    // Plain std::fs writes: downloads run on their own runtime before serving starts, never on server workers.
    async fn download(&self, label: &str, name: &str, e: &Entry, dest: &Path) -> Result<(), String> {
        let url = format!("{}/{name}", self.base);
        let part = part_path(dest);
        // ponytail: a stale .part is truncated and restarts from zero; add Range resume if flaky links re-download GBs.
        let res = async {
            let mut resp = get(&self.client, &url).await?;
            let mut f = std::fs::File::create(&part).map_err(|x| format!("{}: {x}", part.display()))?;
            let (mut h, mut done) = (Sha256::new(), 0u64);
            let (mut last_pct, mut last_at) = (0, Instant::now());
            while let Some(chunk) = resp.chunk().await.map_err(|x| format!("{url}: {x}"))? {
                if done + chunk.len() as u64 > e.size {
                    return Err(format!("{name}: server sent more than the manifest's {} bytes", e.size));
                }
                h.update(&chunk);
                f.write_all(&chunk).map_err(|x| format!("{}: {x}", part.display()))?;
                done += chunk.len() as u64;
                let pct = done * 100 / e.size.max(1);
                if pct >= last_pct + 5 || last_at.elapsed() >= Duration::from_secs(5) {
                    eprintln!("{label}: {name} {} / {} MB ({pct}%)", done >> 20, e.size >> 20);
                    (last_pct, last_at) = (pct, Instant::now());
                }
            }
            f.sync_all().map_err(|x| format!("{}: {x}", part.display()))?;
            let got = hex::encode(h.finalize());
            if done != e.size || !got.eq_ignore_ascii_case(&e.sha256) {
                return Err(format!(
                    "{name}: downloaded {done} bytes with sha256 {got}, manifest says {} bytes with sha256 {}",
                    e.size, e.sha256
                ));
            }
            std::fs::rename(&part, dest).map_err(|x| format!("{}: {x}", dest.display()))
        }
        .await;
        if res.is_err() {
            let _ = std::fs::remove_file(&part);
        }
        res
    }
}

fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    s.into()
}

async fn get(client: &reqwest::Client, url: &str) -> Result<reqwest::Response, String> {
    client.get(url).send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("{url}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Temp dir removed on drop.
    struct Tmp(PathBuf);

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    impl std::ops::Deref for Tmp {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    fn tmp() -> Tmp {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "rsdecider-dl-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn manifest_for(files: &[(&str, &[u8])]) -> String {
        let entries: Vec<String> = files
            .iter()
            .map(|(n, b)| format!(r#""{n}": {{"size": {}, "sha256": "{}"}}"#, b.len(), hex::encode(Sha256::digest(b))))
            .collect();
        format!(r#"{{"files": {{{}}}}}"#, entries.join(", "))
    }

    const FILES: [(&str, &[u8]); 3] =
        [("model.onnx", b"weights weights weights"), ("tokenizer.json", b"{\"tok\":1}"), ("laya.json", b"{}")];

    /// Serves `files` plus a manifest (built from `manifest_files`) under /english on 127.0.0.1.
    async fn serve(files: &[(&'static str, &'static [u8])], manifest_files: &[(&str, &[u8])]) -> String {
        let mut app = Router::new();
        let m = manifest_for(manifest_files);
        app = app.route("/english/manifest.json", axum::routing::get(move || async move { m }));
        for &(n, b) in files {
            app = app.route(&format!("/english/{n}"), axum::routing::get(move || async move { b }));
        }
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await });
        format!("http://{addr}/english")
    }

    fn write(dir: &Path, files: &[(&str, &[u8])]) {
        for (n, b) in files {
            std::fs::write(dir.join(n), b).unwrap();
        }
    }

    #[tokio::test]
    async fn pull_downloads_and_verifies() {
        let url = serve(&FILES, &FILES).await;
        let root = tmp();
        let dir = root.join("english");
        Remote::fetch(&url).await.unwrap().pull("english", &dir).await.unwrap();
        assert_eq!(check(&dir, true), Status::Ok);
        assert_eq!(std::fs::read(dir.join("model.onnx")).unwrap(), FILES[0].1);
    }

    #[tokio::test]
    async fn pull_rejects_corrupt_file_and_leaves_nothing() {
        let served = [("model.onnx", &b"tampered weights weights"[..]), FILES[1], FILES[2]];
        let url = serve(&served, &FILES).await;
        let dir = tmp();
        let err = Remote::fetch(&url).await.unwrap().pull("english", &dir).await.unwrap_err();
        assert!(err.contains("model.onnx"), "{err}");
        assert!(!dir.join("model.onnx").exists() && !dir.join("model.onnx.part").exists());
        assert!(!dir.join("manifest.json").exists());
    }

    #[tokio::test]
    async fn pull_stops_when_server_sends_more_than_manifest() {
        let served = [("model.onnx", &b"weights weights weights and then some more"[..]), FILES[1], FILES[2]];
        let url = serve(&served, &FILES).await;
        let dir = tmp();
        let err = Remote::fetch(&url).await.unwrap().pull("english", &dir).await.unwrap_err();
        assert!(err.contains("model.onnx") && err.contains("more than"), "{err}");
        assert!(!dir.join("model.onnx").exists() && !dir.join("model.onnx.part").exists());
    }

    #[tokio::test]
    async fn pull_keeps_good_files_and_replaces_corrupt_ones() {
        // laya.json is not served at all: pull only succeeds if it keeps the good local copy.
        let url = serve(&FILES[..2], &FILES).await;
        let dir = tmp();
        write(&dir, &[FILES[2], ("tokenizer.json", b"{\"tok\":2}")]);
        Remote::fetch(&url).await.unwrap().pull("english", &dir).await.unwrap();
        assert_eq!(std::fs::read(dir.join("tokenizer.json")).unwrap(), FILES[1].1);
        assert_eq!(check(&dir, true), Status::Ok);
    }

    #[test]
    fn check_reports_missing_and_corrupt() {
        let dir = tmp();
        write(&dir, &FILES);
        std::fs::write(dir.join("manifest.json"), manifest_for(&FILES)).unwrap();
        assert_eq!(check(&dir, true), Status::Ok);
        std::fs::remove_file(dir.join("laya.json")).unwrap();
        std::fs::write(dir.join("model.onnx"), b"weights weights weightX").unwrap(); // same size, wrong hash
        let s = check(&dir, true);
        assert_eq!(s, Status::Bad { missing: vec!["laya.json".into()], corrupt: vec!["model.onnx".into()] });
        assert_eq!(s.to_string(), "missing laya.json; corrupt model.onnx");
        assert!(!check(&dir, false).usable(), "cheap check still sees the missing file");
    }

    #[test]
    fn cheap_check_accepts_legacy_and_rejects_size_mismatch() {
        let dir = tmp();
        write(&dir, &FILES);
        assert_eq!(check(&dir, false), Status::Unverified);
        std::fs::write(dir.join("manifest.json"), manifest_for(&FILES)).unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"short").unwrap();
        assert_eq!(check(&dir, false), Status::Bad { missing: vec![], corrupt: vec!["tokenizer.json".into()] });
    }

    #[test]
    fn manifest_allows_only_known_names() {
        let with = |extra: &str| {
            let entry = format!("{{\"files\": {{\"{extra}\": {{\"size\": 1, \"sha256\": \"00\"}}, ");
            parse_manifest(manifest_for(&FILES).replacen("{\"files\": {", &entry, 1).as_bytes())
        };
        assert!(with("mlx.json").is_ok());
        for bad in ["../x", "/x", "a/b", "", "evil.sh", "manifest.json"] {
            assert!(with(bad).is_err(), "{bad:?} accepted");
        }
        assert!(parse_manifest(manifest_for(&FILES[..2]).as_bytes()).is_err(), "laya.json is required");
        assert!(parse_manifest(manifest_for(&FILES[1..]).as_bytes()).is_err(), "a model file is required");
    }

    fn model(download: Option<&str>) -> ModelCfg {
        let s = format!(
            "name = \"english\"\npath = \"models/english\"\n{}",
            download.map(|u| format!("download = \"{u}\"")).unwrap_or_default()
        );
        toml::from_str(&s).unwrap()
    }

    #[test]
    fn decide_non_interactive_without_flag_names_the_flag() {
        let bad = Status::Bad { missing: vec!["model.onnx".into()], corrupt: vec![] };
        let Action::Fail(e) = decide(&model(Some("http://h/english")), &bad, false, false) else { panic!() };
        assert!(e.contains("--download-models") && e.contains("RSDECIDER_DOWNLOAD_MODELS=1"), "{e}");
        assert!(e.contains("tools/export_onnx.py --out models/english"), "{e}");
        assert_eq!(decide(&model(Some("http://h/english")), &bad, true, false), Action::Pull);
        assert_eq!(decide(&model(Some("http://h/english")), &bad, false, true), Action::Prompt);
        let Action::Fail(e) = decide(&model(None), &bad, true, true) else { panic!() };
        assert!(e.contains("export_onnx.py") && e.contains("--subfolder multilingual"), "{e}");
    }
}
