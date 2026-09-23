//! Real ONNX (and, with `--features mlx`, MLX) model vs fixtures from tools/export_onnx.py.
//! Run: MODEL_DIR=models/english cargo test --release --features parity --test parity
//! MLX: MODEL_DIR=models/english cargo test --release --features parity,mlx --test parity
#![cfg(feature = "parity")]

use rsdecider::lang;
use rsdecider::model::LoadedModel;
use rsdecider::model::postprocess::{Raw, answer};
use rsdecider::model::sequence::{Encoded, QuestionDef, encode, prepare, state_segment};
use rsdecider::model::session::OrtBackend;
use rsdecider::scheduler::{Backend, Batch};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Item {
    case: usize,
    qid: String,
    enc: Arc<Encoded>,
    def: QuestionDef,
}

fn softmax(l: &[f32]) -> Vec<f64> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = l.iter().map(|&x| (x as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn max_diff(a: &[f32], b: &Value) -> f64 {
    let b: Vec<f32> = b.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    softmax(a).iter().zip(softmax(&b)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("MODEL_DIR").expect("set MODEL_DIR to an exported model folder"))
}

/// Encodes every fixture question and checks ids/markers against Laya's.
fn load_items(dir: &Path) -> (LoadedModel, Vec<Value>, Vec<Item>) {
    let model = LoadedModel::load("parity", dir, None).unwrap();
    let fixtures: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("fixtures.json")).unwrap()).unwrap();
    let mask = model.meta.mask_token.clone();
    let mut items = Vec::new();

    for (ci, case) in fixtures.iter().enumerate() {
        assert_eq!(lang::is_english(&case["state"]), case["is_english"].as_bool().unwrap(), "case {ci}: routing");
        let state_ids = model.tokenize(&state_segment(&case["state"], &mask)).unwrap();
        let errors: Vec<&str> = case["errors"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
        for (qid, qv) in case["questions"].as_object().unwrap() {
            let def: QuestionDef = serde_json::from_value(qv.clone()).unwrap();
            let res = prepare(&def, &mask).and_then(|p| encode(&model, &state_ids, &p));
            if errors.contains(&qid.as_str()) {
                assert!(res.is_err(), "case {ci} {qid}: Laya raised, we did not");
                continue;
            }
            let enc = res.unwrap_or_else(|e| panic!("case {ci} {qid}: {e}"));
            let want = &case["encoded"][qid];
            let ids: Vec<u32> = serde_json::from_value(want["ids"].clone()).unwrap();
            let markers: Vec<u32> = serde_json::from_value(want["markers"].clone()).unwrap();
            assert_eq!(enc.ids, ids, "case {ci} {qid}: token ids");
            assert_eq!(enc.markers, markers, "case {ci} {qid}: markers");
            items.push(Item { case: ci, qid: qid.clone(), enc: Arc::new(enc), def });
        }
    }
    (model, fixtures, items)
}

fn run(b: &mut impl Backend, model: &LoadedModel, its: &[&Item]) -> Vec<Raw> {
    b.run(&Batch { items: its.iter().map(|i| i.enc.clone()).collect(), pad_id: model.meta.pad_id }).unwrap()
}

/// Strict parity: |Δ softmax prob| < 1e-3 on single-item runs and one padded mixed-length batch.
fn check_parity(backend: &mut impl Backend, dir: &Path) {
    let (model, fixtures, items) = load_items(dir);
    let mut worst = 0f64;

    // single-item runs, plus answers vs Agent.system_one
    for it in &items {
        let raw: Raw = run(backend, &model, &[it]).remove(0);
        let case = &fixtures[it.case];
        let d = max_diff(&raw.logits, &case["raw"][&it.qid]["logits"]);
        assert!(d < 1e-3, "case {} {}: single prob diff {d}", it.case, it.qid);
        worst = worst.max(d);
        if let Some(want) = case.get("answers").map(|a| &a[&it.qid]) {
            let got = answer(&prepare(&it.def, &model.meta.mask_token).unwrap(), &raw, &model.meta);
            for (k, v) in want.get("probabilities").and_then(Value::as_object).into_iter().flatten() {
                let g = got["probabilities"][k].as_f64().unwrap();
                assert!((g - v.as_f64().unwrap()).abs() < 1e-3, "case {} {} prob {k}", it.case, it.qid);
            }
            assert_eq!(got.get("choice"), want.get("choice"), "case {} {}: choice", it.case, it.qid);
        }
    }

    // one padded mixed-length batch: padding must not change outputs
    let all: Vec<&Item> = items.iter().take(16).collect();
    for (it, raw) in all.iter().zip(run(backend, &model, &all)) {
        let d = max_diff(&raw.logits, &fixtures[it.case]["raw"][&it.qid]["logits"]);
        assert!(d < 1e-3, "case {} {}: batched prob diff {d}", it.case, it.qid);
        worst = worst.max(d);
    }
    eprintln!("max |Δ softmax prob| {worst:.2e}");
}

#[test]
fn onnx_matches_pytorch_fixtures() {
    let dir = model_dir();
    check_parity(&mut OrtBackend::new(&dir.join("model.onnx"), "cpu", 4, false).unwrap(), &dir);
}

#[cfg(feature = "mlx")]
mod mlx {
    use super::*;
    use rsdecider::config::MlxDtype;
    use rsdecider::model::mlx::MlxBackend;
    use rsdecider::model::postprocess::temp_bucket;

    fn mlx_dir() -> Option<PathBuf> {
        let dir = model_dir();
        if dir.join("mlx.safetensors").is_file() {
            return Some(dir);
        }
        eprintln!("skipping: {} has no mlx.safetensors (export with --mlx)", dir.display());
        None
    }

    fn softmax_t(l: &[f64], t: f64) -> Vec<f64> {
        softmax(&l.iter().map(|&x| (x / t) as f32).collect::<Vec<_>>())
    }

    #[test]
    fn mlx_fp32_matches_pytorch_fixtures() {
        let Some(dir) = mlx_dir() else { return };
        check_parity(&mut MlxBackend::new(&dir, MlxDtype::Fp32).unwrap(), &dir);
    }

    /// Same rule as the export gate (tools/export_onnx.py `update_gate`): top-1 must match on decisive questions
    /// (single option, or reference temperature-scaled top-2 margin > 0.10); |Δp| <= 0.05 and |Δ act_prob| <= 0.05
    /// on every question.
    #[test]
    fn mlx_fp16_passes_gate() {
        let Some(dir) = mlx_dir() else { return };
        let (model, fixtures, items) = load_items(&dir);
        let mut backend = MlxBackend::new(&dir, MlxDtype::Fp16).unwrap();
        let all: Vec<&Item> = items.iter().take(16).collect();
        let runs: Vec<_> = items.iter().map(|it| (it, run(&mut backend, &model, &[it]).remove(0))).collect();
        let batched: Vec<_> = all.iter().copied().zip(run(&mut backend, &model, &all)).collect();
        let (mut decisive, mut agree, mut max_dp, mut max_dact) = (0, 0, 0f64, 0f64);
        for (it, raw) in runs.into_iter().chain(batched) {
            let want = &fixtures[it.case]["raw"][&it.qid];
            let reference: Vec<f64> = serde_json::from_value(want["logits"].clone()).unwrap();
            let k = reference.len();
            let qtype = it.enc.qtype;
            let t = model.meta.temperature_by_options.get(&temp_bucket(qtype, k)).copied();
            let t = t.unwrap_or(model.meta.temperature[qtype.id()]).max(1e-3);
            let got: Vec<f64> = raw.logits.iter().map(|&x| x as f64).collect();
            let (p, pr) = (softmax_t(&got, t), softmax_t(&reference, t));
            let argmax = |v: &[f64]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
            let mut sorted = pr.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            if k == 1 || sorted[0] - sorted[1] > 0.10 {
                decisive += 1;
                agree += usize::from(argmax(&p) == argmax(&pr));
            }
            max_dp = p.iter().zip(&pr).map(|(a, b)| (a - b).abs()).fold(max_dp, f64::max);
            max_dact = max_dact.max((raw.act_prob as f64 - want["act_prob"].as_f64().unwrap()).abs());
        }
        eprintln!(
            "mlx fp16 gate: top-1 on decisive {agree}/{decisive}; max |Δp| {max_dp:.4}; max |Δ act_prob| {max_dact:.4}"
        );
        assert_eq!(agree, decisive, "top-1 flipped on a decisive question");
        assert!(max_dp <= 0.05, "max |Δp| {max_dp}");
        assert!(max_dact <= 0.05, "max |Δ act_prob| {max_dact}");
    }
}
