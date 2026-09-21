//! Real ONNX model vs fixtures from tools/export_onnx.py.
//! Run: MODEL_DIR=models/english cargo test --release --features parity --test parity
#![cfg(feature = "parity")]

use rsdecider::lang;
use rsdecider::model::LoadedModel;
use rsdecider::model::postprocess::{Raw, answer};
use rsdecider::model::sequence::{Encoded, QuestionDef, encode, prepare, state_segment};
use rsdecider::model::session::OrtBackend;
use rsdecider::scheduler::{Backend, Batch};
use serde_json::Value;
use std::path::PathBuf;
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

#[test]
fn onnx_matches_pytorch_fixtures() {
    let dir = PathBuf::from(std::env::var("MODEL_DIR").expect("set MODEL_DIR to an exported model folder"));
    let model = LoadedModel::load("parity", &dir).unwrap();
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

    let mut backend = OrtBackend::new(&dir.join("model.onnx"), "cpu", 4, false).unwrap();
    let run = |b: &mut OrtBackend, its: &[&Item]| {
        b.run(&Batch { items: its.iter().map(|i| i.enc.clone()).collect(), pad_id: model.meta.pad_id }).unwrap()
    };

    // single-item runs, plus answers vs Agent.system_one
    for it in &items {
        let raw: Raw = run(&mut backend, &[it]).remove(0);
        let case = &fixtures[it.case];
        let d = max_diff(&raw.logits, &case["raw"][&it.qid]["logits"]);
        assert!(d < 1e-3, "case {} {}: single prob diff {d}", it.case, it.qid);
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
    for (it, raw) in all.iter().zip(run(&mut backend, &all)) {
        let d = max_diff(&raw.logits, &fixtures[it.case]["raw"][&it.qid]["logits"]);
        assert!(d < 1e-3, "case {} {}: batched prob diff {d}", it.case, it.qid);
    }
}
