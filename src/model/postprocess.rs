//! Port of the answer-building half of Laya `agent.py::system_one`.
use super::LayaMeta;
use super::sequence::{Prepared, QType};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// Raw model output for one question. This is what gets cached.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Raw {
    /// Marker logits for the k valid options (padding already stripped).
    pub logits: Vec<f32>,
    pub act_prob: f32,
    pub n_tokens: u32,
}

pub fn temp_bucket(q: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", q.name())
}

// ponytail: half-away-from-zero; Python round() is half-even on the exact binary value.
// Differs only on exact ties at the 5th decimal, far below the 1e-3 parity tolerance.
fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn confidence(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = -p.iter().map(|&x| x * x.clamp(1e-12, 1.0).ln()).sum::<f64>();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Answer object for one question (without the rsdecider-only `cached` flag).
pub fn answer(p: &Prepared, raw: &Raw, meta: &LayaMeta) -> Value {
    let k = raw.logits.len();
    let t =
        meta.temperature_by_options.get(&temp_bucket(p.qtype, k)).copied().unwrap_or(meta.temperature[p.qtype.id()]);
    let z: Vec<f64> = raw.logits.iter().map(|&l| l as f64 / t.max(1e-3)).collect();
    let probs = softmax(&z);
    let conf = round4(confidence(&probs));
    let action = json!({ "act_probability": round4(raw.act_prob as f64) });
    match p.qtype {
        QType::Choice => {
            let mut best = 0;
            for (i, &x) in probs.iter().enumerate() {
                if x > probs[best] {
                    best = i;
                }
            }
            let dist: Map<String, Value> =
                p.labels.iter().zip(&probs).map(|(l, &x)| (l.clone(), json!(round4(x)))).collect();
            json!({"type": "choice", "choice": p.labels[best], "probabilities": dist, "confidence": conf, "action": action})
        }
        QType::Score => {
            let expected: f64 = probs.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
            let legend: Map<String, Value> =
                p.legend.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect();
            let dist: Map<String, Value> =
                probs.iter().enumerate().map(|(i, &x)| (i.to_string(), json!(round4(x)))).collect();
            json!({"type": "score", "score": round4(expected), "legend": legend, "probabilities": dist, "confidence": conf, "action": action})
        }
        QType::Noul => {
            let p1 = probs[1];
            json!({"type": "noul", "noul": round4(p1), "confidence": round4(p1.max(1.0 - p1)), "action": action})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::sequence::{QuestionDef, prepare};

    fn meta() -> LayaMeta {
        serde_json::from_value(json!({
            "max_len": 512, "head_max_len": 192, "mask_token": "[MASK]", "mask_id": 4, "cls_id": 1,
            "sep_id": 2, "pad_id": 3, "temperature": [2.0, 1.0, 1.0], "temperature_by_options": {"choice:3-5": 0.5}
        }))
        .unwrap()
    }

    fn prep(v: Value) -> Prepared {
        prepare(&serde_json::from_value::<QuestionDef>(v).unwrap(), "[MASK]").unwrap()
    }

    #[test]
    fn buckets() {
        assert_eq!(temp_bucket(QType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QType::Score, 5), "score:3-5");
        assert_eq!(temp_bucket(QType::Choice, 10), "choice:6-10");
        assert_eq!(temp_bucket(QType::Choice, 11), "choice:11+");
    }

    #[test]
    fn choice_uses_bucket_temperature_and_first_argmax() {
        let p = prep(json!({"type": "choice", "instructions": "x", "criteria": ["a", "b", "c"]}));
        // bucket choice:3-5 → t = 0.5 → z = [2, 2, 0]
        let a = answer(&p, &Raw { logits: vec![1.0, 1.0, 0.0], act_prob: 0.25, n_tokens: 9 }, &meta());
        assert_eq!(a["choice"], "a");
        assert_eq!(a["probabilities"]["a"], 0.4683);
        assert_eq!(a["probabilities"]["c"], 0.0634);
        assert_eq!(a["action"]["act_probability"], 0.25);
        let keys: Vec<&String> = a["probabilities"].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["a", "b", "c"]);
    }

    #[test]
    fn choice_falls_back_to_type_temperature() {
        let p = prep(json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}));
        // no "choice:2" bucket → temperature[0] = 2.0 → z = [1, 0]
        let a = answer(&p, &Raw { logits: vec![2.0, 0.0], act_prob: 0.5, n_tokens: 9 }, &meta());
        assert_eq!(a["probabilities"]["a"], 0.7311);
        assert_eq!(a["confidence"], 0.1601);
    }

    #[test]
    fn score_expected_value_and_legend() {
        let p = prep(json!({"type": "score", "instructions": "x", "criteria": ["low", {"k": 1}, "high"]}));
        let a = answer(&p, &Raw { logits: vec![0.0, 0.0, 0.0], act_prob: 0.5, n_tokens: 9 }, &meta());
        assert_eq!(a["score"], 1.0);
        assert_eq!(a["legend"]["1"], json!({"k": 1}));
        assert_eq!(a["confidence"], 0.0);
    }

    #[test]
    fn noul_confidence_is_max_side() {
        let p = prep(json!({"type": "noul", "instructions": "x"}));
        let a = answer(&p, &Raw { logits: vec![0.0, (3.0f32).ln()], act_prob: 0.5, n_tokens: 9 }, &meta());
        assert_eq!(a["noul"], 0.75);
        assert_eq!(a["confidence"], 0.75);
    }
}
