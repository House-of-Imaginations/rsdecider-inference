//! Port of Laya `common.py::build_sequence` (+ `render_options`, `Agent._to_internal`).
use super::{LayaMeta, LoadedModel, pyjson};
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn id(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        ["choice", "score", "noul"][self as usize]
    }
}

/// One question as sent by the client (Jev/Laya shape).
#[derive(Debug, Clone, Deserialize)]
pub struct QuestionDef {
    #[serde(rename = "type")]
    pub qtype: String,
    pub instructions: Value,
    #[serde(default)]
    pub criteria: Option<Value>,
}

/// Exact text segments the model sees (after mask replacement), plus what postprocess needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub qtype: QType,
    /// "<type> question: <instructions>"
    pub head: String,
    /// " <option text>" per option, in label order.
    pub options: Vec<String>,
    /// Choice labels (empty for score/noul).
    pub labels: Vec<String>,
    /// Score legend values (empty for choice/noul).
    pub legend: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<u32>,
    pub qtype: QType,
}

fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, false),
    }
}

fn is_blank(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null)) || v == Some(&Value::String(String::new()))
}

pub fn prepare(q: &QuestionDef, mask: &str) -> Result<Prepared, String> {
    let ins = match &q.instructions {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, true),
    };
    let (qtype, labels, legend, opts): (QType, Vec<String>, Vec<Value>, Vec<String>) = match q.qtype.as_str() {
        "choice" => {
            let pairs: Vec<(String, Value)> = match &q.criteria {
                Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                Some(Value::Array(a)) => {
                    let mut seen: Vec<(String, Value)> = Vec::new();
                    for x in a {
                        let Value::String(s) = x else {
                            return Err("choice criteria list must contain strings".into());
                        };
                        if !seen.iter().any(|(k, _)| k == s) {
                            seen.push((s.clone(), Value::Null));
                        }
                    }
                    seen
                }
                _ => return Err("choice criteria must be an object or a list of labels".into()),
            };
            if !(2..=255).contains(&pairs.len()) {
                return Err(format!("choice needs 2-255 options, got {}", pairs.len()));
            }
            let opts = pairs
                .iter()
                .map(|(k, v)| if is_blank(Some(v)) { k.clone() } else { format!("{k}: {}", render_criterion(v)) })
                .collect();
            (QType::Choice, pairs.into_iter().map(|(k, _)| k).collect(), vec![], opts)
        }
        "score" => {
            let Some(Value::Array(levels)) = &q.criteria else { return Err("score criteria must be a list".into()) };
            if !(2..=10).contains(&levels.len()) {
                return Err(format!("score needs 2-10 levels, got {}", levels.len()));
            }
            let opts = levels.iter().enumerate().map(|(i, c)| format!("level {i}: {}", render_criterion(c))).collect();
            (QType::Score, vec![], levels.clone(), opts)
        }
        "noul" => {
            let crit = match &q.criteria {
                None => serde_json::Map::new(),
                Some(Value::Object(m)) => m.clone(),
                Some(_) => return Err("noul criteria must be an object with optional \"false\"/\"true\"".into()),
            };
            let side = |k: &str, default: &str| match crit.get(k) {
                c if is_blank(c) => format!("{k}: {default}"),
                Some(c) => format!("{k}: {}", render_criterion(c)),
                None => unreachable!(),
            };
            let opts = vec![side("false", "no, the statement does not hold"), side("true", "yes, the statement holds")];
            (QType::Noul, vec![], vec![], opts)
        }
        other => return Err(format!("unknown question type {other:?}")),
    };
    Ok(Prepared {
        qtype,
        head: format!("{} question: {}", qtype.name(), ins.replace(mask, " ")),
        options: opts.iter().map(|o| format!(" {}", o.replace(mask, " "))).collect(),
        labels,
        legend,
    })
}

/// `serialize_state(state).replace(mask_tok, " ")`
pub fn state_segment(state: &Value, mask: &str) -> String {
    match state {
        Value::String(s) => s.replace(mask, " "),
        v => pyjson::dumps(v, false).replace(mask, " "),
    }
}

/// Token-level assembly, an exact port of the integer logic in `build_sequence`.
/// Returns `(ids, markers)`; markers that fell past `max_len` are dropped (caller checks the count).
pub fn assemble(meta: &LayaMeta, head: &[u32], opts: &[Vec<u32>], state: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let hml = meta.head_max_len as i64;
    let mut opt_ids: Vec<Vec<u32>> =
        opts.iter().map(|o| std::iter::once(meta.mask_id).chain(o.iter().take(48).copied()).collect()).collect();
    let total = |v: &Vec<Vec<u32>>| v.iter().map(|o| o.len() as i64).sum::<i64>();
    let mut budget = hml - total(&opt_ids);
    if budget < 16 {
        let per = 4.max((hml - 16).div_euclid(opt_ids.len().max(1) as i64)) as usize;
        opt_ids.iter_mut().for_each(|o| o.truncate(per));
        budget = hml - total(&opt_ids);
    }
    let head_len = (8.max(budget) as usize).min(head.len());
    let mut ids = Vec::with_capacity(meta.max_len);
    ids.push(meta.cls_id);
    ids.extend_from_slice(&head[..head_len]);
    ids.push(meta.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len() as u32);
        ids.extend_from_slice(o);
    }
    ids.push(meta.sep_id);
    let room = (meta.max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
    ids.extend_from_slice(&state[..room.min(state.len())]);
    ids.push(meta.sep_id);
    ids.truncate(meta.max_len);
    markers.retain(|&m| (m as usize) < meta.max_len);
    (ids, markers)
}

/// Tokenize one question's head/options and assemble with the (already tokenized) state.
pub fn encode(model: &LoadedModel, state_ids: &[u32], p: &Prepared) -> Result<Encoded, String> {
    let head = model.tokenize(&p.head)?;
    let opts = p.options.iter().map(|o| model.tokenize(o)).collect::<Result<Vec<_>, _>>()?;
    let (ids, markers) = assemble(&model.meta, &head, &opts, state_ids);
    if markers.len() != p.options.len() {
        return Err(format!(
            "options do not fit in max_len={} (head_max_len={})",
            model.meta.max_len, model.meta.head_max_len
        ));
    }
    Ok(Encoded { ids, markers, qtype: p.qtype })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(max_len: usize, head_max_len: usize) -> LayaMeta {
        serde_json::from_value(json!({
            "max_len": max_len, "head_max_len": head_max_len, "mask_token": "[MASK]",
            "mask_id": 4, "cls_id": 1, "sep_id": 2, "pad_id": 3, "temperature": [1.0, 1.0, 1.0]
        }))
        .unwrap()
    }

    fn r(a: u32, n: u32) -> Vec<u32> {
        (a..a + n).collect()
    }

    fn q(v: Value) -> QuestionDef {
        serde_json::from_value(v).unwrap()
    }

    // Expected values produced by running Laya's build_sequence integer logic in Python.
    #[test]
    #[allow(clippy::type_complexity)]
    fn assemble_matches_python() {
        let cases: Vec<(Vec<u32>, Vec<Vec<u32>>, Vec<u32>, usize, usize, Vec<u32>, Vec<u32>)> = vec![
            (
                r(10, 5),
                vec![r(20, 2), r(30, 2), r(40, 2)],
                r(100, 10),
                64,
                32,
                vec![
                    1, 10, 11, 12, 13, 14, 2, 4, 20, 21, 4, 30, 31, 4, 40, 41, 2, 100, 101, 102, 103, 104, 105, 106,
                    107, 108, 109, 2,
                ],
                vec![7, 10, 13],
            ),
            (
                r(10, 12),
                vec![r(20, 6), r(30, 6), r(40, 6)],
                r(100, 5),
                64,
                20,
                vec![
                    1, 10, 11, 12, 13, 14, 15, 16, 17, 2, 4, 20, 21, 22, 4, 30, 31, 32, 4, 40, 41, 42, 2, 100, 101,
                    102, 103, 104, 2,
                ],
                vec![10, 14, 18],
            ),
            (
                r(10, 3),
                vec![r(20, 1), r(30, 1)],
                r(100, 50),
                16,
                32,
                vec![1, 10, 11, 12, 2, 4, 20, 4, 30, 2, 100, 101, 102, 103, 104, 2],
                vec![5, 7],
            ),
            (
                r(10, 3),
                vec![r(20, 3), r(30, 3), r(40, 3)],
                r(100, 5),
                10,
                32,
                vec![1, 10, 11, 12, 2, 4, 20, 21, 22, 4],
                vec![5, 9],
            ),
            (
                r(10, 2),
                vec![r(200, 60), r(30, 1)],
                vec![],
                128,
                128,
                [vec![1, 10, 11, 2, 4], r(200, 48), vec![4, 30, 2, 2]].concat(),
                vec![4, 53],
            ),
        ];
        for (head, opts, st, ml, hml, want_ids, want_markers) in cases {
            let (ids, markers) = assemble(&meta(ml, hml), &head, &opts, &st);
            assert_eq!(ids, want_ids);
            assert_eq!(markers, want_markers);
        }
    }

    #[test]
    fn prepare_choice_object_renders_like_laya() {
        let p = prepare(
            &q(json!({"type": "choice", "instructions": "Pick [MASK] one", "criteria": {"approve": "ok it", "deny": "", "hold": {"days": 3}, "zero": 0}})),
            "[MASK]",
        )
        .unwrap();
        assert_eq!(p.head, "choice question: Pick   one");
        assert_eq!(p.options, vec![" approve: ok it", " deny", " hold: {\"days\": 3}", " zero: 0"]);
        assert_eq!(p.labels, vec!["approve", "deny", "hold", "zero"]);
    }

    #[test]
    fn prepare_choice_list_dedupes_in_order() {
        let p =
            prepare(&q(json!({"type": "choice", "instructions": "x", "criteria": ["b", "a", "b"]})), "[MASK]").unwrap();
        assert_eq!(p.labels, vec!["b", "a"]);
        assert_eq!(p.options, vec![" b", " a"]);
    }

    #[test]
    fn prepare_score_and_noul() {
        let s = prepare(&q(json!({"type": "score", "instructions": "rate", "criteria": ["low", "high"]})), "[MASK]")
            .unwrap();
        assert_eq!(s.options, vec![" level 0: low", " level 1: high"]);
        let n = prepare(&q(json!({"type": "noul", "instructions": "is it?"})), "[MASK]").unwrap();
        assert_eq!(n.options, vec![" false: no, the statement does not hold", " true: yes, the statement holds"]);
        let n2 =
            prepare(&q(json!({"type": "noul", "instructions": "x", "criteria": {"true": "yes!"}})), "[MASK]").unwrap();
        assert_eq!(n2.options[1], " true: yes!");
    }

    #[test]
    fn non_string_instructions_are_ascii_json() {
        let p = prepare(&q(json!({"type": "noul", "instructions": {"rule": "café"}})), "[MASK]").unwrap();
        assert_eq!(p.head, "noul question: {\"rule\": \"caf\\u00e9\"}");
    }

    #[test]
    fn rejects_bad_questions() {
        assert!(prepare(&q(json!({"type": "choice", "instructions": "x", "criteria": {"a": 1}})), "[MASK]").is_err());
        assert!(prepare(&q(json!({"type": "score", "instructions": "x", "criteria": {"a": 1}})), "[MASK]").is_err());
        assert!(prepare(&q(json!({"type": "maybe", "instructions": "x"})), "[MASK]").is_err());
    }

    #[test]
    fn state_segment_keeps_order_and_unicode() {
        assert_eq!(state_segment(&json!({"b": "é", "a": 1}), "[MASK]"), "{\"b\": \"é\", \"a\": 1}");
        assert_eq!(state_segment(&json!("a [MASK] b"), "[MASK]"), "a   b");
    }
}
