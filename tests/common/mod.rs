#![allow(dead_code)]
use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use rsdecider::api::{self, AppState};
use rsdecider::config::Config;
use rsdecider::fake::Fake;
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

pub const KEY: &str = "test";
/// Token id of "panic" in tests/data/tiny-model/tokenizer.json.
pub const PANIC_TOKEN: u32 = 28;

pub fn config(edit: impl FnOnce(&mut Config)) -> Config {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tiny-model");
    let mut c = Config::from_toml_str(&format!(
        r#"
[routing]
english_model = "english"
non_english_model = "multilingual"

[[models]]
name = "english"
path = "{dir}"

[[models]]
name = "multilingual"
path = "{dir}"

[[keys]]
name = "tester"
sha256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
rps = 1000
burst = 1000
"#
    ))
    .unwrap();
    edit(&mut c);
    c
}

pub async fn app(cfg: Config, fake: &Fake) -> (Router, Arc<AppState>) {
    let f = fake.clone();
    let st = api::build(cfg, &move |_, _| f.factory()).await.unwrap();
    (api::router(st.clone()), st)
}

pub async fn call(app: &Router, path: &str, body: Value, headers: &[(&str, &str)]) -> (StatusCode, HeaderMap, Value) {
    let mut req = Request::post(path).header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let res = app.clone().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
    let (parts, body) = res.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    (parts.status, parts.headers, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

pub fn auth() -> (&'static str, &'static str) {
    ("authorization", "Bearer test")
}

pub fn decide_body(state: &str, qid: &str) -> Value {
    json!({
        "state": {"body": state},
        "questions": {
            qid: {"type": "choice", "instructions": "refund the customer", "criteria": {"approve": "", "deny": "", "escalate": ""}},
        }
    })
}
