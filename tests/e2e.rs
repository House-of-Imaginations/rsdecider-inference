//! Black-box HTTP suite against a running server (cd self-hosted && docker compose up).
//! Run: RSD_URL=http://127.0.0.1:3000 RSD_KEY=dev-key cargo test --features e2e --test e2e
#![cfg(feature = "e2e")]

use serde_json::{Value, json};

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}

async fn post(path: &str, body: Value, key: &str) -> (u16, Value) {
    let r = reqwest::Client::new()
        .post(format!("{}{path}", env("RSD_URL", "http://127.0.0.1:3000")))
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Value::Null))
}

fn body(state: &str) -> Value {
    json!({"state": state, "questions": {
        "action": {"type": "choice", "instructions": "How should we handle this?", "criteria": {"approve": "refund", "deny": "no refund", "escalate": "human review"}},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "severity": {"type": "score", "instructions": "Severity", "criteria": ["low", "medium", "high"]}
    }})
}

#[tokio::test]
async fn decide_then_cached_repeat() {
    let key = env("RSD_KEY", "dev-key");
    let state = format!("Charged twice for invoice {}; wants a refund.", ulid::Ulid::generate());
    let (s, first) = post("/v1/decide", body(&state), &key).await;
    assert_eq!(s, 200, "{first}");
    for q in ["action", "urgent", "severity"] {
        assert_eq!(first["answers"][q]["cached"], false);
    }
    let (_, again) = post("/v1/decide", body(&state), &key).await;
    assert_eq!(again["answers"]["action"]["cached"], true);
    assert_eq!(again["answers"]["action"]["probabilities"], first["answers"]["action"]["probabilities"]);
}

#[tokio::test]
async fn routes_non_english_and_rejects_bad_key() {
    let key = env("RSD_KEY", "dev-key");
    let (s, r) = post("/v1/decide", body("ग्राहक से दो बार शुल्क लिया गया"), &key).await;
    assert_eq!((s, r["routing"]["reason"].as_str()), (200, Some("detected:non_english")));
    assert_eq!(post("/v1/decide", body("x"), "wrong").await.0, 401);
}

#[tokio::test]
async fn batch_endpoint() {
    let key = env("RSD_KEY", "dev-key");
    let (s, r) =
        post("/v1/decide/batch", json!({"items": [body("refund please"), body("where is my parcel")]}), &key).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["results"].as_array().unwrap().len(), 2);
}
