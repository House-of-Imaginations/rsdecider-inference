mod common;

use axum::http::StatusCode;
use common::*;
use rsdecider::fake::Fake;
use serde_json::json;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Duration;

#[tokio::test]
async fn decide_returns_laya_shaped_answer() {
    let fake = Fake::default();
    let (app, _) = app(config(|_| {}), &fake).await;
    let (status, headers, body) =
        call(&app, "/v1/decide", decide_body("customer wants a refund", "action"), &[auth()]).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(headers["x-request-id"], body["id"].as_str().unwrap());
    let a = &body["answers"]["action"];
    assert_eq!(a["type"], "choice");
    assert!(["approve", "deny", "escalate"].contains(&a["choice"].as_str().unwrap()));
    assert_eq!(a["cached"], false);
    assert_eq!(body["routing"], json!({"model": "english", "reason": "detected:english"}));
    assert!(body["model"].as_str().unwrap().starts_with("laya-english@"));
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn missing_or_unknown_key_is_401() {
    let (app, _) = app(config(|_| {}), &Fake::default()).await;
    let (s, _, b) = call(&app, "/v1/decide", decide_body("x", "q"), &[]).await;
    assert_eq!((s, b["error"]["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("unauthorized")));
    let (s, _, _) = call(&app, "/v1/decide", decide_body("x", "q"), &[("authorization", "Bearer nope")]).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn oversized_body_is_413() {
    let (app, _) = app(config(|c| c.limits.max_body_bytes = 256), &Fake::default()).await;
    let (s, _, b) = call(&app, "/v1/decide", decide_body(&"a ".repeat(500), "q"), &[auth()]).await;
    assert_eq!((s, b["error"]["code"].as_str()), (StatusCode::PAYLOAD_TOO_LARGE, Some("payload_too_large")));
}

#[tokio::test]
async fn invalid_requests_are_422() {
    let (app, _) = app(config(|_| {}), &Fake::default()).await;
    let bad_type = json!({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "x"}}});
    let (s, _, b) = call(&app, "/v1/decide", bad_type, &[auth()]).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(b["error"]["message"].as_str().unwrap().contains("questions.q"));
    let unknown_model =
        json!({"state": "x", "model": "nope", "questions": {"q": {"type": "noul", "instructions": "x"}}});
    assert_eq!(call(&app, "/v1/decide", unknown_model, &[auth()]).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    // one bad item fails the whole batch and names its index
    let batch = json!({"items": [decide_body("fine", "q"), {"state": 5, "questions": {"q": {"type": "noul", "instructions": "x"}}}]});
    let (s, _, b) = call(&app, "/v1/decide/batch", batch, &[auth()]).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(b["error"]["message"].as_str().unwrap().starts_with("items[1]."));
}

#[tokio::test]
async fn options_overflowing_max_len_are_422() {
    // tiny model: max_len 64, head_max_len 32 → 40 options cannot all keep a marker
    let (app, _) = app(config(|_| {}), &Fake::default()).await;
    let crit: Vec<String> = (0..40).map(|i| format!("option{i}")).collect();
    let body = json!({"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": crit}}});
    let (s, _, b) = call(&app, "/v1/decide", body, &[auth()]).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{b}");
}

#[tokio::test]
async fn rate_limit_is_429_with_retry_after() {
    let (app, _) = app(
        config(|c| {
            c.limits.max_batch_states = 2;
            c.keys[0].rps = 1;
            c.keys[0].burst = 2;
        }),
        &Fake::default(),
    )
    .await;
    for _ in 0..2 {
        assert_eq!(call(&app, "/v1/decide", decide_body("x", "q"), &[auth()]).await.0, StatusCode::OK);
    }
    let (s, h, _) = call(&app, "/v1/decide", decide_body("x", "q"), &[auth()]).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h["retry-after"].to_str().unwrap().parse::<u64>().unwrap() >= 1);
}

#[tokio::test]
async fn full_queue_is_529_and_cache_hits_still_pass() {
    let fake = Fake { delay: Duration::from_millis(300), ..Fake::default() };
    let (app, _) = app(
        config(|c| {
            c.limits.max_batch_states = 1;
            c.limits.max_questions_per_state = 1;
            c.models.iter_mut().for_each(|m| m.max_pending = 1);
        }),
        &fake,
    )
    .await;
    // warm the cache for "cached", then occupy the single permit with "busy"
    assert_eq!(call(&app, "/v1/decide", decide_body("cached", "q"), &[auth()]).await.0, StatusCode::OK);
    let busy = {
        let app = app.clone();
        tokio::spawn(async move { call(&app, "/v1/decide", decide_body("busy", "q"), &[auth()]).await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (s, h, b) = call(&app, "/v1/decide", decide_body("other", "q"), &[auth()]).await;
    assert_eq!((s.as_u16(), b["error"]["code"].as_str()), (529, Some("overloaded")));
    assert!(h.contains_key("retry-after"));
    let (s, _, b) = call(&app, "/v1/decide", decide_body("cached", "q"), &[auth()]).await;
    assert_eq!((s, &b["answers"]["q"]["cached"]), (StatusCode::OK, &json!(true)));
    assert_eq!(busy.await.unwrap().0, StatusCode::OK);
}

#[tokio::test]
async fn slow_inference_is_504() {
    let fake = Fake { delay: Duration::from_millis(500), ..Fake::default() };
    let (app, _) = app(config(|c| c.server.request_timeout_ms = 100), &fake).await;
    let (s, _, b) = call(&app, "/v1/decide", decide_body("x", "q"), &[auth()]).await;
    assert_eq!((s, b["error"]["code"].as_str()), (StatusCode::GATEWAY_TIMEOUT, Some("deadline_exceeded")));
}

#[tokio::test]
async fn hundred_identical_requests_make_one_backend_item() {
    let fake = Fake { delay: Duration::from_millis(100), ..Fake::default() };
    let (app, _) = app(config(|_| {}), &fake).await;
    let tasks: Vec<_> = (0..100)
        .map(|_| {
            let app = app.clone();
            tokio::spawn(async move { call(&app, "/v1/decide", decide_body("same", "q"), &[auth()]).await.0 })
        })
        .collect();
    for t in tasks {
        assert_eq!(t.await.unwrap(), StatusCode::OK);
    }
    assert_eq!(fake.items.load(SeqCst), 1);
}

#[tokio::test]
async fn partial_cache_hit_infers_only_misses() {
    let fake = Fake::default();
    let (app, _) = app(config(|_| {}), &fake).await;
    call(&app, "/v1/decide", decide_body("s", "a"), &[auth()]).await;
    assert_eq!(fake.items.load(SeqCst), 1);
    let mut body = decide_body("s", "a");
    body["questions"]["b"] = json!({"type": "noul", "instructions": "the customer wants a refund"});
    let (s, _, b) = call(&app, "/v1/decide", body, &[auth()]).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["answers"]["a"]["cached"], true);
    assert_eq!(b["answers"]["b"]["cached"], false);
    assert_eq!(fake.items.load(SeqCst), 2);
}

#[tokio::test]
async fn worker_panic_fails_one_request_then_recovers() {
    let fake = Fake { panic_token: Some(PANIC_TOKEN), ..Fake::default() };
    let (app, _) = app(config(|_| {}), &fake).await;
    let (s, _, b) = call(&app, "/v1/decide", decide_body("panic", "q"), &[auth()]).await;
    assert_eq!((s, b["error"]["code"].as_str()), (StatusCode::INTERNAL_SERVER_ERROR, Some("internal")));
    assert_eq!(call(&app, "/v1/decide", decide_body("panic", "q"), &[auth()]).await.0, StatusCode::OK);
}

#[tokio::test]
async fn idempotency_replays_conflicts_and_rejects_mismatch() {
    let fake = Fake { delay: Duration::from_millis(200), ..Fake::default() };
    let (app, _) = app(config(|_| {}), &fake).await;
    let h = [auth(), ("idempotency-key", "abc")];
    let first = {
        let app = app.clone();
        tokio::spawn(async move {
            call(&app, "/v1/decide", decide_body("x", "q"), &[auth(), ("idempotency-key", "abc")]).await
        })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (s, hd, _) = call(&app, "/v1/decide", decide_body("x", "q"), &h).await;
    assert_eq!(s, StatusCode::CONFLICT);
    assert!(hd.contains_key("retry-after"));
    let (s1, _, b1) = first.await.unwrap();
    assert_eq!(s1, StatusCode::OK);
    let (s2, _, b2) = call(&app, "/v1/decide", decide_body("x", "q"), &h).await;
    assert_eq!((s2, &b2), (StatusCode::OK, &b1), "replay returns the stored response, same id");
    assert_eq!(call(&app, "/v1/decide", decide_body("different", "q"), &h).await.0, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn idempotency_never_replays_errors() {
    let fake = Fake { fail_with: Some("boom".into()), ..Fake::default() };
    let (app, _) = app(config(|_| {}), &fake).await;
    let h = [auth(), ("idempotency-key", "k")];
    assert_eq!(call(&app, "/v1/decide", decide_body("x", "q"), &h).await.0, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(call(&app, "/v1/decide", decide_body("x", "q"), &h).await.0, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(fake.calls.load(SeqCst), 2, "second call recomputed instead of replaying");
}

#[tokio::test]
async fn batch_routes_each_item_independently() {
    let (app, _) = app(config(|_| {}), &Fake::default()).await;
    let french = "Le client a été facturé deux fois et il demande un remboursement pour la facture qui a été payée le mois dernier avec la carte de crédit";
    let batch = json!({"items": [
        decide_body("customer wants a refund", "q"),
        decide_body(french, "q"),
        {"state": "x", "model": "multilingual", "questions": {"q": {"type": "noul", "instructions": "x"}}},
    ]});
    let (s, _, b) = call(&app, "/v1/decide/batch", batch, &[auth()]).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let reasons: Vec<&str> =
        b["results"].as_array().unwrap().iter().map(|r| r["routing"]["reason"].as_str().unwrap()).collect();
    assert_eq!(reasons, ["detected:english", "detected:non_english", "explicit"]);
    assert_eq!(b["results"][1]["routing"]["model"], "multilingual");
}

#[tokio::test]
async fn health_endpoints_need_no_auth() {
    use tower::ServiceExt;
    let (app, _) = app(config(|_| {}), &Fake::default()).await;
    for p in ["/healthz", "/readyz"] {
        let r =
            app.clone().oneshot(axum::http::Request::get(p).body(axum::body::Body::empty()).unwrap()).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
    }
}
