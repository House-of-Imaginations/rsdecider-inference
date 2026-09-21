//! Redis-backed L2 cache and idempotency. Needs Docker.
//! Run: cargo test --features redis-tests --test redis
#![cfg(feature = "redis-tests")]
mod common;

use axum::http::StatusCode;
use common::*;
use rsdecider::fake::Fake;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Duration;
use testcontainers_modules::{redis::Redis, testcontainers::runners::AsyncRunner};

#[tokio::test]
async fn l2_survives_restart_and_outage_degrades_to_l1() {
    let node = Redis::default().start().await.unwrap();
    let url = format!("redis://127.0.0.1:{}", node.get_host_port_ipv4(6379).await.unwrap());
    let cfg = || config(|c| c.cache.redis_url = Some(url.clone()));

    let fake1 = Fake::default();
    let (app1, _) = app(cfg(), &fake1).await;
    assert_eq!(call(&app1, "/v1/decide", decide_body("x", "q"), &[auth()]).await.0, StatusCode::OK);
    tokio::time::sleep(Duration::from_millis(100)).await; // L2 write is spawned after completion

    // a fresh process (empty L1) gets an L2 hit
    let fake2 = Fake::default();
    let (app2, _) = app(cfg(), &fake2).await;
    let (s, _, b) = call(&app2, "/v1/decide", decide_body("x", "q"), &[auth()]).await;
    assert_eq!((s, &b["answers"]["q"]["cached"]), (StatusCode::OK, &serde_json::json!(true)));
    assert_eq!(fake2.items.load(SeqCst), 0);

    // Redis down: requests still succeed (L1 + in-process idempotency)
    node.stop().await.unwrap();
    let h = [auth(), ("idempotency-key", "during-outage")];
    assert_eq!(call(&app2, "/v1/decide", decide_body("y", "q"), &h).await.0, StatusCode::OK);
    let (s, _, _) = call(&app2, "/v1/decide", decide_body("y", "q"), &h).await;
    assert_eq!(s, StatusCode::OK, "replayed from the in-process store");
}

#[tokio::test]
async fn idempotency_is_shared_across_instances() {
    let node = Redis::default().start().await.unwrap();
    let url = format!("redis://127.0.0.1:{}", node.get_host_port_ipv4(6379).await.unwrap());
    let fake = Fake { delay: Duration::from_millis(300), ..Fake::default() };
    let (a, _) = app(config(|c| c.cache.redis_url = Some(url.clone())), &fake).await;
    let (b, _) = app(config(|c| c.cache.redis_url = Some(url.clone())), &fake).await;
    let h = [auth(), ("idempotency-key", "shared")];
    let first = {
        let a = a.clone();
        tokio::spawn(async move {
            call(&a, "/v1/decide", decide_body("z", "q"), &[auth(), ("idempotency-key", "shared")]).await
        })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(call(&b, "/v1/decide", decide_body("z", "q"), &h).await.0, StatusCode::CONFLICT);
    let (_, _, body1) = first.await.unwrap();
    let (s, _, body2) = call(&b, "/v1/decide", decide_body("z", "q"), &h).await;
    assert_eq!((s, body2), (StatusCode::OK, body1));
}
