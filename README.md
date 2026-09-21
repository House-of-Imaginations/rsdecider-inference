<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="rsdecider: Laya decisions over HTTP. Batched, cached, rate-limited, and shed before the box tips over.">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#endpoints">Endpoints</a> ·
  <a href="#configure-rsdecidertoml">Configuration</a> ·
  <a href="#self-hosted-docker">Self-hosted</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="./openapi.yaml">OpenAPI</a>
</p>

**rsdecider** is a Rust (tokio + axum) server for [Laya](https://github.com/NandhaKishorM/laya) decision models: you send a
situation (`state`) and a few structured questions, and it returns calibrated choices, yes/no probabilities and scores.
It routes each request to Laya's English or multilingual model, batches work onto ONNX Runtime, caches every answer, and
rejects early with `529` when the machine is full instead of letting latency explode.

```bash
curl -s localhost:3000/v1/decide -H 'authorization: Bearer dev-key' -H 'content-type: application/json' -d '{
  "state": "Customer was charged twice for invoice #4411 and is asking for a refund.",
  "questions": {
    "action":   {"type": "choice", "instructions": "How should support handle this?",
                 "criteria": {"approve": "issue the refund", "deny": "refuse the refund", "escalate": "send to a human agent"}},
    "urgent":   {"type": "noul",   "instructions": "Is this urgent?"},
    "severity": {"type": "score",  "instructions": "How severe is the issue?", "criteria": ["low", "medium", "high"]}
  }}'
```

<details>
<summary><b>Response</b> — real output from <code>laya-english</code> (click to expand)</summary>

```json
{
  "id": "req_01M31BAVX63WFSFXX0WQWM0FAK",
  "model": "laya-english@c3e0839d173a",
  "answers": {
    "action":   { "type": "choice", "choice": "escalate",
                  "probabilities": { "approve": 0.4047, "deny": 0.1535, "escalate": 0.4417 },
                  "confidence": 0.0764, "action": { "act_probability": 1.0 }, "cached": false },
    "urgent":   { "type": "noul", "noul": 0.7262, "confidence": 0.7262,
                  "action": { "act_probability": 1.0 }, "cached": false },
    "severity": { "type": "score", "score": 1.1731, "legend": { "0": "low", "1": "medium", "2": "high" },
                  "probabilities": { "0": 0.1098, "1": 0.6073, "2": 0.2829 },
                  "confidence": 0.1784, "action": { "act_probability": 1.0 }, "cached": false }
  },
  "usage":     { "input_tokens": 140, "output_tokens": 0 },
  "routing":   { "model": "english", "reason": "detected:english" },
  "timing_ms": { "tokenize": 6.8, "wait": 582.6, "total": 589.4 }
}
```

The same request again returns in `0.1 ms` with `"cached": true`. A Vietnamese `state` goes to
`laya-multilingual` with `"reason": "detected:non_english"`.
</details>

<p align="center">
  <img src="./assets/readme/benchmarks.svg" width="100%" alt="Stress results on an Apple M1 Pro: cache hits p99 1.4 ms at 2,000 req/s; cold p99 575 ms at capacity; zero 5xx or timeouts at about 30× capacity, excess shed as 529.">
</p>

## What you get

| | |
|---|---|
| **Laya routing mode** | Script/language detection sends English to `english`, everything else to `multilingual`; `"model"` overrides it. |
| **Controlled concurrency** | Per-model queue bound (`max_pending`), all-or-nothing admission, deadline-aware shedding (`529` + `Retry-After`) and a fixed ORT thread budget sized to your CPUs. |
| **Micro-batching** | Questions from concurrent requests are packed into padded batches (`max_batch_items`, `max_batch_tokens`, `max_wait_ms`). |
| **Coalescing** | 100 identical in-flight questions run the model once. |
| **Two-tier cache** | L1 in-process (moka, byte-bounded) and optional L2 Redis shared across instances. |
| **API keys + rate limits** | Only SHA-256 hashes live in config; per-key token bucket (`rps`, `burst`), a batch costs one token per item. `kill -HUP` reloads `[[keys]]` without a restart. |
| **Idempotency** | `Idempotency-Key` replays the stored response; Redis `SET NX` across instances, bounded in-process fallback. |
| **Operability** | Prometheus metrics, `/healthz` + `/readyz`, request ids in logs and `x-request-id`, worker panics isolated. |

## Quick start

**1. Export the models** (once; needs Python 3.10+). Writes `models/<name>/{model.onnx, tokenizer.json, laya.json}` and
self-checks ONNX Runtime against PyTorch.

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/export_onnx.py --out models/english
.venv/bin/python tools/export_onnx.py --subfolder multilingual --out models/multilingual
```

**2. Create an API key and a config.**

```bash
cargo build --release
./target/release/rsdecider hash-key 'my-secret-key'      # → paste into [[keys]].sha256
cp rsdecider.example.toml rsdecider.toml
```

**3. Run it.**

```bash
./target/release/rsdecider serve --config rsdecider.toml
curl localhost:3000/readyz                                 # → ready
```

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/decide` | Bearer key | Answer every question about one `state`. |
| `POST` | `/v1/decide/batch` | Bearer key | `{"items": [ …decide bodies… ]}` — up to `limits.max_batch_states` states, each routed on its own. Returns `{"id", "results": [...]}`. |
| `GET` | `/healthz` | — | Liveness: `ok`. |
| `GET` | `/readyz` | — | `ready` while every model has a live worker, else `503`. |
| `GET` | `/docs` | — | Swagger UI — browse the API and try requests (click **Authorize**, paste your key). |
| `GET` | `/openapi.yaml` | — | The OpenAPI 3.1 document, embedded in the binary. |
| `GET` | `:9000/metrics` | — | Prometheus exposition (separate listener, `server.metrics_listen`). |

Full schema: **[`openapi.yaml`](./openapi.yaml)** (OpenAPI 3.1), also served live at `http://localhost:3000/docs` (Swagger UI,
loaded from jsDelivr) and `/openapi.yaml` for Postman or SDK generators.

<details>
<summary><b>Question types</b></summary>

| `type` | `criteria` | Answer fields |
|---|---|---|
| `choice` | list `["a","b"]` or map `{"label": "description"}` | `choice`, `probabilities` per label, `confidence` |
| `score` | ordered list, e.g. `["low","medium","high"]` | `score` (expected index), `legend`, `probabilities`, `confidence` |
| `noul` | optional map, e.g. `{"true": "..."}` | `noul` (probability of yes), `confidence` |

Every answer also carries `action.act_probability` and `cached`. `state` and `instructions` may be strings or any JSON value.
</details>

<details>
<summary><b>Status codes</b></summary>

Errors are `{"error": {"code", "message"}}`. Checks run in this order:

| Status | `code` | When |
|---|---|---|
| `401` | `unauthorized` | Missing or unknown bearer key |
| `413` | `payload_too_large` | Body over `limits.max_body_bytes` |
| `422` | `invalid_request` | Bad JSON/field (message names the path, e.g. `items[1].`), unknown model, non-ASCII `Idempotency-Key`, key reused with a different body |
| `429` | `rate_limited` | Per-key bucket empty — honour `Retry-After` |
| `409` | `idempotency_in_progress` | Same `Idempotency-Key` still running — honour `Retry-After` |
| `529` | `overloaded` | Queue can't finish this work before the deadline — honour `Retry-After` |
| `504` | `deadline_exceeded` | Passed `server.request_timeout_ms` |
| `500` | `internal` | Inference failed (e.g. worker panic); the next request is served normally |
</details>

## Configure `rsdecider.toml`

Start from [`rsdecider.example.toml`](./rsdecider.example.toml). Only `[routing]` and `[[models]]` are required
(with no `[[keys]]`, every call gets `401`), so a minimal file is:

```toml
[routing]
english_model = "english"
non_english_model = "multilingual"

[[models]]
name = "english"
path = "models/english"

[[models]]
name = "multilingual"
path = "models/multilingual"

[[keys]]
name = "my-app"
sha256 = "<output of: rsdecider hash-key my-secret-key>"
rps = 50
burst = 100
```

<details open>
<summary><b>Every option, with defaults</b></summary>

```toml
[server]
listen             = "0.0.0.0:3000"
metrics_listen     = "127.0.0.1:9000"   # Prometheus; keep it private
worker_threads     = 2                  # tokio threads for HTTP (>= 1)
tokenize_threads   = 2                  # concurrent tokenizations on the blocking pool
request_timeout_ms = 10000              # end-to-end deadline → 504

[limits]
max_body_bytes          = 1048576       # → 413
max_state_chars         = 100000        # → 422
max_batch_states        = 8             # items per /v1/decide/batch
max_questions_per_state = 16

[routing]                               # Laya routing mode
english_model     = "english"           # must name a [[models]] entry
non_english_model = "multilingual"

[[models]]                              # one table per model
name               = "english"
path               = "models/english"   # model.onnx + tokenizer.json + laya.json
execution_provider = "cpu"              # or "cuda" (build with --features cuda)
workers            = 1                  # ORT sessions, one blocking thread each
intra_op_threads   = 6                  # threads per session
max_pending        = 256                # queued questions before 529 (memory bound)
max_batch_items    = 8                  # questions per forward pass
max_batch_tokens   = 8192               # padded tokens per forward pass
max_wait_ms        = 2                  # how long a batch waits to fill

[cache]
l1_max_bytes         = 268435456        # in-process answer cache (256 MiB)
l1_ttl_secs          = 3600
redis_url            = "redis://127.0.0.1:6379"   # optional: enables L2 + shared idempotency
l2_ttl_secs          = 86400
idempotency_ttl_secs = 86400

[[keys]]                                # one table per client
name   = "my-app"                       # shows up in metrics and logs
sha256 = "…64 lowercase hex…"           # rsdecider hash-key <secret>
rps    = 50                             # sustained questions/s (batch items count individually)
burst  = 100
```

The server refuses to start on an invalid file (unknown routing target, duplicate names, zero workers, bad key hash)
and names the offending field. Edit `[[keys]]` and send `SIGHUP` to add, rotate or revoke keys live (other sections need a restart).
</details>

### Sizing for your machine

- **CPU budget:** `Σ(workers × intra_op_threads) + tokenize_threads + worker_threads ≈ vCPUs`. Giving the busier model more
  `intra_op_threads` usually beats adding `workers`. The stress config for a 10-core M1 Pro uses english `6`,
  multilingual `2`, tokenize `1`, HTTP `2` ([`stress/rsdecider.stress.toml`](./stress/rsdecider.stress.toml)).
- **`max_pending ≈ items/s × 1–2 s`.** It bounds memory; latency is protected separately because admission estimates
  queue time (EMA seconds/item × depth ÷ workers) and returns `529` up front when work can't finish by the deadline.
- **GPU:** build with `cargo build --release --features cuda` and set `execution_provider = "cuda"`.
- **More than one instance:** set `redis_url` so the cache and idempotency are shared.

## Self-hosted (Docker)

Everything for a container deployment lives in [`self-hosted/`](./self-hosted): a two-stage `Dockerfile`, a
`docker-compose.yml` with Redis (512 MB, LRU), and the container config `rsdecider.toml`.

```bash
# export models into ./models first (see Quick start), then:
cd self-hosted
docker compose up --build
curl localhost:3000/readyz
```

Compose mounts `../models` read-only at `/models` and `self-hosted/rsdecider.toml` at `/etc/rsdecider/rsdecider.toml`;
edit that file (keys, threads, `max_pending`) and restart. Metrics are published on `127.0.0.1:9000` only. The file
ships with demo keys `dev-key` and `stress-key` — **replace them before exposing the port.**

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/readme/system-dark.png">
  <img src="./assets/readme/system-light.png" width="100%" alt="System architecture: axum edge guards, idempotency, Laya router, tokenizer pool, L1/L2 cache, scheduler with admission and micro-batcher, per-model ORT worker pools, Redis and Prometheus.">
</picture>

A request is authenticated, parsed, charged against its key's rate limit and checked for an `Idempotency-Key`. It is
then routed to a model, and each question is looked up in the cache — only misses are tokenized and sent to the
scheduler. The scheduler joins identical in-flight questions, admits the rest all-or-nothing against the model's queue
(shedding with `529` when the deadline can't be met), and a per-model micro-batcher feeds padded batches to blocking ONNX
Runtime workers. Answers are post-processed into Laya's format, written to L1 and (asynchronously) Redis, and returned.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/readme/request-flow-dark.png">
  <img src="./assets/readme/request-flow-light.png" width="100%" alt="Request flow for POST /v1/decide: edge gates, idempotency, validate and route, cache lookup, tokenize misses, coalesce and admit, micro-batcher, ORT worker, postprocess, 200 JSON — with 4xx, 409, 500, 504 and 529 exits.">
</picture>

> Interactive versions (pan, zoom, search, trace a path, export): [`docs/diagrams/system.html`](./docs/diagrams/system.html)
> and [`docs/diagrams/request-flow.html`](./docs/diagrams/request-flow.html) — open locally in a browser. Sources are the
> neighbouring `*.json` files ([archify](https://github.com/tt-a1i/archify) specs).

## Performance

Measured with k6 against the real fp32 models on an Apple M1 Pro (10 cores), CPU execution provider, no Redis.

| Scenario | Result |
|---|---|
| Cache hits, 3,000 req/s offered | 2,000 req/s served, p99 **1.4 ms**, 0 errors |
| Cold, 3 req/s × 3 questions | p99 **575 ms**, 0 errors |
| Cold, 6 req/s | 15% shed as `529`, accepted p99 6.5 s (inside the 10 s deadline) |
| Overload, 100 req/s (~30× capacity) | only `200` and `529` — **0 timeouts, 0 5xx**, queue drains to 0 |

Cold capacity is inference-bound (~10 questions/s English, ~4/s multilingual). The biggest lever is the model, not the
server: an int8 export (`tools/export_onnx.py --quantize int8`) or a GPU/CoreML execution provider. Full numbers:
[`stress/results/2026-09-21-m1pro.md`](./stress/results/2026-09-21-m1pro.md). Re-run with `stress/run.sh` against a
running server (`-e RATE=…` per scenario; see the script).

## Development

```bash
cargo test                                                        # unit + API tests (fake backend, no models)
MODEL_DIR=models/english cargo test --release --features parity --test parity   # ONNX vs PyTorch fixtures
cargo test --features redis-tests --test redis                    # needs Docker (testcontainers)
(cd self-hosted && docker compose up -d) && cargo test --features e2e --test e2e
./target/release/rsdecider serve --fake-delay-ms 20               # server without models, for overhead tests
```

<details>
<summary><b>Project layout</b></summary>

```text
src/
  main.rs          CLI (serve, hash-key), runtime + metrics setup
  api.rs           axum routes, request pipeline, errors
  docs.html        Swagger UI page served at /docs
  auth.rs          hashed API keys + per-key governor limiter
  idempotency.rs   Redis SET NX with bounded moka fallback
  lang.rs          Laya routing (English vs multilingual)
  cache.rs         L1 moka + L2 Redis answer cache
  scheduler.rs     coalescing, admission, micro-batcher, ORT worker pool
  model/           tokenization, Laya sequence encoding, ONNX session, postprocess
  config.rs        rsdecider.toml schema + validation
tests/             api, parity, redis, e2e suites
tools/             export_onnx.py (PyTorch → ONNX + fixtures)
stress/            k6 scenarios, runner, results
self-hosted/       Dockerfile, docker-compose.yml, container config
docs/diagrams/     archify sources + interactive HTML
openapi.yaml       HTTP API
```
</details>

## Known limits

- Batches are admitted all-or-nothing, so under contention large `/v1/decide/batch` calls lose to single requests.
- The admission estimate (EMA) adapts upward quickly but has no explicit decay after a slow spell.
- The Redis and Docker paths are covered by tests that need Docker; they were compile-checked, not run, for this release.
