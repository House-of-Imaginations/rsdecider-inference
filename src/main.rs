use clap::{Parser, Subcommand};
use rsdecider::config::{Config, ModelCfg};
use rsdecider::fake::Fake;
use rsdecider::model::LoadedModel;
use rsdecider::model::session::OrtBackend;
use rsdecider::scheduler::{Backend, BackendFactory};
use rsdecider::{api, auth, metrics};
use std::future::IntoFuture;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser)]
#[command(version, about = "Concurrent inference server for Laya decision models")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTP server.
    Serve {
        #[arg(long, default_value = "rsdecider.toml")]
        config: PathBuf,
        /// Use the deterministic fake backend with this per-batch latency (stress-testing server overhead).
        #[arg(long)]
        fake_delay_ms: Option<u64>,
    },
    /// Print the SHA-256 of an API key for the [[keys]] table.
    HashKey { key: String },
}

fn main() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::HashKey { key } => {
            println!("{}", auth::hash_key(&key));
            Ok(())
        }
        Cmd::Serve { config, fake_delay_ms } => {
            let cfg = Config::load(&config)?;
            // Must run before any Session is built, so every model picks up the shared pool.
            if let Some(n) = cfg.knobs.ort_global_threads {
                let pool = ort::environment::GlobalThreadPoolOptions::default()
                    .with_intra_threads(n)
                    .and_then(|p| p.with_spin_control(false))
                    .map_err(|e| e.to_string())?;
                ort::init().with_global_thread_pool(pool).commit();
            }
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(cfg.server.worker_threads)
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?
                .block_on(serve(cfg, config, fake_delay_ms))
        }
    }
}

fn ort_factory(m: &ModelCfg, _lm: &LoadedModel, arena_shrink: bool, global_pool: bool) -> BackendFactory {
    let (path, ep, intra) = (m.path.join("model.onnx"), m.execution_provider.clone(), m.intra_op_threads);
    Arc::new(move || {
        OrtBackend::new(&path, &ep, intra, arena_shrink, global_pool).map(|b| Box::new(b) as Box<dyn Backend>)
    })
}

async fn serve(cfg: Config, path: PathBuf, fake_delay_ms: Option<u64>) -> Result<(), String> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let prom = metrics::install()?;
    // Histograms only drain on render(); upkeep keeps them bounded when nothing scrapes /metrics.
    let upkeep = prom.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(5));
        loop {
            tick.tick().await;
            upkeep.run_upkeep();
        }
    });

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = cfg.server.worker_threads
        + cfg.server.tokenize_threads
        + match cfg.knobs.ort_global_threads {
            Some(n) => n,
            None => cfg.models.iter().map(|m| m.workers * m.intra_op_threads).sum::<usize>(),
        };
    if threads > cores {
        tracing::warn!("thread budget {threads} exceeds {cores} cores; expect contention");
    }
    for m in &cfg.models {
        let size = std::fs::metadata(m.path.join("model.onnx")).map(|x| x.len()).unwrap_or(0);
        tracing::info!(model = %m.name, "estimated resident weights: {} MiB", (m.workers as u64 * size) >> 20);
    }

    let st = match fake_delay_ms {
        Some(ms) => {
            tracing::warn!("serving the FAKE backend ({ms} ms per batch): answers are not real model output");
            let fake = Fake { delay: Duration::from_millis(ms), ..Fake::default() };
            api::build(cfg.clone(), &move |_, _| fake.factory()).await?
        }
        None => {
            let (arena_shrink, global_pool) = (cfg.knobs.ort_arena_shrink, cfg.knobs.ort_global_threads.is_some());
            api::build(cfg.clone(), &move |m, lm| ort_factory(m, lm, arena_shrink, global_pool)).await?
        }
    };

    #[cfg(unix)]
    {
        let auth = st.auth.clone();
        tokio::spawn(async move {
            let mut hup =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()).expect("SIGHUP handler");
            while hup.recv().await.is_some() {
                match Config::load(&path) {
                    Ok(c) => {
                        auth.reload(&c.keys);
                        tracing::info!("reloaded {} keys", c.keys.len());
                    }
                    Err(e) => tracing::error!("key reload failed, keeping old keys: {e}"),
                }
            }
        });
    }

    let listener =
        tokio::net::TcpListener::bind(&cfg.server.listen).await.map_err(|e| format!("{}: {e}", cfg.server.listen))?;
    let mlistener = tokio::net::TcpListener::bind(&cfg.server.metrics_listen)
        .await
        .map_err(|e| format!("{}: {e}", cfg.server.metrics_listen))?;
    tracing::info!("listening on {} (metrics on {})", cfg.server.listen, cfg.server.metrics_listen);

    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let stopped = |mut rx: tokio::sync::watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|s| *s).await;
    };
    let app = axum::serve(listener, api::router(st)).with_graceful_shutdown(stopped(stop_rx.clone()));
    let mapp = axum::serve(mlistener, metrics::router(prom)).with_graceful_shutdown(stopped(stop_rx));
    let servers = tokio::spawn(async move { tokio::join!(app.into_future(), mapp.into_future()) });

    shutdown_signal().await;
    tracing::info!("shutting down, draining in-flight requests");
    let _ = stop_tx.send(true);
    let drain = Duration::from_millis(cfg.server.request_timeout_ms) + Duration::from_secs(1);
    if tokio::time::timeout(drain, servers).await.is_err() {
        tracing::warn!("drain timeout, exiting with requests in flight");
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.expect("ctrl-c handler") };
    #[cfg(unix)]
    let term = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}
