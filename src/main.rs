use clap::{Parser, Subcommand};
use rsdecider::config::{Config, ModelCfg};
use rsdecider::download::{self, Action, Remote};
use rsdecider::fake::Fake;
use rsdecider::model::LoadedModel;
use rsdecider::model::session::OrtBackend;
use rsdecider::scheduler::{Backend, BackendFactory};
use rsdecider::{api, auth, metrics};
use std::future::IntoFuture;
use std::io::IsTerminal;
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
        /// Download missing or broken model folders from their `download` URL without asking.
        #[arg(long, env = "RSDECIDER_DOWNLOAD_MODELS", action = clap::ArgAction::SetTrue,
              value_parser = clap::builder::BoolishValueParser::new())]
        download_models: bool,
    },
    /// Print the SHA-256 of an API key for the [[keys]] table.
    HashKey { key: String },
    /// Verify or download model folders.
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Check every model folder against its manifest.json (sizes and SHA-256).
    Check {
        #[arg(long, default_value = "rsdecider.toml")]
        config: PathBuf,
    },
    /// Download model folders that fail the check from their `download` URL.
    Pull {
        #[arg(long, default_value = "rsdecider.toml")]
        config: PathBuf,
        /// Only this model.
        #[arg(long)]
        model: Option<String>,
    },
}

fn main() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::HashKey { key } => {
            println!("{}", auth::hash_key(&key));
            Ok(())
        }
        Cmd::Models { cmd: ModelsCmd::Check { config } } => {
            let cfg = Config::load(&config)?;
            let mut bad = 0;
            for m in &cfg.models {
                let s = download::check(&m.path, true);
                println!("{}: {s}", m.name);
                bad += usize::from(!s.usable());
            }
            if bad > 0 { Err(format!("{bad} model folder(s) failed the check")) } else { Ok(()) }
        }
        Cmd::Models { cmd: ModelsCmd::Pull { config, model } } => {
            let cfg = Config::load(&config)?;
            if let Some(n) = &model
                && !cfg.models.iter().any(|m| &m.name == n)
            {
                return Err(format!("no model named {n:?} in {}", config.display()));
            }
            block_on(async {
                for m in cfg.models.iter().filter(|m| model.as_ref().is_none_or(|n| n == &m.name)) {
                    let s = download::check(&m.path, true);
                    if s.usable() {
                        println!("{}: {s}", m.name);
                        continue;
                    }
                    let Some(url) = &m.download else {
                        return Err(format!(
                            "model {:?} at {}: {s}; no `download` URL configured, {}",
                            m.name,
                            m.path.display(),
                            download::export_hint(m)
                        ));
                    };
                    Remote::fetch(url).await?.pull(&m.name, &m.path).await?;
                    println!("{}: {}", m.name, download::check(&m.path, true));
                }
                Ok(())
            })
        }
        Cmd::Serve { config, fake_delay_ms, download_models } => {
            let cfg = Config::load(&config)?;
            ensure_models(&cfg, download_models)?;
            // Must run before any Session is built, so every model picks up the shared pool.
            if let Some(n) = cfg.knobs.ort_global_threads {
                let pool = ort::environment::GlobalThreadPoolOptions::default()
                    .with_intra_threads(n)
                    .and_then(|p| p.with_spin_control(false))
                    .map_err(|e| e.to_string())?;
                if !ort::init().with_global_thread_pool(pool).commit() {
                    return Err("ORT environment already initialised; ort_global_threads not applied".into());
                }
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

/// Downloads run on this throwaway runtime, before the server's runtime exists.
fn block_on<T>(f: impl std::future::Future<Output = Result<T, String>>) -> Result<T, String> {
    tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e| e.to_string())?.block_on(f)
}

/// Cheap (size-only) check of every model folder; pulls, asks, or fails per `download::decide`.
fn ensure_models(cfg: &Config, download_flag: bool) -> Result<(), String> {
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    for m in &cfg.models {
        let status = download::check(&m.path, false);
        if status.usable() {
            continue;
        }
        let url = m.download.as_deref().unwrap_or_default();
        let with_hint = |e: String| format!("model {:?}: {e}; or {}", m.name, download::export_hint(m));
        let what = match &status {
            download::Status::Bad { corrupt, .. } if corrupt.is_empty() => "not found".to_string(),
            s => format!("not usable ({s})"),
        };
        match download::decide(m, &status, download_flag, interactive) {
            Action::Fail(e) => return Err(e),
            Action::Pull => {
                block_on(async { Remote::fetch(url).await?.pull(&m.name, &m.path).await }).map_err(with_hint)?
            }
            Action::Prompt => block_on(async {
                let remote = Remote::fetch(url).await.map_err(with_hint)?;
                eprint!(
                    "Model {:?} {what} at {}. Download {} MB from {url}? [y/N] ",
                    m.name,
                    m.path.display(),
                    remote.total_bytes() >> 20
                );
                let mut answer = String::new();
                std::io::stdin().read_line(&mut answer).map_err(|e| e.to_string())?;
                if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
                    return Err(format!("model {:?}: download declined; {}", m.name, download::export_hint(m)));
                }
                remote.pull(&m.name, &m.path).await.map_err(with_hint)
            })?,
        }
    }
    Ok(())
}

fn ort_factory(m: &ModelCfg, _lm: &LoadedModel, global_pool: bool) -> BackendFactory {
    let (path, ep, intra) = (m.path.join("model.onnx"), m.execution_provider.clone(), m.intra_op_threads);
    Arc::new(move || OrtBackend::new(&path, &ep, intra, global_pool).map(|b| Box::new(b) as Box<dyn Backend>))
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
            // Each concurrent Run's calling thread computes alongside the pool's n - 1 threads.
            Some(n) => n - 1 + cfg.models.iter().map(|m| m.workers).sum::<usize>(),
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
            let global_pool = cfg.knobs.ort_global_threads.is_some();
            api::build(cfg.clone(), &move |m, lm| ort_factory(m, lm, global_pool)).await?
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_models_env_accepts_boolish_values() {
        let flag = |v: Option<&str>| {
            // SAFETY: the only test in this binary; nothing else reads or writes the environment concurrently.
            unsafe {
                match v {
                    Some(v) => std::env::set_var("RSDECIDER_DOWNLOAD_MODELS", v),
                    None => std::env::remove_var("RSDECIDER_DOWNLOAD_MODELS"),
                }
            }
            match Cli::try_parse_from(["rsdecider", "serve"]).unwrap().cmd {
                Cmd::Serve { download_models, .. } => download_models,
                _ => unreachable!(),
            }
        };
        assert!(flag(Some("1")) && flag(Some("true")) && flag(Some("yes")));
        assert!(!flag(Some("0")) && !flag(Some("false")) && !flag(None));
    }
}
