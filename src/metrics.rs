//! Prometheus recorder + `/metrics` router (served on the separate metrics listener).
use axum::{Router, routing::get};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

pub fn install() -> Result<PrometheusHandle, String> {
    PrometheusBuilder::new().install_recorder().map_err(|e| e.to_string())
}

pub fn router(handle: PrometheusHandle) -> Router {
    Router::new().route("/metrics", get(move || std::future::ready(handle.render())))
}
