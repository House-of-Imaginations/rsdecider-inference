//! MLX backend for Apple Silicon (`execution_provider = "mlx"`). The forward pass lands in Task 3;
//! this stub only validates the model folder so startup wiring (Task 2) can be tested end to end.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("the mlx feature needs Apple Silicon macOS");

use super::postprocess::Raw;
use crate::config::MlxDtype;
use crate::scheduler::{Backend, Batch};
use std::path::Path;

pub struct MlxBackend {
    #[allow(dead_code)] // read by the Task 3 forward pass
    dtype: MlxDtype,
}

impl MlxBackend {
    pub fn new(dir: &Path, dtype: MlxDtype) -> Result<Self, String> {
        for f in ["mlx.safetensors", "mlx.json"] {
            let p = dir.join(f);
            if !p.is_file() {
                return Err(format!("{}: not found", p.display()));
            }
        }
        Ok(Self { dtype })
    }
}

impl Backend for MlxBackend {
    /// Stub: the real forward pass (Task 3) replaces this body.
    fn run(&mut self, _batch: &Batch) -> Result<Vec<Raw>, String> {
        Err("not implemented".into())
    }
}
