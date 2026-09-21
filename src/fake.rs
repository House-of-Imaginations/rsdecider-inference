//! Deterministic backend for tests and server-overhead stress runs (no model files needed).
use crate::model::postprocess::Raw;
use crate::scheduler::{Backend, BackendFactory, Batch};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Default)]
pub struct Fake {
    /// Sleep per batch (simulates inference time).
    pub delay: Duration,
    /// Every batch fails with this error.
    pub fail_with: Option<String>,
    /// Panic (once) on a batch containing this token id.
    pub panic_token: Option<u32>,
    pub panicked: Arc<AtomicBool>,
    pub calls: Arc<AtomicUsize>,
    pub items: Arc<AtomicUsize>,
    pub batch_sizes: Arc<Mutex<Vec<usize>>>,
}

impl Fake {
    pub fn factory(&self) -> BackendFactory {
        let f = self.clone();
        Arc::new(move || Ok(Box::new(f.clone()) as Box<dyn Backend>))
    }
}

impl Backend for Fake {
    fn run(&mut self, batch: &Batch) -> Result<Vec<Raw>, String> {
        self.calls.fetch_add(1, SeqCst);
        self.items.fetch_add(batch.items.len(), SeqCst);
        self.batch_sizes.lock().unwrap().push(batch.items.len());
        if let Some(t) = self.panic_token
            && batch.items.iter().any(|e| e.ids.contains(&t))
            && !self.panicked.swap(true, SeqCst)
        {
            panic!("fake backend panic");
        }
        std::thread::sleep(self.delay);
        if let Some(e) = &self.fail_with {
            return Err(e.clone());
        }
        Ok(batch
            .items
            .iter()
            .map(|e| {
                let seed: u32 = e.ids.iter().fold(0u32, |a, &t| a.wrapping_add(t));
                Raw {
                    logits: (0..e.markers.len() as u32)
                        .map(|j| (seed.wrapping_add(31 * j) % 17) as f32 / 4.0)
                        .collect(),
                    act_prob: 0.5,
                    n_tokens: e.ids.len() as u32,
                }
            })
            .collect())
    }
}
