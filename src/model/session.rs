//! ORT backend: one `Session` per worker thread (`Session::run` takes `&mut self`).
use super::postprocess::Raw;
use crate::scheduler::{Backend, Batch};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::Path;

/// Padded int64 inputs for one batch, row-major.
#[derive(Debug, PartialEq)]
pub struct Padded {
    pub b: usize,
    pub l: usize,
    pub k: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub type_ids: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub marker_mask: Vec<i64>,
}

/// Same layout as Laya `collate_items`: pad ids with pad_id, zero attention/markers past each row.
pub fn pad(batch: &Batch) -> Padded {
    let b = batch.items.len();
    let l = batch.items.iter().map(|e| e.ids.len()).max().unwrap_or(0);
    let k = batch.items.iter().map(|e| e.markers.len()).max().unwrap_or(0);
    let mut p = Padded {
        b,
        l,
        k,
        input_ids: vec![batch.pad_id as i64; b * l],
        attention_mask: vec![0; b * l],
        type_ids: Vec::with_capacity(b),
        marker_pos: vec![0; b * k],
        marker_mask: vec![0; b * k],
    };
    for (i, e) in batch.items.iter().enumerate() {
        for (j, &t) in e.ids.iter().enumerate() {
            p.input_ids[i * l + j] = t as i64;
            p.attention_mask[i * l + j] = 1;
        }
        for (j, &m) in e.markers.iter().enumerate() {
            p.marker_pos[i * k + j] = m as i64;
            p.marker_mask[i * k + j] = 1;
        }
        p.type_ids.push(e.qtype.id() as i64);
    }
    p
}

pub struct OrtBackend {
    session: Session,
}

impl OrtBackend {
    pub fn new(model_onnx: &Path, execution_provider: &str, intra_threads: usize) -> Result<Self, String> {
        let ep = match execution_provider {
            "cpu" => ort::ep::CPU::default().build(),
            #[cfg(feature = "cuda")]
            "cuda" => ort::ep::CUDA::default().with_device_id(0).build(),
            other => return Err(format!("execution provider {other:?} is not available in this build")),
        };
        let build = || -> ort::Result<Session> {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(intra_threads)?
                // Two sessions plus tokio would otherwise spin-steal the fast cores between batches.
                .with_intra_op_spinning(false)?
                .with_execution_providers([ep])?
                .commit_from_file(model_onnx)
        };
        let session = build().map_err(|e| format!("{}: {e}", model_onnx.display()))?;
        Ok(Self { session })
    }
}

impl Backend for OrtBackend {
    fn run(&mut self, batch: &Batch) -> Result<Vec<Raw>, String> {
        let p = pad(batch);
        let t = |shape: Vec<usize>, data: Vec<i64>| Tensor::from_array((shape, data)).map_err(|e| e.to_string());
        let outs = self
            .session
            .run(ort::inputs![
                "input_ids" => t(vec![p.b, p.l], p.input_ids)?,
                "attention_mask" => t(vec![p.b, p.l], p.attention_mask)?,
                "type_ids" => t(vec![p.b], p.type_ids)?,
                "marker_pos" => t(vec![p.b, p.k], p.marker_pos)?,
                "marker_mask" => t(vec![p.b, p.k], p.marker_mask)?,
            ])
            .map_err(|e| e.to_string())?;
        let (_, logits) = outs["logits"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        let (_, act) = outs["act_prob"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok(batch
            .items
            .iter()
            .enumerate()
            .map(|(i, e)| Raw {
                logits: logits[i * p.k..i * p.k + e.markers.len()].to_vec(),
                act_prob: act[i],
                n_tokens: e.ids.len() as u32,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::sequence::{Encoded, QType};
    use std::sync::Arc;

    #[test]
    fn pads_like_collate_items() {
        let batch = Batch {
            pad_id: 9,
            items: vec![
                Arc::new(Encoded { ids: vec![1, 5, 2], markers: vec![1], qtype: QType::Noul }),
                Arc::new(Encoded { ids: vec![1, 6, 7, 8, 2], markers: vec![1, 2], qtype: QType::Choice }),
            ],
        };
        let p = pad(&batch);
        assert_eq!((p.b, p.l, p.k), (2, 5, 2));
        assert_eq!(p.input_ids, vec![1, 5, 2, 9, 9, 1, 6, 7, 8, 2]);
        assert_eq!(p.attention_mask, vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(p.type_ids, vec![2, 0]);
        assert_eq!(p.marker_pos, vec![1, 0, 1, 2]);
        assert_eq!(p.marker_mask, vec![1, 0, 1, 1]);
    }
}
