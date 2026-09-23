//! MLX backend for Apple Silicon (`execution_provider = "mlx"`): the Laya forward pass (ModernBERT/mmBERT encoder,
//! 2-layer transformer head, marker scorer, act head), ported layer by layer from the MLX gate model in
//! `tools/export_onnx.py` (`mlx_export_and_gate`), which is checked against PyTorch at export time.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("the mlx feature needs Apple Silicon macOS");

use super::postprocess::Raw;
use super::sequence::{Encoded, QType};
use super::session::{Padded, pad};
use crate::config::MlxDtype;
use crate::scheduler::{Backend, Batch};
use mlx_rs::error::Exception;
use mlx_rs::fast::{layer_norm, rope, scaled_dot_product_attention};
use mlx_rs::ops::indexing::topk_axis;
use mlx_rs::ops::{addmm, concatenate, erf, maximum, select, softmax_axis, stack};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type R<T> = Result<T, Exception>;

/// `mlx.json`, written next to `mlx.safetensors` by `tools/export_onnx.py --mlx`.
#[derive(Deserialize)]
struct Cfg {
    hidden_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: usize,
    global_attn_every_n_layers: usize,
    local_attention: i32,
    rope_theta_global: f32,
    rope_theta_local: f32,
    head_layers: usize,
    head_dim_head: i32,
    mask_value: f32,
}

/// Owned by one worker thread (`Array` is Send, not Sync).
pub struct MlxBackend {
    cfg: Cfg,
    /// Tensors by their `mlx.safetensors` name (`layers.3.Wqkv.weight`, ...), cast to `dt` except the fp32 act head.
    // ponytail: looked up by name on every run (~200 hash lookups per batch, noise next to the GPU work)
    // instead of typed per-layer structs.
    w: HashMap<String, Array>,
    dt: Dtype,
}

impl MlxBackend {
    pub fn new(dir: &Path, dtype: MlxDtype) -> Result<Self, String> {
        let json = dir.join("mlx.json");
        let cfg = std::fs::read_to_string(&json)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Cfg>(&s).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", json.display()))?;
        let st = dir.join("mlx.safetensors");
        let dt = match dtype {
            MlxDtype::Fp16 => Dtype::Float16,
            MlxDtype::Fp32 => Dtype::Float32,
        };
        let w = Array::load_safetensors(&st)
            .map_err(|e| format!("{}: {e}", st.display()))?
            .into_iter()
            .map(|(k, v)| Ok((k.clone(), if k.starts_with("act") { v } else { v.as_dtype(dt)? })))
            .collect::<R<HashMap<_, _>>>()
            .and_then(|w| eval(w.values()).map(|_| w))
            .map_err(|e| format!("{}: {e}", st.display()))?;
        let mut me = Self { cfg, w, dt };
        // A tiny forward at load: a missing or misshapen tensor fails startup instead of the first request.
        let probe = Encoded { ids: vec![0; 4], markers: vec![1, 2], qtype: QType::Choice };
        me.run(&Batch { items: vec![Arc::new(probe)], pad_id: 0 }).map_err(|e| format!("{}: {e}", st.display()))?;
        Ok(me)
    }

    fn w(&self, name: &str) -> R<&Array> {
        self.w.get(name).ok_or_else(|| Exception::custom(format!("missing tensor {name}")))
    }

    /// `nn.Linear`: `x @ W.T (+ b)`.
    fn lin(&self, x: &Array, name: &str) -> R<Array> {
        let w = self.w(&format!("{name}.weight"))?.t();
        match self.w.get(&format!("{name}.bias")) {
            Some(b) => addmm(b, x, &w, None, None),
            None => x.matmul(&w),
        }
    }

    /// `nn.LayerNorm(eps=1e-5)`; the encoder norms carry no bias.
    fn ln(&self, x: &Array, name: &str) -> R<Array> {
        layer_norm(x, self.w(&format!("{name}.weight"))?, self.w.get(&format!("{name}.bias")), 1e-5)
    }

    /// Multi-head attention over a fused `[B,L,3·D]` qkv projection; `theta` applies RoPE to q and k.
    fn attn(&self, qkv: &Array, heads: i32, mask: &Array, theta: Option<f32>) -> R<Array> {
        let (b, l, d) = (qkv.shape()[0], qkv.shape()[1], qkv.shape()[2] / 3);
        let dh = d / heads;
        let qkv = qkv.reshape(&[b, l, 3, heads, dh])?.transpose_axes(&[2, 0, 3, 1, 4])?;
        let [q, k, v]: [Array; 3] = qkv
            .split_equal(3, 0)?
            .iter()
            .map(|a| a.squeeze_axes(&[0]))
            .collect::<R<Vec<_>>>()?
            .try_into()
            .map_err(|_| Exception::custom("qkv split"))?;
        let (q, k) = match theta {
            Some(t) => (rope(&q, dh, false, t, 1.0, 0, None)?, rope(&k, dh, false, t, 1.0, 0, None)?),
            None => (q, k),
        };
        let o = scaled_dot_product_attention(&q, &k, &v, (dh as f32).powf(-0.5), mask, None)?;
        o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, d])
    }

    /// Returns `(logits [B,K], act_prob [B])`, both fp32 and not yet evaluated.
    fn forward(&self, p: &Padded) -> R<(Array, Array)> {
        let c = &self.cfg;
        let (b, l, k) = (p.b as i32, p.l as i32, p.k as i32);
        let int = |v: &[i64], shape: &[i32]| Array::from_slice(&v.iter().map(|&x| x as i32).collect::<Vec<_>>(), shape);
        let ids = int(&p.input_ids, &[b, l]);
        let (am, qtype) = (int(&p.attention_mask, &[b, l]), int(&p.type_ids, &[b]));
        let (mpos, mmask) = (int(&p.marker_pos, &[b, k]), int(&p.marker_mask, &[b, k]));
        let one = Array::from_int(1);
        let zero = Array::from_f32(0.0).as_dtype(self.dt)?;

        // Additive masks built in fp32, then cast: -3e4 stays finite in fp16, so fully padded rows don't go NaN.
        let neg = Array::from_f32(c.mask_value);
        let key = select(am.eq(&one)?.reshape(&[b, 1, 1, l])?, Array::from_f32(0.0), &neg)?;
        let pos = Array::arange::<_, i32>(None, l, None)?;
        let win = pos.reshape(&[l, 1])?.subtract(pos.reshape(&[1, l])?)?.abs()?;
        let win = win.le(Array::from_int(c.local_attention / 2))?;
        let gmask = key.as_dtype(self.dt)?;
        let lmask = select(&win, &key, &neg)?.as_dtype(self.dt)?;

        let mut x = self.ln(&self.w("tok.weight")?.take_axis(&ids, 0)?, "emb_norm")?;
        for i in 0..c.num_hidden_layers {
            let n = format!("layers.{i}");
            let global = i % c.global_attn_every_n_layers == 0;
            let h = if i == 0 { x.clone() } else { self.ln(&x, &format!("{n}.attn_norm"))? };
            let (mask, theta) = if global { (&gmask, c.rope_theta_global) } else { (&lmask, c.rope_theta_local) };
            let o = self.attn(&self.lin(&h, &format!("{n}.Wqkv"))?, c.num_attention_heads, mask, Some(theta))?;
            x = x.add(self.lin(&o, &format!("{n}.Wo"))?)?;
            let ag = self.lin(&self.ln(&x, &format!("{n}.mlp_norm"))?, &format!("{n}.Wi"))?.split_equal(2, -1)?;
            x = x.add(self.lin(&gelu(&ag[0])?.multiply(&ag[1])?, &format!("{n}.Wo2"))?)?;
        }
        x = self.ln(&x, "final_norm")?.add(self.w("type_emb.weight")?.take_axis(&qtype, 0)?.expand_dims(1)?)?;
        for i in 0..c.head_layers {
            let n = format!("head.{i}");
            let qkv = self.lin(&self.ln(&x, &format!("{n}.norm1"))?, &format!("{n}.in_proj"))?;
            x = x.add(
                self.lin(&self.attn(&qkv, c.hidden_size / c.head_dim_head, &gmask, None)?, &format!("{n}.out_proj"))?,
            )?;
            let h = maximum(self.lin(&self.ln(&x, &format!("{n}.norm2"))?, &format!("{n}.linear1"))?, &zero)?;
            x = x.add(self.lin(&h, &format!("{n}.linear2"))?)?;
        }

        // Scorer at the marker positions (padded markers point at token 0 and are masked out).
        let m = x.take_along_axis(mpos.reshape(&[b, k, 1])?, 1)?;
        let s = self.lin(&gelu(&self.lin(&self.ln(&m, "sc_norm")?, "sc1")?)?, "sc2")?;
        let logits = select(mmask.eq(&one)?, s.squeeze_axes(&[-1])?.as_dtype(Dtype::Float32)?, Array::from_f32(-1e4))?;

        // Act head in fp32, like the torch model's `.float()` tail: [CLS, top-1, margin, normalized entropy, k/255].
        let p = softmax_axis(&logits, -1, None)?;
        let kk = maximum(mmask.sum_axis(-1, None)?, Array::from_int(2))?.as_dtype(Dtype::Float32)?;
        let ent = p.multiply(maximum(&p, Array::from_f32(1e-9))?.log()?)?.sum_axis(-1, None)?.negative()?;
        let ent = ent.divide(kk.log()?)?;
        let top2 = topk_axis(&p, 2, -1)?;
        let (t0, t1) = (top2.max_axis(-1, None)?, top2.min_axis(-1, None)?);
        let feats = stack(&[&t0, &t0.subtract(&t1)?, &ent, &kk.divide(Array::from_f32(255.0))?], -1)?;
        let cls = x.take_axis(Array::from_int(0), 1)?.as_dtype(Dtype::Float32)?;
        let act = self.lin(&gelu(&self.lin(&concatenate(&[&cls, &feats], -1)?, "act1")?)?, "act2")?;
        Ok((logits, softmax_axis(&act, -1, None)?.take_axis(Array::from_int(0), 1)?))
    }
}

/// `nn.gelu` (erf form) with its constants in `x`'s dtype: mlx-rs's `nn::gelu` uses f32 constants, which would
/// silently promote an fp16 graph to fp32.
fn gelu(x: &Array) -> R<Array> {
    let c = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
    x.multiply(erf(x.divide(c(std::f32::consts::SQRT_2)?)?)?.add(c(1.0)?)?)?.divide(c(2.0)?)
}

impl Backend for MlxBackend {
    fn run(&mut self, batch: &Batch) -> Result<Vec<Raw>, String> {
        let p = pad(batch);
        let (logits, act) = self.forward(&p).map_err(|e| e.to_string())?;
        eval([&logits, &act]).map_err(|e| e.to_string())?;
        let logits = logits.try_as_slice::<f32>().map_err(|e| e.to_string())?;
        let act = act.try_as_slice::<f32>().map_err(|e| e.to_string())?;
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
