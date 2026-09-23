"""Export a Laya checkpoint to ONNX + fixtures for rsdecider.

Usage:
  python tools/export_onnx.py --out models/english [--quantize int8|w8 [--force]]  # repo root = English
  python tools/export_onnx.py --subfolder multilingual --out models/multilingual

Writes <out>/model.onnx, tokenizer.json, laya.json, fixtures.json and self-checks ORT vs PyTorch.
"""
import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoModel

import laya
from laya.common import QTYPES, DecisionModel, build_sequence, render_options, temp_bucket
from laya.lang import is_english


class Wrapper(torch.nn.Module):
    """DecisionModel.forward + the act softmax from Agent.system_one. All inputs int64."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, type_ids, marker_pos, marker_mask):
        logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask.bool(), type_ids)
        return logits, torch.softmax(act.float(), -1)[:, 0]


def mha_forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):
    """Self-attention for the head layers. The TorchScript trace of nn.MultiheadAttention bakes the
    trace-time sequence length into q/k.view(tgt_len, ...), so the exported graph fails on any other L.
    key_padding_mask arrives as the additive float mask TransformerEncoderLayer builds."""
    E, H = self.embed_dim, self.num_heads
    qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias).unflatten(-1, (3, H, E // H)).permute(2, 0, 3, 1, 4)
    mask = None if key_padding_mask is None else key_padding_mask[:, None, None, :]
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
    return self.out_proj(o.transpose(1, 2).flatten(2)), None


def eager_model(agent):
    """Rebuild with eager attention (SDPA/flash paths do not export cleanly) and copy the weights."""
    enc = AutoModel.from_config(agent.model.encoder.config, attn_implementation="eager")
    n_act = agent.model.act_head[-1].out_features
    head_layers = len(agent.model.head.layers) if agent.model.head is not None else 0
    m = DecisionModel(enc, head_layers, n_act)
    m.load_state_dict(agent.model.state_dict())
    for layer in m.head.layers if m.head is not None else []:
        layer.self_attn.forward = types.MethodType(mha_forward, layer.self_attn)
    return m.eval()


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def temperature(t_by_type, t_by_options, qtype, k):
    """Mirror src/model/postprocess.rs answer(): the option-count bucket, else the type's temperature."""
    return max(t_by_options.get(temp_bucket(qtype, k), t_by_type[qtype]), 1e-3)


def update_gate(stats, p_cmp, p_ref, act_cmp, act_ref):
    """Near-tie-aware decision gate, shared by the quantized-ONNX and MLX gates. Decisive = single option
    (trivially agrees), or reference top-2 margin > 0.10. With |Δp| <= 0.05 per option, only a question whose
    margin <= 0.10 can flip top-1, so a flip on a near-tie is a tie broken differently, not accuracy damage;
    only decisive questions gate the export."""
    top1_cmp, top1_ref = int(np.argmax(p_cmp)), int(np.argmax(p_ref))
    if len(p_ref) == 1:
        decisive = True
    else:
        p_ref_sorted = np.sort(p_ref)[::-1]
        decisive = float(p_ref_sorted[0] - p_ref_sorted[1]) > 0.10
        if not decisive:
            stats["near_ties"] += 1
            stats["near_ties_flipped"] += int(top1_cmp != top1_ref)
    if decisive:
        stats["n_decisive"] += 1
        stats["agree_decisive"] += int(top1_cmp == top1_ref)
    stats["max_dp"] = max(stats["max_dp"], float(np.abs(p_cmp - p_ref).max()))
    stats["max_dact"] = max(stats["max_dact"], abs(act_cmp - act_ref))


def gate_failed(stats):
    return stats["agree_decisive"] < stats["n_decisive"] or stats["max_dp"] > 0.05 or stats["max_dact"] > 0.05


def gate_line(stats):
    return (f"top-1 agreement on decisive questions {stats['agree_decisive']}/{stats['n_decisive']}; "
            f"near-ties {stats['near_ties']} ({stats['near_ties_flipped']} flipped); "
            f"max |Δp| {stats['max_dp']:.4f} (temperature-scaled); max |Δ act_prob| {stats['max_dact']:.4f}")


def encode(agent, state, qdef):
    q = agent._to_internal(qdef)
    cfg = agent.cfg
    ids, markers = build_sequence(agent.tok, state, q, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    if len(markers) != len(render_options(q)):
        return None
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    K = max(len(it["markers"]) for it in items)
    a = {k: np.zeros(s, dtype=np.int64) for k, s in
         [("input_ids", (n, L)), ("attention_mask", (n, L)), ("marker_pos", (n, K)), ("marker_mask", (n, K))]}
    a["input_ids"][:] = pad_id
    for i, it in enumerate(items):
        a["input_ids"][i, :len(it["ids"])] = it["ids"]
        a["attention_mask"][i, :len(it["ids"])] = 1
        a["marker_pos"][i, :len(it["markers"])] = it["markers"]
        a["marker_mask"][i, :len(it["markers"])] = 1
    a["type_ids"] = np.array([it["qtype"] for it in items], dtype=np.int64)
    return a


def generated_cases():
    long_state = {"ticket": " ".join(["The customer reports the invoice total is wrong again."] * 120)}
    many = {f"option_{i}": f"route to team {i}" for i in range(40)}
    too_many = {f"o{i}": "" for i in range(255)}
    long_zh = "客户反映发票金额有误，要求核实并处理退款申请。" * 30
    long_ja = "お客様から請求金額に誤りがあるとのご連絡をいただき、返金についてご案内しております。" * 20
    return [
        {"state": long_state, "questions": {"q": {"type": "noul", "instructions": "Is this a billing issue?"}}},
        {"state": "Pick the right team.", "questions": {"many": {"type": "choice", "instructions": "Route", "criteria": many}}},
        {"state": "x", "questions": {"overflow": {"type": "choice", "instructions": "Route", "criteria": too_many}}},
        {"state": "Hi", "questions": {"greet": {"type": "noul", "instructions": "Is this a greeting?"}}},
        {"state": "Cancel my subscription immediately.", "questions": {
            "urgency": {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "medium", "high"]},
            "action": {"type": "choice", "instructions": "What should support do?",
                       "criteria": {"cancel": "cancel the plan", "retain": "offer a discount"}}}},
        {"state": "Multiple production databases are corrupted and customer data may be lost.", "questions": {
            "sev": {"type": "score", "instructions": "Severity", "criteria": ["ok", "warning", "critical"]}}},
        {"state": "The server room temperature is 42C and rising.", "questions": {
            "sev": {"type": "score", "instructions": "Severity", "criteria": ["ok", "warning", "critical"]}}},
        {"state": "Thanks, that solved it!", "questions": {"resolved": {"type": "noul", "instructions": "Is the issue resolved?"}}},
        {"state": {"ticket_id": 1002, "priority": "high", "tags": ["billing", "refund"]},
         "questions": {"tag": {"type": "choice", "instructions": "Pick the primary tag",
                                "criteria": {"billing": "money related", "other": "everything else"}}}},
        {"state": "quick yes or no", "questions": {
            "confirm": {"type": "choice", "instructions": "Confirm", "criteria": {"yes": "confirmed", "no": "not confirmed"}}}},
        {"state": "binary severity call", "questions": {
            "solo": {"type": "score", "instructions": "Rate", "criteria": ["fine", "not fine"]}}},
        {"state": [
            {"role": "user", "content": "I want a refund"},
            {"role": "agent", "content": "Can you provide the order number?"},
            {"role": "user", "content": "12345"},
        ], "questions": {"has_order": {"type": "noul", "instructions": "Did the user provide an order number?"}}},
        {"state": "😀 great job team, ticket closed 🎉", "questions": {
            "positive": {"type": "noul", "instructions": "Is the sentiment positive?"}}},
        {"state": "客户反映账单金额有误，要求核实并退款。", "questions": {
            "action": {"type": "choice", "instructions": "决定如何处理", "criteria": {
                "approve": "退款合理", "deny": "退款不合理", "escalate": "需要人工审核"}}}},
        {"state": long_zh, "questions": {
            "sev": {"type": "score", "instructions": "评估严重程度", "criteria": ["轻微", "中等", "严重"]}}},
        {"state": "お問い合わせありがとうございます。返金についてご案内します。", "questions": {
            "urgent": {"type": "noul", "instructions": "これは緊急ですか?"}}},
        {"state": long_ja, "questions": {
            "topic": {"type": "choice", "instructions": "トピックを選択", "criteria": ["請求", "配送", "アカウント"]}}},
        {"state": "고객이 두 번 결제되었다고 문의했습니다.", "questions": {
            "action": {"type": "choice", "instructions": "처리 방법을 결정하세요", "criteria": {
                "approve": "환불이 타당함", "deny": "환불이 타당하지 않음"}}}},
        {"state": "客户遇到登录问题，多次尝试后仍然失败。", "questions": {
            "urgent": {"type": "noul", "instructions": "是否紧急?"},
            "sev": {"type": "score", "instructions": "严重程度", "criteria": ["低", "中", "高"]}}},
        {"state": "Order #12345 status?", "questions": {
            "valid": {"type": "noul", "instructions": "Is this a valid ticket id?"}}},
        {"state": {"amount": None, "note": ""}, "questions": {
            "action": {"type": "choice", "instructions": "Decide", "criteria": {"escalate": "", "ignore": None}}}},
    ]


def resolve_checkpoint_dir(repo, subfolder):
    """The local HF snapshot dir laya.load() reads from (mirrors laya.agent.Agent.__init__)."""
    model_dir = repo
    if not os.path.exists(model_dir):
        from huggingface_hub import snapshot_download
        kw = {"token": os.environ.get("HF_TOKEN")}
        if subfolder:
            kw["allow_patterns"] = [f"{subfolder}/*"]
        model_dir = snapshot_download(repo, **kw)
    return os.path.join(model_dir, subfolder) if subfolder else model_dir


def mlx_export_and_gate(agent, checkpoint_dir, out, fixtures, force):
    """Write mlx.safetensors + mlx.json (Rust-layout weights, fp16 except the fp32 act head) and gate them
    against the PyTorch logits/act_prob already computed for `fixtures`. On a failing gate, no mlx.* files
    are kept unless `force`."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        raise SystemExit("--mlx needs Apple Silicon and `pip install mlx`")

    def ln(d, bias):
        return nn.LayerNorm(d, eps=1e-5, bias=bias)

    class EncLayer(nn.Module):
        def __init__(self, i, mc):
            super().__init__()
            D, self.H, I = mc["hidden_size"], mc["num_attention_heads"], mc["intermediate_size"]
            self.is_global = i % mc["global_attn_every_n_layers"] == 0
            self.attn_norm = ln(D, False) if i else None
            self.Wqkv = nn.Linear(D, 3 * D, bias=False)
            self.Wo = nn.Linear(D, D, bias=False)
            theta = mc["rope_theta_global"] if self.is_global else mc["rope_theta_local"]
            self.rope = nn.RoPE(D // self.H, base=theta)
            self.mlp_norm = ln(D, False)
            self.Wi = nn.Linear(D, 2 * I, bias=False)
            self.Wo2 = nn.Linear(I, D, bias=False)

        def __call__(self, x, gmask, lmask):
            B, L, D = x.shape
            h = self.attn_norm(x) if self.attn_norm is not None else x
            q, k, v = self.Wqkv(h).reshape(B, L, 3, self.H, D // self.H).transpose(2, 0, 3, 1, 4)
            o = mx.fast.scaled_dot_product_attention(self.rope(q), self.rope(k), v, scale=(D // self.H) ** -0.5,
                                                       mask=gmask if self.is_global else lmask)
            x = x + self.Wo(o.transpose(0, 2, 1, 3).reshape(B, L, D))
            a, g = mx.split(self.Wi(self.mlp_norm(x)), 2, axis=-1)
            return x + self.Wo2(nn.gelu(a) * g)

    class HeadLayer(nn.Module):  # nn.TransformerEncoderLayer(norm_first=True, activation=relu)
        def __init__(self, D, hd):
            super().__init__()
            self.hd = hd
            self.norm1, self.norm2 = ln(D, True), ln(D, True)
            self.in_proj = nn.Linear(D, 3 * D)
            self.out_proj = nn.Linear(D, D)
            self.linear1, self.linear2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

        def __call__(self, x, kmask):
            B, L, D = x.shape
            hh = D // self.hd
            q, k, v = self.in_proj(self.norm1(x)).reshape(B, L, 3, hh, self.hd).transpose(2, 0, 3, 1, 4)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=kmask)
            x = x + self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))
            return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))

    class Laya(nn.Module):
        def __init__(self, mc):
            super().__init__()
            D = mc["hidden_size"]
            self.mask_value, self.local_attention = mc["mask_value"], mc["local_attention"]
            self.tok = nn.Embedding(mc["vocab_size"], D)
            self.emb_norm = ln(D, False)
            self.layers = [EncLayer(i, mc) for i in range(mc["num_hidden_layers"])]
            self.final_norm = ln(D, False)
            self.type_emb = nn.Embedding(3, D)
            self.head = [HeadLayer(D, mc["head_dim_head"]) for _ in range(mc["head_layers"])]
            self.sc_norm = ln(D, True)
            self.sc1, self.sc2 = nn.Linear(D, D), nn.Linear(D, 1)
            self.act1, self.act2 = nn.Linear(D + 4, 256), nn.Linear(256, 2)

        def __call__(self, ids, am, qtype, mpos, mmask):
            B, L = ids.shape
            dt = self.tok.weight.dtype
            key = mx.where(am[:, None, None, :] == 1, 0.0, self.mask_value)
            pos = mx.arange(L)
            win = mx.abs(pos[:, None] - pos[None, :]) <= self.local_attention // 2
            gmask = key.astype(dt)
            lmask = mx.where(win[None, None], key, self.mask_value).astype(dt)
            x = self.emb_norm(self.tok(ids))
            for layer in self.layers:
                x = layer(x, gmask, lmask)
            x = self.final_norm(x) + self.type_emb(qtype)[:, None, :]
            for layer in self.head:
                x = layer(x, gmask)
            m = mx.take_along_axis(x, mx.maximum(mpos, 0)[:, :, None], axis=1)
            logits = self.sc2(nn.gelu(self.sc1(self.sc_norm(m)))).squeeze(-1).astype(mx.float32)
            logits = mx.where(mmask == 1, logits, -1e4)
            p = mx.softmax(logits, -1)
            kk = mx.maximum(mmask.sum(-1), 2).astype(mx.float32)
            ent = -(p * mx.log(mx.maximum(p, 1e-9))).sum(-1) / mx.log(kk)
            top2 = mx.topk(p, 2, axis=-1)  # ascending order in mlx
            t0, t1 = top2.max(-1), top2.min(-1)
            feats = mx.stack([t0, t0 - t1, ent, kk / 255.0], -1)
            act = self.act2(nn.gelu(self.act1(mx.concatenate([x[:, 0].astype(mx.float32), feats], -1))))
            return logits, mx.softmax(act, -1)[:, 0]

    # rename map from the reference port (.superpowers/sdd/2026-09-22-mlx-backend/ref/mlx_laya.py)
    ren = {"encoder.embeddings.tok_embeddings": "tok", "encoder.embeddings.norm": "emb_norm",
           "encoder.final_norm": "final_norm", "scorer.0": "sc_norm", "scorer.1": "sc1", "scorer.3": "sc2",
           "act_head.0": "act1", "act_head.2": "act2", "type_emb": "type_emb"}
    raw = mx.load(os.path.join(checkpoint_dir, "model.safetensors"))
    weights = {}
    for k, v in raw.items():
        if k == "temperature":
            continue
        p = k
        for a, b in ren.items():
            if k.startswith(a + "."):
                p = b + k[len(a):]
        p = (p.replace("encoder.layers.", "layers.").replace("head.layers.", "head.")
               .replace(".attn.Wqkv", ".Wqkv").replace(".attn.Wo", ".Wo")
               .replace(".mlp.Wi", ".Wi").replace(".mlp.Wo", ".Wo2")
               .replace(".self_attn.in_proj_weight", ".in_proj.weight")
               .replace(".self_attn.in_proj_bias", ".in_proj.bias")
               .replace(".self_attn.out_proj", ".out_proj"))
        weights[p] = v.astype(mx.float32 if p.startswith(("act1.", "act2.")) else mx.float16)

    with open(os.path.join(checkpoint_dir, "encoder", "config.json")) as f:
        ec = json.load(f)
    rp = ec["rope_parameters"]
    mc = {"hidden_size": ec["hidden_size"], "num_attention_heads": ec["num_attention_heads"],
          "num_hidden_layers": ec["num_hidden_layers"], "intermediate_size": ec["intermediate_size"],
          "global_attn_every_n_layers": ec["global_attn_every_n_layers"], "local_attention": ec["local_attention"],
          "rope_theta_global": rp["full_attention"]["rope_theta"],
          "rope_theta_local": rp["sliding_attention"]["rope_theta"], "vocab_size": ec["vocab_size"],
          "head_layers": 2, "head_dim_head": 64, "mask_value": -30000.0}

    # mx.load() infers the format from the extension, so the weights tmp file must still end in .safetensors
    st_tmp, json_tmp = os.path.join(out, "mlx.tmp.safetensors"), os.path.join(out, "mlx.json.tmp")
    st_path, json_path = os.path.join(out, "mlx.safetensors"), os.path.join(out, "mlx.json")
    try:
        mx.save_safetensors(st_tmp, weights)
        with open(json_tmp, "w") as f:
            json.dump(mc, f, indent=2)

        m = Laya(mc)
        m.load_weights(st_tmp, strict=True)
        mx.eval(m.parameters())

        stats = dict(n_decisive=0, agree_decisive=0, near_ties=0, near_ties_flipped=0, max_dp=0.0, max_dact=0.0)
        for case in fixtures:
            for qid, enc in case["encoded"].items():
                qtype, k = QTYPES[case["questions"][qid]["type"]], len(enc["markers"])
                ids = mx.array(np.array([enc["ids"]], dtype=np.int32))
                am = mx.array(np.ones((1, len(enc["ids"])), dtype=np.int32))
                mpos = mx.array(np.array([enc["markers"]], dtype=np.int32))
                mmask = mx.array(np.ones((1, k), dtype=np.int32))
                logits, act = m(ids, am, mx.array(np.array([qtype], dtype=np.int32)), mpos, mmask)
                got, ref = np.array(logits)[0, :k].astype(np.float64), np.array(case["raw"][qid]["logits"])
                t = temperature(agent.temperature, agent.temperature_by_options, qtype, k)
                update_gate(stats, softmax(got / t), softmax(ref / t), float(np.array(act)[0]),
                            case["raw"][qid]["act_prob"])
        print("mlx " + gate_line(stats))

        if gate_failed(stats) and not force:
            raise SystemExit("mlx decision gate failed: top-1 must match on every decisive question (reference "
                              "top-2 margin > 0.10), |Δp| <= 0.05 and |Δ act_prob| <= 0.05 on every question; "
                              "no mlx.safetensors/mlx.json written (rerun with --force to override)")
        if gate_failed(stats):
            print("WARNING: mlx decision gate failed (see numbers above) but --force was given; writing "
                  "mlx.safetensors/mlx.json anyway")
        os.replace(st_tmp, st_path)
        os.replace(json_tmp, json_path)
    finally:
        for p in (st_tmp, json_tmp):  # leftover only when the gate failed without --force, or on any error
            if os.path.exists(p):
                os.remove(p)


def write_manifest(out):
    """manifest.json: size + sha256 of every served file, so rsdecider can verify (and download) the folder.
    mlx.safetensors/mlx.json are included when a --mlx export wrote them."""
    files = {}
    names = ["model.onnx", "tokenizer.json", "laya.json", "fixtures.json"]
    names += [n for n in ("mlx.safetensors", "mlx.json") if os.path.exists(os.path.join(out, n))]
    for name in names:
        h = hashlib.sha256()
        with open(os.path.join(out, name), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        files[name] = {"size": os.path.getsize(os.path.join(out, name)), "sha256": h.hexdigest()}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"files": files}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None, help="checkpoint subfolder, e.g. multilingual; omit for the repo root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", choices=["int8", "w8"])
    ap.add_argument("--mlx", action="store_true",
                     help="also write mlx.safetensors + mlx.json, gated against PyTorch on the fixtures")
    ap.add_argument("--force", action="store_true",
                     help="with --quantize/--mlx, keep the quantized model.onnx / mlx.* files even if the "
                          "decision gate fails (prints the gate numbers and a warning first)")
    args = ap.parse_args()
    if args.force and args.quantize is None and not args.mlx:
        ap.error("--force only makes sense with --quantize or --mlx")
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest):  # never leave an old manifest beside a half-written new export
        os.remove(manifest)
    torch.backends.mha.set_fastpath_enabled(False)  # fused MHA kernels are not exportable

    agent = laya.load(args.repo, device="cpu", subfolder=args.subfolder)
    model = eager_model(agent)
    wrapper = Wrapper(model).eval()
    tok = agent.tok

    sample = collate([encode(agent, "hello", {"type": "noul", "instructions": "ok?"})] * 2, tok.pad_token_id)
    names = ["input_ids", "attention_mask", "type_ids", "marker_pos", "marker_mask"]
    onnx_path = os.path.join(args.out, "model.onnx")
    torch.onnx.export(
        wrapper, tuple(torch.from_numpy(sample[n]) for n in names), onnx_path,
        input_names=names, output_names=["logits", "act_prob"], opset_version=17, dynamo=False,
        dynamic_axes={"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "type_ids": {0: "B"},
                      "marker_pos": {0: "B", 1: "K"}, "marker_mask": {0: "B", 1: "K"},
                      "logits": {0: "B", 1: "K"}, "act_prob": {0: "B"}},
    )
    check_path = onnx_path
    try:  # check_path is set before quantizing starts, so the finally below removes a partial file too
        if args.quantize == "int8":  # quantize beside the fp32; it replaces model.onnx only if the gate passes
            from onnxruntime.quantization import QuantType, quantize_dynamic
            check_path = os.path.join(args.out, "model.int8.onnx")
            quantize_dynamic(onnx_path, check_path, weight_type=QuantType.QInt8)
        elif args.quantize == "w8":  # 8-bit weight-only (MatMulNBits); quantize beside the fp32, same gate as int8
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            check_path = os.path.join(args.out, "model.w8.onnx")
            quantizer = MatMulNBitsQuantizer(onnx_path, bits=8, block_size=128, is_symmetric=True, accuracy_level=4)
            quantizer.process()
            quantizer.model.save_model_to_file(check_path, use_external_data_format=False)
        tok.backend_tokenizer.save(os.path.join(args.out, "tokenizer.json"))
        cfg = agent.cfg
        with open(os.path.join(args.out, "laya.json"), "w") as f:
            json.dump({
                "max_len": cfg.get("max_len", 512), "head_max_len": cfg.get("head_max_len", 192),
                "mask_token": tok.mask_token, "mask_id": tok.mask_token_id, "cls_id": tok.cls_token_id,
                "sep_id": tok.sep_token_id, "pad_id": tok.pad_token_id,
                "temperature": list(agent.temperature), "temperature_by_options": dict(agent.temperature_by_options),
            }, f, indent=2)

        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "fixture_requests.json")) as f:
            requests = json.load(f) + generated_cases()
        sess = ort.InferenceSession(check_path, providers=["CPUExecutionProvider"])
        fixtures, worst = [], 0.0
        stats = dict(n_decisive=0, agree_decisive=0, near_ties=0, near_ties_flipped=0, max_dp=0.0, max_dact=0.0)
        for req in requests:
            case = {"state": req["state"], "questions": req["questions"], "is_english": is_english(req["state"]),
                    "encoded": {}, "raw": {}, "errors": []}
            items = []
            for qid, qdef in req["questions"].items():
                enc = encode(agent, req["state"], qdef)
                if enc is None:
                    case["errors"].append(qid)
                    continue
                with torch.no_grad():
                    t = {k: torch.from_numpy(v) for k, v in collate([enc], tok.pad_token_id).items()}
                    logits, act = wrapper(*(t[n] for n in names))
                k = len(enc["markers"])
                case["encoded"][qid] = {"ids": enc["ids"], "markers": enc["markers"]}
                case["raw"][qid] = {"logits": logits[0, :k].tolist(), "act_prob": float(act[0]),
                                     "n_tokens": len(enc["ids"])}
                items.append((qid, enc))
            if not case["errors"]:
                case["answers"] = agent.system_one(req["state"], req["questions"])["answers"]
            if items:  # self-check: ORT on the padded mixed batch vs PyTorch single runs
                out_logits, out_act = sess.run(None, collate([e for _, e in items], tok.pad_token_id))
                for i, (qid, enc) in enumerate(items):
                    k = len(enc["markers"])
                    ref = np.array(case["raw"][qid]["logits"])
                    worst = max(worst, float(np.abs(out_logits[i, :k] - ref).max()))
                    t = temperature(agent.temperature, agent.temperature_by_options, enc["qtype"], k)
                    p_ort, p_ref = softmax(out_logits[i, :k] / t), softmax(ref / t)
                    update_gate(stats, p_ort, p_ref, float(out_act[i]), case["raw"][qid]["act_prob"])
            fixtures.append(case)
        with open(os.path.join(args.out, "fixtures.json"), "w") as f:
            json.dump(fixtures, f, ensure_ascii=False)
        print(f"wrote {len(fixtures)} fixtures; worst ORT-vs-PyTorch logit diff {worst:.2e}")
        print(gate_line(stats))
        if args.quantize is None and worst > 1e-2:
            raise SystemExit("parity self-check failed: ORT output differs from PyTorch")
        if args.quantize in ("int8", "w8"):
            del sess  # release the file before moving or deleting it
            if gate_failed(stats) and not args.force:
                raise SystemExit(f"{args.quantize} decision gate failed: top-1 must match on every decisive "
                                  "question (reference top-2 margin > 0.10), |Δp| <= 0.05 and |Δ act_prob| <= 0.05 "
                                  "on every question; kept the fp32 model.onnx (rerun with --force to override)")
            if gate_failed(stats):
                print(f"WARNING: {args.quantize} decision gate failed (see numbers above) but --force was given; "
                      "writing the quantized model.onnx anyway")
            os.replace(check_path, onnx_path)
        if args.mlx:
            checkpoint_dir = resolve_checkpoint_dir(args.repo, args.subfolder)
            mlx_export_and_gate(agent, checkpoint_dir, args.out, fixtures, args.force)
    finally:
        # Any exception or interrupt before the gate (or a gate failure) leaves the quantized temp file
        # behind unless we clean it up here; a successful gate already moved it onto onnx_path.
        if check_path != onnx_path and os.path.exists(check_path):
            os.remove(check_path)
    write_manifest(args.out)  # last: only a folder whose model.onnx is final gets a manifest
    if args.mlx:
        # ponytail: mlx's Metal/libc++ teardown can SIGABRT during interpreter exit even after a fully
        # correct export (this manifest included), turning a real success into a misleading non-zero exit.
        # Skip teardown on this success path only — failures still fall through to the normal exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
