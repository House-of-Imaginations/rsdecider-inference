"""Export a Laya checkpoint to ONNX + fixtures for rsdecider.

Usage:
  python tools/export_onnx.py --out models/english [--quantize int8]          # repo root = English
  python tools/export_onnx.py --subfolder multilingual --out models/multilingual

Writes <out>/model.onnx, tokenizer.json, laya.json, fixtures.json and self-checks ORT vs PyTorch.
"""
import argparse
import json
import os
import types

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoModel

import laya
from laya.common import QTYPES, DecisionModel, build_sequence, render_options
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
    return [
        {"state": long_state, "questions": {"q": {"type": "noul", "instructions": "Is this a billing issue?"}}},
        {"state": "Pick the right team.", "questions": {"many": {"type": "choice", "instructions": "Route", "criteria": many}}},
        {"state": "x", "questions": {"overflow": {"type": "choice", "instructions": "Route", "criteria": too_many}}},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None, help="checkpoint subfolder, e.g. multilingual; omit for the repo root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", choices=["int8"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
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
    if args.quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        fp32 = onnx_path + ".fp32"
        os.replace(onnx_path, fp32)
        quantize_dynamic(fp32, onnx_path, weight_type=QuantType.QInt8)
        os.remove(fp32)

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
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    fixtures, worst = [], 0.0
    n_q = agree = 0
    max_dp = 0.0
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
            case["raw"][qid] = {"logits": logits[0, :k].tolist(), "act_prob": float(act[0]), "n_tokens": len(enc["ids"])}
            items.append((qid, enc))
        if not case["errors"]:
            case["answers"] = agent.system_one(req["state"], req["questions"])["answers"]
        if items:  # self-check: ORT on the padded mixed batch vs PyTorch single runs
            out_logits, out_act = sess.run(None, collate([e for _, e in items], tok.pad_token_id))
            for i, (qid, enc) in enumerate(items):
                k = len(enc["markers"])
                ref = np.array(case["raw"][qid]["logits"])
                worst = max(worst, float(np.abs(out_logits[i, :k] - ref).max()))
                p_ort, p_ref = softmax(out_logits[i, :k]), softmax(ref)
                n_q += 1
                agree += int(np.argmax(p_ort) == np.argmax(p_ref))
                max_dp = max(max_dp, float(np.abs(p_ort - p_ref).max()))
        fixtures.append(case)
    with open(os.path.join(args.out, "fixtures.json"), "w") as f:
        json.dump(fixtures, f, ensure_ascii=False)
    print(f"wrote {len(fixtures)} fixtures; worst ORT-vs-PyTorch logit diff {worst:.2e}")
    print(f"top-1 agreement {agree}/{n_q}; max |Δp| {max_dp:.4f}")
    if args.quantize is None and worst > 1e-2:
        raise SystemExit("parity self-check failed: ORT output differs from PyTorch")
    if args.quantize == "int8" and (agree < n_q or max_dp > 0.05):
        raise SystemExit("int8 decision gate failed: top-1 must match on every question and |Δp| <= 0.05")


if __name__ == "__main__":
    main()
