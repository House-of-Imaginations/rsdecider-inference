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


def write_manifest(out):
    """manifest.json: size + sha256 of every served file, so rsdecider can verify (and download) the folder."""
    files = {}
    for name in ("model.onnx", "tokenizer.json", "laya.json", "fixtures.json"):
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
    ap.add_argument("--force", action="store_true",
                     help="with --quantize, replace model.onnx with the quantized model even if the decision "
                          "gate fails (prints the gate numbers and a warning first)")
    args = ap.parse_args()
    if args.force and args.quantize is None:
        ap.error("--force only makes sense with --quantize")
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
        n_decisive = agree_decisive = near_ties = near_ties_flipped = 0
        max_dp = max_dact = 0.0
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
                    top1_ort, top1_ref = int(np.argmax(p_ort)), int(np.argmax(p_ref))
                    # Decisive = single option (trivially agrees), or reference top-2 margin > 0.10. With
                    # |Δp| <= 0.05 per option, only a question whose margin <= 0.10 can flip top-1, so a flip
                    # on a near-tie is a tie broken differently, not accuracy damage; only decisive questions
                    # gate the export.
                    if k == 1:
                        decisive = True
                    else:
                        p_ref_sorted = np.sort(p_ref)[::-1]
                        decisive = float(p_ref_sorted[0] - p_ref_sorted[1]) > 0.10
                        if not decisive:
                            near_ties += 1
                            near_ties_flipped += int(top1_ort != top1_ref)
                    if decisive:
                        n_decisive += 1
                        agree_decisive += int(top1_ort == top1_ref)
                    max_dp = max(max_dp, float(np.abs(p_ort - p_ref).max()))
                    max_dact = max(max_dact, abs(float(out_act[i]) - case["raw"][qid]["act_prob"]))
            fixtures.append(case)
        with open(os.path.join(args.out, "fixtures.json"), "w") as f:
            json.dump(fixtures, f, ensure_ascii=False)
        print(f"wrote {len(fixtures)} fixtures; worst ORT-vs-PyTorch logit diff {worst:.2e}")
        print(f"top-1 agreement on decisive questions {agree_decisive}/{n_decisive}; near-ties {near_ties} "
              f"({near_ties_flipped} flipped); max |Δp| {max_dp:.4f} (temperature-scaled); "
              f"max |Δ act_prob| {max_dact:.4f}")
        if args.quantize is None and worst > 1e-2:
            raise SystemExit("parity self-check failed: ORT output differs from PyTorch")
        if args.quantize in ("int8", "w8"):
            del sess  # release the file before moving or deleting it
            gate_failed = agree_decisive < n_decisive or max_dp > 0.05 or max_dact > 0.05
            if gate_failed and not args.force:
                raise SystemExit(f"{args.quantize} decision gate failed: top-1 must match on every decisive "
                                  "question (reference top-2 margin > 0.10), |Δp| <= 0.05 and |Δ act_prob| <= 0.05 "
                                  "on every question; kept the fp32 model.onnx (rerun with --force to override)")
            if gate_failed:
                print(f"WARNING: {args.quantize} decision gate failed (see numbers above) but --force was given; "
                      "writing the quantized model.onnx anyway")
            os.replace(check_path, onnx_path)
    finally:
        # Any exception or interrupt before the gate (or a gate failure) leaves the quantized temp file
        # behind unless we clean it up here; a successful gate already moved it onto onnx_path.
        if check_path != onnx_path and os.path.exists(check_path):
            os.remove(check_path)
    write_manifest(args.out)  # last: only a folder whose model.onnx is final gets a manifest


if __name__ == "__main__":
    main()
