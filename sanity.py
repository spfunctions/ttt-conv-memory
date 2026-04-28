"""
sanity.py — pre-flight sanity checks for the TTT mechanism.

Five checks (matches SPEC.md). Run AFTER training, otherwise check #1 + #2 will
naturally fail because TTT params are at init (ttt_conv is zero-init).

  1. With trained TTT params, the model output differs from zero-TTT-params on
     the same long input. (i.e. TTT actually does something.)
  2. ttt_proj weight norm > 0, ttt_conv weight norm > 0 after training.
  3. Fast weight is reproducible (same input twice → same past_w).
  4. Fast weight is input-dependent (different inputs → different past_w).
  5. KV-stripped cache forward works (no shape errors, sensible output).

Exit code 0 = all pass, 1 = any fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from model_utils import (
    fresh_ttt_cache,
    kv_stripped_clone,
    load_ttt_params,
    load_ttt_qwen3,
    restore_ttt_params,
    snapshot_ttt_params,
    ttt_param_norms,
    zero_ttt_params,
)


def long_dummy_text(tokenizer, target_tokens: int = 200) -> str:
    """Text that comfortably exceeds ttt_chunk so TTT update fires."""
    seed = "今天天气不错，我们讨论一下项目进度。负责人是张伟，工号7742，办公室在3号楼12层。"
    out = ""
    while len(tokenizer.encode(out)) < target_tokens:
        out += seed + " "
    return out


def check_1_trained_vs_zero(model, tokenizer, trained_snapshot) -> tuple[bool, str]:
    """Compare logits (not generated text) under zeroed vs trained TTT params on
    a long single input. Logit comparison is more sensitive than greedy-decode
    text — even small fast-weight effects show up as logit shifts even if greedy
    decoding lands on the same first token."""
    text = long_dummy_text(tokenizer, target_tokens=300) + "\n请问会议室在哪？答："
    inp = tokenizer(text, return_tensors="pt").to(model.device)

    try:
        # Pass 1: zeroed
        zero_ttt_params(model)
        with torch.no_grad():
            out_zero = model(input_ids=inp["input_ids"], past_key_values=fresh_ttt_cache(), use_cache=True)
        logits_zero = out_zero.logits[0, -1, :].detach().clone()

        # Pass 2: trained
        restore_ttt_params(model, trained_snapshot)
        with torch.no_grad():
            out_trained = model(input_ids=inp["input_ids"], past_key_values=fresh_ttt_cache(), use_cache=True)
        logits_trained = out_trained.logits[0, -1, :].detach().clone()
    finally:
        # Always restore so subsequent checks see trained params
        restore_ttt_params(model, trained_snapshot)

    diff = (logits_zero - logits_trained).abs().mean().item()
    max_diff = (logits_zero - logits_trained).abs().max().item()
    top1_zero = int(logits_zero.argmax())
    top1_trained = int(logits_trained.argmax())
    same_top1 = top1_zero == top1_trained

    if diff < 1e-6:
        return False, f"logits identical (mean abs diff = {diff:.2e}) — TTT params have no effect on output"
    return True, (
        f"logits differ: mean abs diff = {diff:.4f}, max abs diff = {max_diff:.4f}, "
        f"top1 token same={same_top1} (zero={top1_zero}, trained={top1_trained})"
    )


def check_2_ttt_params_nonzero(model) -> tuple[bool, str]:
    norms = ttt_param_norms(model)
    proj_norms = [v for k, v in norms.items() if "ttt_proj" in k]
    conv_norms = [v for k, v in norms.items() if "ttt_conv" in k]
    if not proj_norms or not conv_norms:
        return False, f"no TTT params found in model: {list(norms.keys())[:5]}"
    if min(proj_norms) <= 0 or sum(conv_norms) == 0:
        return False, (
            f"TTT params at init? proj norms[:3] {proj_norms[:3]}, conv norms[:3] {conv_norms[:3]} "
            f"(zero ttt_conv = untrained — needs training)"
        )
    return True, f"proj norms[:2] {[f'{x:.3f}' for x in proj_norms[:2]]}, conv norms[:2] {[f'{x:.3f}' for x in conv_norms[:2]]}"


def _extract_past_w(cache, layer_idx: int = 0):
    if not hasattr(cache, "ttt_states") or not cache.ttt_states:
        return None
    if layer_idx >= len(cache.ttt_states):
        return None
    st = cache.ttt_states[layer_idx]
    if len(st) < 3 or st[2] is None:
        return None
    return st[2].detach().clone()


def check_3_fast_weight_reproducible(model, tokenizer, ttt_layer_idx: int = 0) -> tuple[bool, str]:
    text = long_dummy_text(tokenizer, target_tokens=200)
    inp = tokenizer(text, return_tensors="pt").to(model.device)

    cache_a = fresh_ttt_cache()
    with torch.no_grad():
        _ = model(input_ids=inp["input_ids"], past_key_values=cache_a, use_cache=True)
    pw_a = _extract_past_w(cache_a, ttt_layer_idx)

    cache_b = fresh_ttt_cache()
    with torch.no_grad():
        _ = model(input_ids=inp["input_ids"], past_key_values=cache_b, use_cache=True)
    pw_b = _extract_past_w(cache_b, ttt_layer_idx)

    if pw_a is None or pw_b is None:
        n = len(cache_a.ttt_states) if hasattr(cache_a, "ttt_states") else 0
        return False, f"could not extract past_w from layer {ttt_layer_idx} (cache has {n} ttt_states)"
    diff = (pw_a - pw_b).abs().mean().item()
    if diff > 1e-4:
        return False, f"past_w diverges across runs: mean abs diff = {diff:.6f}"
    return True, f"past_w reproducible: mean abs diff = {diff:.2e}"


def check_4_fast_weight_input_dependent(model, tokenizer, ttt_layer_idx: int = 0) -> tuple[bool, str]:
    text_a = long_dummy_text(tokenizer, target_tokens=200)
    text_b = "苹果公司发布了新一代芯片，采用台积电3纳米工艺。" * 30

    inp_a = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=400).to(model.device)
    inp_b = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=400).to(model.device)

    cache_a = fresh_ttt_cache()
    cache_b = fresh_ttt_cache()
    with torch.no_grad():
        _ = model(input_ids=inp_a["input_ids"], past_key_values=cache_a, use_cache=True)
        _ = model(input_ids=inp_b["input_ids"], past_key_values=cache_b, use_cache=True)

    pw_a = _extract_past_w(cache_a, ttt_layer_idx)
    pw_b = _extract_past_w(cache_b, ttt_layer_idx)
    if pw_a is None or pw_b is None:
        return False, "could not extract past_w"
    diff = (pw_a - pw_b).abs().mean().item()
    if diff < 1e-4:
        return False, f"past_w identical for different inputs (diff={diff:.6f})"
    return True, f"past_w differs by input: mean abs diff = {diff:.4f}"


def check_5_kv_strip_works(model, tokenizer) -> tuple[bool, str]:
    text = long_dummy_text(tokenizer, target_tokens=200)
    cache = fresh_ttt_cache()
    with torch.no_grad():
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        _ = model(input_ids=inp["input_ids"], past_key_values=cache, use_cache=True)

    stripped = kv_stripped_clone(cache)

    probe = tokenizer("会议室在哪？", return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=probe["input_ids"],
                past_key_values=stripped,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        return True, f"forward succeeded; sample answer: {ans!r}"
    except Exception as e:
        return False, f"KV-stripped forward failed: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ttt-chunk", type=int, default=64)
    args = parser.parse_args()

    print("[sanity] loading model...")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=args.ttt_chunk)
    if args.checkpoint and Path(args.checkpoint).exists():
        load_ttt_params(model, args.checkpoint)
    else:
        print(f"[sanity] WARNING: no checkpoint at {args.checkpoint}")
        print("[sanity] checks 1+2 will likely fail — run train_minimal.py first.")
    trained_snapshot = snapshot_ttt_params(model)
    model.eval()

    checks = [
        ("1. trained TTT vs zeroed TTT output diverges",
         lambda: check_1_trained_vs_zero(model, tokenizer, trained_snapshot)),
        ("2. ttt_proj + ttt_conv params non-zero (post-train)",
         lambda: check_2_ttt_params_nonzero(model)),
        ("3. fast weight reproducible across forward passes",
         lambda: check_3_fast_weight_reproducible(model, tokenizer)),
        ("4. fast weight is input-dependent",
         lambda: check_4_fast_weight_input_dependent(model, tokenizer)),
        ("5. KV-stripped cache forward works",
         lambda: check_5_kv_strip_works(model, tokenizer)),
    ]

    n_pass = 0
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            import traceback
            ok = False
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc().splitlines()[-3:]}"
        marker = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{marker}] {name}")
        for line in detail.splitlines():
            print(f"           {line}")
        if ok:
            n_pass += 1

    # Restore trained params before exit so the caller can continue using the model
    restore_ttt_params(model, trained_snapshot)

    print(f"\n[sanity] {n_pass}/{len(checks)} checks passed")
    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
