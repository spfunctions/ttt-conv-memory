"""
sanity.py — pre-flight sanity checks for the TTT mechanism.

Five checks (matches SPEC.md):
  1. TTT-on vs TTT-off output diverges on long input.
  2. ttt_proj weight norm > 0, ttt_conv weight norm > 0.
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
    disable_ttt_updates,
    enable_ttt_updates,
    fresh_ttt_cache,
    kv_stripped_clone,
    load_ttt_params,
    load_ttt_qwen3,
    ttt_param_norms,
)


def long_dummy_text(tokenizer, target_tokens: int = 1500) -> str:
    """A piece of text that comfortably exceeds ttt_chunk so TTT update fires."""
    seed = "今天天气不错，我们讨论一下项目进度。负责人是张伟，工号7742，办公室在3号楼12层。"
    out = ""
    while len(tokenizer.encode(out)) < target_tokens:
        out += seed + " "
    return out


def check_1_ttt_changes_output(model, tokenizer) -> tuple[bool, str]:
    text = long_dummy_text(tokenizer, target_tokens=1500)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    # Pass 1 — TTT off
    disable_ttt_updates(model)
    with torch.no_grad():
        cache_off = fresh_ttt_cache()
        _ = model(input_ids=inp["input_ids"], past_key_values=cache_off, use_cache=True)
        probe = tokenizer("总结一下：", return_tensors="pt").to(model.device)
        out_off = model.generate(
            input_ids=probe["input_ids"], past_key_values=cache_off,
            max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text_off = tokenizer.decode(out_off[0], skip_special_tokens=True)

    # Pass 2 — TTT on
    enable_ttt_updates(model)
    with torch.no_grad():
        cache_on = fresh_ttt_cache()
        _ = model(input_ids=inp["input_ids"], past_key_values=cache_on, use_cache=True)
        out_on = model.generate(
            input_ids=probe["input_ids"], past_key_values=cache_on,
            max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text_on = tokenizer.decode(out_on[0], skip_special_tokens=True)

    if text_off == text_on:
        return False, f"identical outputs:\n  off: {text_off!r}\n  on : {text_on!r}"
    return True, f"differ ✓\n  off: {text_off!r}\n  on : {text_on!r}"


def check_2_ttt_params_nonzero(model) -> tuple[bool, str]:
    norms = ttt_param_norms(model)
    proj_norms = [v for k, v in norms.items() if "ttt_proj" in k]
    conv_norms = [v for k, v in norms.items() if "ttt_conv" in k]
    if not proj_norms or not conv_norms:
        return False, f"no TTT params found: {list(norms.keys())[:5]}"
    if min(proj_norms) <= 0 or sum(conv_norms) == 0:
        return False, (
            f"TTT params at init? proj norms {proj_norms[:3]}, conv norms {conv_norms[:3]} "
            f"(conv at init is zero — needs training)"
        )
    return True, f"proj norms {proj_norms[:2]}, conv norms {conv_norms[:2]}"


def _extract_past_w(cache, layer_idx: int = 0):
    if not hasattr(cache, "ttt_states") or not cache.ttt_states:
        return None
    if layer_idx >= len(cache.ttt_states):
        return None
    st = cache.ttt_states[layer_idx]
    if len(st) < 3 or st[2] is None:
        return None
    return st[2].detach().clone()


def check_3_fast_weight_reproducible(model, tokenizer) -> tuple[bool, str]:
    text = long_dummy_text(tokenizer, target_tokens=1500)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    enable_ttt_updates(model)

    # First pass
    cache_a = fresh_ttt_cache()
    with torch.no_grad():
        _ = model(input_ids=inp["input_ids"], past_key_values=cache_a, use_cache=True)
    pw_a = _extract_past_w(cache_a)

    # Second pass — same input
    cache_b = fresh_ttt_cache()
    with torch.no_grad():
        _ = model(input_ids=inp["input_ids"], past_key_values=cache_b, use_cache=True)
    pw_b = _extract_past_w(cache_b)

    if pw_a is None or pw_b is None:
        return False, f"could not extract past_w (cache has {len(cache_a.ttt_states) if hasattr(cache_a, 'ttt_states') else 0} ttt_states)"
    diff = (pw_a - pw_b).abs().mean().item()
    if diff > 1e-4:
        return False, f"past_w diverges across runs: mean abs diff = {diff:.6f}"
    return True, f"past_w reproducible: mean abs diff = {diff:.2e}"


def check_4_fast_weight_input_dependent(model, tokenizer) -> tuple[bool, str]:
    enable_ttt_updates(model)
    # Two genuinely different inputs of similar length
    text_a = long_dummy_text(tokenizer, target_tokens=1500)
    text_b = "苹果公司发布了新一代芯片，采用台积电3纳米工艺。" * 50

    inp_a = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=2000).to(model.device)
    inp_b = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=2000).to(model.device)

    cache_a = fresh_ttt_cache()
    cache_b = fresh_ttt_cache()
    with torch.no_grad():
        _ = model(input_ids=inp_a["input_ids"], past_key_values=cache_a, use_cache=True)
        _ = model(input_ids=inp_b["input_ids"], past_key_values=cache_b, use_cache=True)

    pw_a = _extract_past_w(cache_a)
    pw_b = _extract_past_w(cache_b)
    if pw_a is None or pw_b is None:
        return False, "could not extract past_w"
    diff = (pw_a - pw_b).abs().mean().item()
    if diff < 1e-4:
        return False, f"past_w identical for different inputs: {diff:.6f}"
    return True, f"past_w differs by input: mean abs diff = {diff:.4f}"


def check_5_kv_strip_works(model, tokenizer) -> tuple[bool, str]:
    text = long_dummy_text(tokenizer, target_tokens=1500)
    enable_ttt_updates(model)
    cache = fresh_ttt_cache()
    with torch.no_grad():
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        _ = model(input_ids=inp["input_ids"], past_key_values=cache, use_cache=True)

    stripped = kv_stripped_clone(cache)
    disable_ttt_updates(model)

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
    parser.add_argument("--ttt-chunk", type=int, default=1024)
    args = parser.parse_args()

    print("[sanity] loading model...")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=args.ttt_chunk)
    if args.checkpoint and args.checkpoint.exists():
        load_ttt_params(model, args.checkpoint)
    model.eval()

    checks = [
        ("1. TTT-on vs TTT-off output diverges", lambda: check_1_ttt_changes_output(model, tokenizer)),
        ("2. ttt_proj + ttt_conv params non-zero", lambda: check_2_ttt_params_nonzero(model)),
        ("3. fast weight reproducible", lambda: check_3_fast_weight_reproducible(model, tokenizer)),
        ("4. fast weight input-dependent", lambda: check_4_fast_weight_input_dependent(model, tokenizer)),
        ("5. KV-stripped cache forward works", lambda: check_5_kv_strip_works(model, tokenizer)),
    ]

    n_pass = 0
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok = False
            detail = f"{type(e).__name__}: {e}"
        marker = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{marker}] {name}")
        print(f"           {detail}")
        if ok:
            n_pass += 1

    print(f"\n[sanity] {n_pass}/{len(checks)} checks passed")
    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
