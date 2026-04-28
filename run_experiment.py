"""
run_experiment.py — 4-condition evaluation pipeline.

Each condition writes results/condition_{a,b,c,d}.json with the same schema:
  [
    {
      "sample_id": "...",
      "level": 1|2|3,
      "predictions": [
        {"probe_id": "...", "question": "...", "gold": "...", "predicted": "..."}
      ]
    }
  ]

Usage:
    python run_experiment.py --condition all
    python run_experiment.py --condition b --limit 5    # smoke test

Implementation note — TTT cannot be disabled by a runtime flag. The upstream
modeling code attaches `ttt_proj` and `ttt_conv` at construction time, gated on
`config.ttt_mode` and `config.ttt_layers`. Setting `config.ttt_mode = False` after
init is a no-op for inference. To make the model behave as vanilla Qwen3-8B for
conditions A and C, we *zero* the trained TTT parameters; their forward path
still executes but produces the original `down_proj.weight` output bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from model_utils import (
    fresh_ttt_cache,
    kv_stripped_clone,
    load_ttt_params,
    load_ttt_qwen3,
    restore_ttt_params,
    snapshot_ttt_params,
    zero_ttt_params,
)


SYSTEM_PROMPT = "你是一个助手，请用最简短的方式直接回答问题。只输出答案，不要解释。"


def render_prompt(conversation: str | None, question: str) -> str:
    if conversation:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"以下是一段对话内容：\n{conversation}\n\n"
            f"问题：{question}\n回答："
        )
    return f"{SYSTEM_PROMPT}\n\n问题：{question}\n回答："


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, past_kv=None, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if past_kv is None:
        # Provide a fresh TTTDynamicCache so the TTT layer code path has the
        # right cache type (regular DynamicCache lacks .ttt_states).
        past_kv = fresh_ttt_cache()
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        past_key_values=past_kv,
    )
    out = model.generate(**gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_condition_a(model, tokenizer, samples) -> list[dict]:
    """A — context baseline. TTT params zeroed (functional vanilla), conversation in context."""
    out = []
    for s in tqdm(samples, desc="cond A"):
        item = {"sample_id": s["sample_id"], "level": s["level"], "predictions": []}
        for probe in s["probes"]:
            prompt = render_prompt(s["conversation"], probe["question"])
            # Fresh cache per probe (so prior probe's KV doesn't leak)
            ans = generate_answer(model, tokenizer, prompt, past_kv=fresh_ttt_cache())
            item["predictions"].append({
                "probe_id": probe["probe_id"],
                "question": probe["question"],
                "gold": probe["gold_answer"],
                "predicted": ans,
            })
        out.append(item)
    return out


def run_condition_b(model, tokenizer, samples) -> list[dict]:
    """B — the experiment. TTT params loaded from trained checkpoint. Forward
    conversation through TTT path → fast weight encodes facts → strip KV → probe."""
    out = []
    for s in tqdm(samples, desc="cond B"):
        # Forward conversation with TTT enabled (params already trained at this point)
        cache = fresh_ttt_cache()
        conv_inputs = tokenizer(
            s["conversation"], return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)
        with torch.no_grad():
            _ = model(
                input_ids=conv_inputs["input_ids"],
                attention_mask=conv_inputs["attention_mask"],
                past_key_values=cache,
                use_cache=True,
            )
        # cache.ttt_states[i][2] (past_w) is now conversation-modified

        item = {"sample_id": s["sample_id"], "level": s["level"], "predictions": []}
        for probe in s["probes"]:
            per_probe_cache = kv_stripped_clone(cache)
            prompt = render_prompt(None, probe["question"])
            ans = generate_answer(model, tokenizer, prompt, past_kv=per_probe_cache)
            item["predictions"].append({
                "probe_id": probe["probe_id"],
                "question": probe["question"],
                "gold": probe["gold_answer"],
                "predicted": ans,
            })
        out.append(item)
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def run_condition_c(model, tokenizer, samples) -> list[dict]:
    """C — no-memory baseline. TTT params zeroed, no context, just probe alone."""
    out = []
    for s in tqdm(samples, desc="cond C"):
        item = {"sample_id": s["sample_id"], "level": s["level"], "predictions": []}
        for probe in s["probes"]:
            prompt = render_prompt(None, probe["question"])
            ans = generate_answer(model, tokenizer, prompt, past_kv=fresh_ttt_cache())
            item["predictions"].append({
                "probe_id": probe["probe_id"],
                "question": probe["question"],
                "gold": probe["gold_answer"],
                "predicted": ans,
            })
        out.append(item)
    return out


def run_condition_d(model, tokenizer, samples) -> list[dict]:
    """D — TTT memory + distractor (level 3 only)."""
    out = []
    for s in tqdm(samples, desc="cond D"):
        if not s.get("distractor"):
            continue
        cache = fresh_ttt_cache()
        full_text = s["conversation"] + "\n\n" + s["distractor"]
        toks = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=8192
        ).to(model.device)
        with torch.no_grad():
            _ = model(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                past_key_values=cache,
                use_cache=True,
            )
        item = {"sample_id": s["sample_id"], "level": s["level"], "predictions": []}
        for probe in s["probes"]:
            per_probe_cache = kv_stripped_clone(cache)
            prompt = render_prompt(None, probe["question"])
            ans = generate_answer(model, tokenizer, prompt, past_kv=per_probe_cache)
            item["predictions"].append({
                "probe_id": probe["probe_id"],
                "question": probe["question"],
                "gold": probe["gold_answer"],
                "predicted": ans,
            })
        out.append(item)
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def select_samples(samples, condition: str, limit: int | None) -> list[dict]:
    if condition == "d":
        sel = [s for s in samples if s["level"] == 3]
    else:
        sel = list(samples)
    if limit:
        per_level: dict[int, list] = {}
        for s in sel:
            per_level.setdefault(s["level"], []).append(s)
        sel = []
        for L, items in sorted(per_level.items()):
            sel.extend(items[:limit])
    return sel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark_v1.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ttt_minimal.pt"))
    parser.add_argument("--condition", choices=["a", "b", "c", "d", "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ttt-chunk", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.benchmark.read_text())
    print(f"[run] benchmark: {len(data['samples'])} samples")

    print(f"[run] loading model (ttt_chunk={args.ttt_chunk})")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=args.ttt_chunk)
    load_ttt_params(model, args.checkpoint)
    trained_snapshot = snapshot_ttt_params(model)
    model.eval()
    print(f"[run] trained checkpoint loaded; will toggle between zero (A/C) and trained (B/D)")

    conds = ["a", "b", "c", "d"] if args.condition == "all" else [args.condition]

    for cond in conds:
        samples = select_samples(data["samples"], cond, args.limit)
        print(f"\n[run] === condition {cond.upper()} on {len(samples)} samples ===")
        t0 = time.time()

        # Set TTT param state for this condition
        if cond in ("a", "c"):
            zero_ttt_params(model)
        else:  # b, d
            restore_ttt_params(model, trained_snapshot)

        if cond == "a":
            res = run_condition_a(model, tokenizer, samples)
        elif cond == "b":
            res = run_condition_b(model, tokenizer, samples)
        elif cond == "c":
            res = run_condition_c(model, tokenizer, samples)
        elif cond == "d":
            res = run_condition_d(model, tokenizer, samples)
        else:
            raise ValueError(cond)

        out_path = args.out_dir / f"condition_{cond}.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"[run] -> {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
