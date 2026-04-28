"""
train_minimal.py — minimal continual-pretrain to bring TTT params out of init.

Self-supervised next-token prediction loss. Only ttt_proj + ttt_conv learn.
Base Qwen3-8B is frozen. Goal is to make the fast-weight delta meaningfully
non-zero, NOT to match paper-scale quality.

Run inside the GPU container:
    python train_minimal.py --steps 400 --ttt-chunk 1024 \
        --out checkpoints/ttt_minimal.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset

from model_utils import (
    freeze_base_train_ttt_only,
    load_ttt_qwen3,
    save_ttt_params,
    ttt_param_norms,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--ttt-chunk", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/no_robots")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ttt_minimal.pt"))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[train] loading model with ttt_chunk={args.ttt_chunk}")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=args.ttt_chunk)
    n_total, n_trainable = freeze_base_train_ttt_only(model)
    print(f"[train] trainable: {n_trainable / 1e6:.1f}M / total: {n_total / 1e9:.2f}B")

    init_norms = ttt_param_norms(model)
    print(f"[train] initial TTT param norms (sample): "
          f"{list(init_norms.items())[:3]}")

    print(f"[train] loading dataset {args.dataset}/{args.dataset_split}")
    ds = load_dataset(args.dataset, split=args.dataset_split)

    def render(ex):
        if "messages" in ex and ex["messages"]:
            return tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        # fallback: pick first text-like field
        for k in ("text", "prompt", "input", "content"):
            if k in ex and isinstance(ex[k], str):
                return ex[k]
        return None

    ds = ds.map(lambda ex: {"text": render(ex)})
    ds = ds.filter(lambda ex: ex["text"] is not None and 200 < len(ex["text"]) < args.seq_len * 6)
    print(f"[train] dataset filtered: {len(ds)} examples")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.parent / "train_log.jsonl"
    log_f = open(log_path, "w")

    model.train()
    t_start = time.time()
    losses: list[float] = []

    optim.zero_grad()
    for step in range(args.steps):
        # Sample a batch (deterministic walk through dataset, repeats if needed)
        idx_base = (step * args.batch_size) % max(len(ds) - args.batch_size, 1)
        batch_texts = [ds[idx_base + i]["text"] for i in range(args.batch_size)]
        toks = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = toks["input_ids"].to(model.device)
        attention_mask = toks["attention_mask"].to(model.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            optim.zero_grad()

        losses.append(loss.item() * args.grad_accum)
        log_entry = {
            "step": step,
            "loss": losses[-1],
            "elapsed": time.time() - t_start,
        }
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        if step % args.log_every == 0:
            recent = losses[-args.log_every:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t_start
            eta = elapsed / max(step + 1, 1) * (args.steps - step - 1)
            print(f"[train] step {step:4d}/{args.steps} | "
                  f"loss {avg:.4f} | elapsed {elapsed:.0f}s | eta {eta:.0f}s")

        if step > 0 and step % args.save_every == 0:
            ckpt_path = args.out.parent / f"ttt_step{step}.pt"
            save_ttt_params(model, ckpt_path)

    save_ttt_params(model, args.out)
    final_norms = ttt_param_norms(model)
    print(f"[train] final TTT param norms (sample): "
          f"{list(final_norms.items())[:3]}")
    # diagnostic — how much did TTT params move?
    total_init = sum(init_norms.values())
    total_final = sum(final_norms.values())
    print(f"[train] sum of TTT param norms: init={total_init:.2f} -> final={total_final:.2f}")
    log_f.close()
    print(f"[train] done. saved to {args.out}")


if __name__ == "__main__":
    main()
