"""analyze_phase_a.py — qualitative + quantitative comparison of Phase A noise
sweep against cond B and cond C.

For each `condition_a_prime_<suffix>.json` in results/, compute:
  - mean prediction length (chars)
  - fraction of degenerate outputs (loop / single-char collapse / empty)
  - exact match rate
  - 5 random sample predictions for visual inspection

Compare against cond B (TTT memory, gibberish baseline) and cond C (vanilla, fluent baseline).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path


def is_loop(text: str, min_chunk: int = 4, min_consecutive: int = 3) -> bool:
    """Loop = a chunk (>= min_chunk chars) appears CONSECUTIVELY min_consecutive
    or more times anywhere in the text. Tighter than 'appears at least N times'
    so legitimate phrasing repetition (e.g. 'X is 50000 元 ... X is 50000 元 ...')
    isn't flagged."""
    n = len(text)
    if n < min_chunk * min_consecutive:
        return False
    for size in range(min_chunk, min(20, n // min_consecutive) + 1):
        for start in range(n - size * min_consecutive + 1):
            chunk = text[start:start + size]
            consec = 1
            for k in range(1, n // size):
                next_start = start + size * k
                if text[next_start:next_start + size] == chunk:
                    consec += 1
                    if consec >= min_consecutive:
                        return True
                else:
                    break
    return False


def is_single_char_collapse(text: str, min_run: int = 6) -> bool:
    if not text:
        return False
    return bool(re.search(r"(.)\1{" + str(min_run - 1) + r",}", text))


def is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if is_single_char_collapse(t):
        return True
    if is_loop(t):
        return True
    return False


def load_predictions(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for s in data:
        for p in s["predictions"]:
            out.append(p)
    return out


def stats(preds: list[dict]) -> dict:
    n = len(preds)
    if n == 0:
        return {"n": 0}
    em = sum(1 for p in preds if (p.get("predicted") or "").strip() == p["gold"]) / n
    lengths = [len(p.get("predicted") or "") for p in preds]
    deg = sum(1 for p in preds if is_degenerate(p.get("predicted") or "")) / n
    return {
        "n": n,
        "em": em,
        "degenerate_frac": deg,
        "mean_pred_len": statistics.mean(lengths),
        "median_pred_len": statistics.median(lengths),
    }


def sample_dump(preds: list[dict], k: int = 5, seed: int = 0) -> list[str]:
    random.seed(seed)
    sel = random.sample(preds, min(k, len(preds)))
    out = []
    for p in sel:
        text = (p.get("predicted") or "")[:120]
        out.append(f"  Q: {p['question']!r}\n    gold: {p['gold']!r}\n    pred: {text!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    rd = args.results_dir
    targets: list[tuple[str, Path]] = []

    # Baselines
    if (rd / "condition_b.json").exists():
        targets.append(("B (TTT memory)", rd / "condition_b.json"))
    if (rd / "condition_c.json").exists():
        targets.append(("C (no memory, vanilla)", rd / "condition_c.json"))

    # Phase A sweep
    for f in sorted(rd.glob("condition_a_prime_*.json")):
        suffix = f.stem.replace("condition_a_prime_", "")
        scale = float(suffix.replace("p", "."))
        targets.append((f"A' (noise {scale})", f))

    # Inference-time TTT scaling (scaled cond B)
    for f in sorted(rd.glob("condition_b_scaled_*.json")):
        suffix = f.stem.replace("condition_b_scaled_", "")
        scale = float(suffix.replace("p", "."))
        targets.append((f"B-scaled (α={scale})", f))

    rows: list[dict] = []
    for label, path in targets:
        preds = load_predictions(path)
        s = stats(preds)
        s["label"] = label
        rows.append(s)
        print(f"\n=== {label} ===")
        print(f"  n={s['n']}  EM={s['em']:.4f}  degenerate={s['degenerate_frac']:.3f}  "
              f"mean_len={s['mean_pred_len']:.1f}  median_len={s['median_pred_len']}")
        print("  --- 5 random samples ---")
        for line in sample_dump(preds, k=5, seed=42):
            print(line)

    print("\n\n=== SUMMARY ===")
    print(f"{'cond':<28} {'n':>5} {'EM':>7} {'deg%':>7} {'meanLen':>8}")
    for r in rows:
        print(f"{r['label']:<28} {r['n']:>5} {r['em']:>7.4f} "
              f"{r['degenerate_frac']*100:>6.1f}% {r['mean_pred_len']:>8.1f}")


if __name__ == "__main__":
    main()
