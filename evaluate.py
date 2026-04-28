"""
evaluate.py — compute EM, F1, breakdowns, render figures.

Run after `run_experiment.py`. Reads results/condition_{a,b,c,d}.json,
writes results/report.json + results/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Both ASCII and full-width Chinese punctuation. We assemble the character class
# from a string of literal punctuation characters and escape it cleanly.
_PUNCT_CHARS = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "！？｡。，、；：「」『』《》【】〔〕（）—–"
    "…·～‧〜｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝】"
)
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")


def normalize(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def em_score(gold: str, pred: str) -> int:
    """Strict containment: normalized gold appears in normalized pred."""
    g, p = normalize(gold), normalize(pred)
    if not g:
        return 0
    return int(g in p)


def f1_score(gold: str, pred: str) -> float:
    """Char-level F1 (Chinese-friendly).

    Token-level F1 with English requires whitespace tokenization which is
    meaningless for Chinese, so we go char-level. Approximation: same as
    SQuAD F1 in spirit.
    """
    g_chars = list(normalize(gold))
    p_chars = list(normalize(pred))
    if not g_chars or not p_chars:
        return 0.0
    gc, pc = Counter(g_chars), Counter(p_chars)
    overlap = sum((gc & pc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_chars)
    recall = overlap / len(g_chars)
    return 2 * precision * recall / (precision + recall)


def evaluate_condition(predictions, benchmark) -> dict:
    sample_lookup = {s["sample_id"]: s for s in benchmark["samples"]}

    n = 0
    em_total = 0.0
    f1_total = 0.0
    by_level: dict = defaultdict(lambda: [0, 0.0, 0.0])
    by_category: dict = defaultdict(lambda: [0, 0.0, 0.0])
    by_position: dict = defaultdict(lambda: [0, 0.0, 0.0])

    for sp in predictions:
        sample = sample_lookup.get(sp["sample_id"])
        if not sample:
            continue
        level = sample["level"]
        facts_by_id = {f["fact_id"]: f for f in sample["facts"]}
        n_probes_in_sample = len(sample["probes"])

        for j, pp in enumerate(sp["predictions"]):
            n += 1
            em = em_score(pp["gold"], pp["predicted"])
            f1 = f1_score(pp["gold"], pp["predicted"])
            em_total += em
            f1_total += f1

            by_level[level][0] += 1
            by_level[level][1] += em
            by_level[level][2] += f1

            probe = next((p for p in sample["probes"] if p["probe_id"] == pp["probe_id"]), None)
            if probe:
                for fid in probe["required_facts"]:
                    fact = facts_by_id.get(fid)
                    if fact:
                        by_category[fact["category"]][0] += 1
                        by_category[fact["category"]][1] += em
                        by_category[fact["category"]][2] += f1

            # Position bucket — only meaningful for L1/L3 where probes are 1:1 with facts in order
            if level in (1, 3):
                pos_pct = j / max(n_probes_in_sample, 1)
                bucket = "front" if pos_pct < 1 / 3 else "middle" if pos_pct < 2 / 3 else "back"
                by_position[bucket][0] += 1
                by_position[bucket][1] += em
                by_position[bucket][2] += f1

    def to_dict(d):
        return {
            k: {"n": v[0], "em": v[1] / max(v[0], 1), "f1": v[2] / max(v[0], 1)}
            for k, v in d.items()
        }

    return {
        "overall": {
            "n": n,
            "em": em_total / max(n, 1),
            "f1": f1_total / max(n, 1),
        },
        "by_level": to_dict(by_level),
        "by_category": to_dict(by_category),
        "by_position": to_dict(by_position),
    }


def make_figures(reports: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conds_present = [c for c in ("a", "b", "c", "d") if c in reports]

    # --- Figure 1: EM bar by condition × level
    fig, ax = plt.subplots(figsize=(11, 6))
    levels = [1, 2, 3]
    width = 0.18
    x = np.arange(len(levels))
    for i, c in enumerate(conds_present):
        ems = []
        for L in levels:
            stats = reports[c]["by_level"].get(L) or reports[c]["by_level"].get(str(L))
            ems.append(stats["em"] if stats else 0.0)
        ax.bar(x + i * width, ems, width, label=f"Cond {c.upper()}")
    ax.set_xticks(x + (len(conds_present) - 1) * width / 2)
    ax.set_xticklabels([f"L{L}" for L in levels])
    ax.set_ylabel("Exact Match")
    ax.set_ylim(0, 1)
    ax.set_title("EM by Condition × Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_em_by_condition_level.png", dpi=120)
    plt.close(fig)

    # --- Figure 2: heatmap conditions × categories
    cats: list[str] = sorted({c for r in reports.values() for c in r["by_category"]})
    if cats and conds_present:
        mat = []
        for c in conds_present:
            row = [reports[c]["by_category"].get(cat, {}).get("em", 0.0) for cat in cats]
            mat.append(row)
        fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(cats)), 1.0 + 0.7 * len(conds_present)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_yticks(range(len(conds_present)))
        ax.set_yticklabels([f"Cond {c.upper()}" for c in conds_present])
        for i in range(len(conds_present)):
            for j in range(len(cats)):
                color = "white" if mat[i][j] < 0.6 else "black"
                ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, label="EM")
        ax.set_title("EM by Condition × Fact Category")
        fig.tight_layout()
        fig.savefig(out_dir / "fig2_em_heatmap.png", dpi=120)
        plt.close(fig)

    # --- Figure 3: position decay for cond B
    if "b" in reports:
        positions = ["front", "middle", "back"]
        ems = [reports["b"]["by_position"].get(p, {}).get("em", 0.0) for p in positions]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(positions, ems, "o-", linewidth=2, markersize=12, color="C1")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact Match")
        ax.set_xlabel("Position in conversation")
        ax.set_title("Cond B — fact recall by conversation position\n(does TTT have a forgetting gradient?)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_position_decay.png", dpi=120)
        plt.close(fig)

    # --- Figure 4: distractor effect, B vs D on level 3
    if "b" in reports and "d" in reports:
        em_b3 = reports["b"]["by_level"].get(3, {}).get("em") or reports["b"]["by_level"].get("3", {}).get("em") or 0.0
        em_d = reports["d"]["overall"]["em"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(["B (no distractor)", "D (with distractor)"], [em_b3, em_d], color=["C0", "C3"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact Match (Level 3 samples)")
        ax.set_title("Distractor effect on TTT memory")
        for i, v in enumerate([em_b3, em_d]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fig.tight_layout()
        fig.savefig(out_dir / "fig4_distractor_effect.png", dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark_v1.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/report.json"))
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())

    reports: dict = {}
    for cond in ("a", "b", "c", "d"):
        path = args.results_dir / f"condition_{cond}.json"
        if not path.exists():
            print(f"[eval] skipping {cond.upper()} — no file at {path}")
            continue
        preds = json.loads(path.read_text())
        rep = evaluate_condition(preds, benchmark)
        reports[cond] = rep
        print(f"[eval] cond {cond.upper()}: EM={rep['overall']['em']:.3f}  "
              f"F1={rep['overall']['f1']:.3f}  n={rep['overall']['n']}")

    headline: dict = {}
    if "a" in reports and "b" in reports:
        em_a = reports["a"]["overall"]["em"]
        em_b = reports["b"]["overall"]["em"]
        em_c = reports.get("c", {}).get("overall", {}).get("em", 0.0)
        headline["em_a_context_baseline"] = em_a
        headline["em_b_ttt_memory"] = em_b
        headline["em_c_no_memory"] = em_c
        headline["memory_efficiency_ratio"] = em_b / max(em_a, 1e-6)
        headline["lift_over_no_memory"] = em_b - em_c

        if headline["memory_efficiency_ratio"] > 0.7:
            headline["verdict"] = "POSITIVE — TTT fast weights substantially substitute for context window"
        elif headline["memory_efficiency_ratio"] > 0.3:
            headline["verdict"] = "PARTIAL — mechanism works but at significantly reduced fidelity"
        else:
            headline["verdict"] = "NEGATIVE — fast weights do not substitute for context under this setup"

    if "b" in reports and "d" in reports:
        em_b3 = (reports["b"]["by_level"].get(3, {}).get("em")
                 or reports["b"]["by_level"].get("3", {}).get("em") or 0.0)
        em_d = reports["d"]["overall"]["em"]
        headline["em_b_level3"] = em_b3
        headline["em_d_with_distractor"] = em_d
        headline["distractor_decay_rate"] = (em_d - em_b3) / max(em_b3, 1e-6)

    final = {"headline": headline, "by_condition": reports}
    args.out.write_text(json.dumps(final, ensure_ascii=False, indent=2))

    figures_dir = args.out.parent / "figures"
    make_figures(reports, figures_dir)

    print("\n=== HEADLINE ===")
    for k, v in headline.items():
        if isinstance(v, float):
            print(f"  {k:34s} = {v:7.3f}")
        else:
            print(f"  {k:34s} = {v}")
    print(f"\n[eval] report -> {args.out}")
    print(f"[eval] figures -> {figures_dir}")


if __name__ == "__main__":
    main()
