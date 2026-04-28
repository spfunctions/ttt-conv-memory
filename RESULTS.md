# RESULTS — TTT conversation-memory experiment

> This file is filled in after `evaluate.py` runs. Until then it's a template.
> Source data: `results/report.json`, figures in `results/figures/`.

## TL;DR

(headline number + verdict, 1-2 sentences)

## Setup

| | |
|---|---|
| Vehicle | ByteDance-Seed/In-Place-TTT (Qwen3-8B, ICLR 2026 oral) |
| Checkpoint | self-trained B-mini (see `DECISIONS.md` D-001) |
| Trained params | `ttt_proj` + `ttt_conv` only, base frozen |
| Training | (steps × seq × batch — fill in from train_log.jsonl) |
| `ttt_chunk` | 64 |
| TTT layers | `[0, 6, 12, 18, 24, 30]` (every 6th of 36) |
| Eval samples | 300 (100 per layer × 3 layers) |
| Eval probes | ~2620 (~11/sample × 3 levels, except L2) |
| Hardware | 1× A100-40G via Modal |
| Total cost | (fill in from STATE.md cost tracker) |

## Headline numbers

| Metric | Value | Interpretation |
|---|---|---|
| EM(A) — context baseline | (fill in) | Upper bound: model with conv in context |
| EM(B) — TTT memory | (fill in) | The experiment: only fast-weight memory |
| EM(C) — no memory | (fill in) | Lower bound: probe alone |
| **Memory efficiency ratio = EM(B) / EM(A)** | (fill in) | **The verdict number** |
| Lift over no-memory = EM(B) − EM(C) | (fill in) | Did fast weight add information? |
| EM(D) — TTT + distractor (L3 only) | (fill in) | |
| Distractor decay rate = (EM(D) − EM(B@L3)) / EM(B@L3) | (fill in) | How much distractor erodes memory |

### Verdict

(POSITIVE / PARTIAL / NEGATIVE — one sentence why)

## Per-level breakdown

(table with EM and F1 per condition × level)

## Per-category breakdown

(heatmap reference + commentary on which fact categories TTT favors)

## Per-position breakdown

(does TTT show a forgetting gradient? front vs middle vs back of conversation)

## Distractor effect

(reference to fig4 + commentary)

## Sanity-check results

(from `modal run modal_app.py::sanity` — pass/fail of all 5 checks)

## Surprises and caveats

(unexpected behaviors, things that pushed back on initial assumptions, things to be careful about)

## Implications for the original question

(does TTT replace context window for conversational memory? — what the data says about that specific claim)

## What we did NOT test

(scope limitations — paper-scale training, multi-turn fact accumulation, longer conversations, etc.)

## Reproduce

```bash
# On a Linux GPU host with cu128 + Python 3.11
bash setup.sh
python build_benchmark.py
python train_minimal.py --steps 400 --ttt-chunk 64 --seq-len 512 --batch-size 1 --grad-accum 4
python sanity.py --checkpoint checkpoints/ttt_minimal.pt
python run_experiment.py --condition all
python evaluate.py
```

Or end-to-end on Modal:

```bash
modal run modal_app.py::full_pipeline
modal run modal_app.py::pull_results --out-dir ./results
```

## Files

- `benchmark_v1.json` — the dataset (committed)
- `results/condition_{a,b,c,d}.json` — per-probe predictions
- `results/report.json` — aggregate metrics + headline
- `results/figures/*.png` — bar chart, heatmap, position decay, distractor effect
- `logs/train_log.jsonl` — per-step training loss
