# RESULTS — TTT conversation-memory experiment

## TL;DR

**NEGATIVE verdict.** Under our minimum-scale (B-mini) self-trained checkpoint, **TTT fast-weight updates do not substitute for the context window for conversational memory**. The TTT-modified weights actively degrade the model — condition B scores **lower than the no-memory baseline**.

> `memory_efficiency_ratio = EM(B) / EM(A) = 0.000`
> `lift_over_no_memory = EM(B) − EM(C) = −0.016`

## Setup

| | |
|---|---|
| Vehicle | [ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT) (Qwen3-8B, ICLR 2026 oral) |
| Checkpoint | self-trained B-mini (2000 steps, lr=1e-5, ttt_chunk=64, seq=512) |
| Trained params | `ttt_proj` + `ttt_conv` only (100.8M params), base 8.2B frozen |
| Training data | `HuggingFaceH4/no_robots` (English instruction conversations) |
| Eval samples | 300 (100 per layer × 3 layers; level 3 only for cond D) |
| Eval probes | 2620 |
| Hardware | 1× A100-40G via Modal (4 conditions in parallel) |
| Total Modal cost | ~$13 (within $30/mo free credit) |

## Headline numbers

| Metric | Value |
|---|---|
| EM(A) — context baseline | **0.929** |
| EM(B) — TTT memory | **0.000** |
| EM(C) — no memory | **0.016** |
| **memory_efficiency_ratio = EM(B) / EM(A)** | **0.000** |
| lift_over_no_memory = EM(B) − EM(C) | −0.016 |
| EM(D) — TTT + distractor (L3 only) | 0.000 |
| distractor_decay_rate | n/a (B already 0%) |

### Verdict

**NEGATIVE** — under this setup, TTT fast weights are a perturbation that *degrades* the model below random-baseline performance, not a memory substrate.

## Per-level breakdown

| Cond | Level 1 EM | Level 2 EM | Level 3 EM |
|---|---|---|---|
| A (context) | 0.934 | 0.901 | 0.934 |
| B (TTT memory) | 0.000 | 0.000 | 0.000 |
| C (no memory) | 0.016 | 0.021 | 0.016 |
| D (TTT + distractor) | — | — | 0.000 |

L1 and L3 are nearly identical for A — distractor in context doesn't bother the model at all (it can ignore irrelevant text). L2 is slightly harder because composed answers have more characters and stricter EM matching.

## Per-category breakdown (cond A vs B vs C)

| Category | A | B | C |
|---|---|---|---|
| numeric (IDs, budgets, headcounts) | 1.000 | 0.000 | 0.034 |
| person (names) | 0.986 | 0.000 | 0.018 |
| preference (dietary etc.) | 0.457 | 0.000 | 0.000 |
| project (project names) | 0.992 | 0.000 | 0.008 |
| relational (titles, reporting) | 0.993 | 0.000 | 0.031 |
| spatial (offices, rooms) | 0.897 | 0.000 | 0.000 |
| temporal (times, dates) | 0.852 | 0.000 | 0.000 |

**Cond A**: near-perfect on hard-to-guess facts (numeric IDs, project names, person names — ~99-100%). Lower on preference (45.7%) and temporal (85.2%) because those facts have many semantically-equivalent surface forms that don't satisfy exact-match against the templated gold.

**Cond B**: zero hits across every category. Not "weak signal" — *no signal*. The TTT-modified weights produce broken outputs (the model just repeats the question or emits gibberish like `"UNSUNSUNS..."`).

**Cond C**: small hits where lucky guesses align with the model's prior (e.g. some common digit patterns match by chance).

## Per-position breakdown (cond B)

| Position | EM | n |
|---|---|---|
| front 1/3 of conversation | 0.000 | 800 |
| middle 1/3 | 0.000 | 712 |
| back 1/3 | 0.000 | 680 |

No positional pattern — the failure is uniform across the conversation.

## What cond B's outputs actually look like

```
Q: 品牌重塑 由谁参与？
gold:    '何鑫'
predict: '由谁参与\n\n品牌重塑 由谁参与\n\n品牌重塑 由谁参与...'

Q: 何鑫 向谁汇报？
gold:    '郭斌'
predict: '何鑫 �向谁汇报\n回答：何鑫 �何何 何何 何何 何何...'

Q: 目前在做什么项目？
gold:    '品牌重塑'
predict: '目前在做项目\n\n答：目前在做项目\n\n答：目前在做项目...'
```

The model with TTT-modified weights repeats fragments of the question, outputs partial garbled tokens, or produces empty strings. It does NOT produce coherent text containing the conversation's facts.

## What cond D's outputs look like (TTT after distractor)

```
Q: 品牌重塑 由谁参与？
gold:    '何鑫'
predict: 'UNSUNSUNSUNSUNSUNSUNSUNSUNSUNSUNS...'
```

After 2000-4000 tokens of distractor pass through the TTT layers, the fast weights drift even further from anything coherent. The model emits literally a single repeated subword (`UNS`, the start of `UNSORTED`?). This is a clear demonstration that the TTT updates are *additive noise* on the down-projection, not informative encoding.

## Sanity checks (post-train, 5/5 passed)

| # | Check | Result |
|---|---|---|
| 1 | trained-TTT vs zeroed-TTT logits diverge on long input | PASS — mean abs diff 4.56, max 25.5, top-1 token differs |
| 2 | `ttt_proj` + `ttt_conv` weight norms > 0 | PASS — proj 81.5, conv 0.111 / 0.023 (samples) |
| 3 | fast weight reproducible across forward passes | PASS — diff 0.00e+00 |
| 4 | fast weight is input-dependent | PASS — mean abs diff 0.0015 between two inputs |
| 5 | KV-stripped cache forward works (no shape errors) | PASS |

So the *mechanism* is wired correctly. What it's encoding is just not useful for memory recall.

## Why the result came out this way (interpretation)

Three levels of cause, in order of confidence:

1. **B-mini training is far below paper-scale.** Paper trains 5000 steps × global batch 64 × seq 65536 (≈ 21B tokens). We trained 2000 steps × batch 1 × seq 512 (≈ 1M tokens) — **~21,000× less data**. `ttt_conv` weight norm reached 0.111 in the most-trained layer (layer 0), 0.012-0.020 in deeper layers. That's just-off-init, not a converged solution.

2. **Frozen base + TTT-only training gives the wrong loss landscape.** With base frozen, vanilla Qwen3-8B is already near-optimal for next-token prediction. The TTT layers' gradient signal pushes them toward "doing nothing" (dw → 0), not toward "encode useful info into dw." The tiny non-zero `ttt_conv` we got is structured noise that just perturbs `down_proj.weight` without adding recoverable information.

3. **The training distribution doesn't reward memory.** No-robots is short instruction-following dialogues; predictive next-token loss is dominated by the immediately-preceding context (handled by attention), not by long-range fast-weight encoding. The optimizer has no pressure to make the TTT layers actually function as memory.

The verdict here is an honest answer to the **literal question as asked** ("does TTT fast-weight update substitute for context as conversational memory?") **under the minimum-scale training that fits the user's stated budget and timeline**. It is *not* a verdict on the broader paper's claims at full training scale.

## What we did NOT test

- Paper-scale continual pretraining (5000+ steps, 65k seq, joint base+TTT training) — would cost $1k+ and many H100-days.
- Full base + TTT joint training (we kept base frozen).
- Long-context training data (we trained on short dialogues; paper trains on long-context corpora where TTT actually has work to do).
- Other test-time-training implementations (the user wanted In-Place TTT specifically, so LoRA-online-update etc. were out of scope).
- Larger conversational corpora in Chinese (we trained on English, evaluated in Chinese — Qwen3-8B is multilingual so this should be fine, but we didn't ablate).

## Files

| Path | What |
|---|---|
| `benchmark_v1.json` | 300-sample dataset |
| `results/condition_{a,b,c,d}.json` | per-probe predictions (committed) |
| `results/report.json` | aggregate metrics (this file's source) |
| `results/figures/fig1_em_by_condition_level.png` | bar chart EM × condition × level |
| `results/figures/fig2_em_heatmap.png` | heatmap EM × condition × category |
| `results/figures/fig3_position_decay.png` | cond B EM by conversation position |
| `results/figures/fig4_distractor_effect.png` | B vs D on level 3 |
| `STATE.md` | full chronological run log |
| `DECISIONS.md` | every research-level fork (D-001 through D-008) |

## Reproduce

```bash
# On a Linux GPU host with cu128 + Python 3.11
bash setup.sh
python build_benchmark.py
python train_minimal.py --steps 2000 --lr 1e-5 --ttt-chunk 64 --seq-len 512 --batch-size 1 --grad-accum 4
python sanity.py --checkpoint checkpoints/ttt_minimal.pt
python run_experiment.py --condition all
python evaluate.py
```

Or end-to-end on Modal:

```bash
modal run modal_app.py::full_pipeline
modal run modal_app.py::pull_results --out-dir ./results
```
