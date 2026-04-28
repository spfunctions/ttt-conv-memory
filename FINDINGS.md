# FINDINGS — what went wrong, and what to try next

Iteration 1 (this commit) ended with `EM(B)/EM(A) = 0/0.929 = 0.000`. Below: what
I think actually broke, ranked by how confident I am, and a concrete iteration-2
plan with cost estimates.

## Iteration-2 final verdict — past_w contains no conversation memory

The inference-time TTT scaling experiment settled the question.
α ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} sweeps `ttt_proj`/`ttt_conv` weights at
inference time and re-runs cond B. The strongest signal isn't degenerate% —
it's the rate at which the gold answer appears anywhere in the prediction:

| condition          | n    | gold-in-pred |
|--------------------|------|--------------|
| A (context)        | 2617 | **92.2%** |
| C (vanilla)        | 2617 | 1.6 % |
| B (original)       | 2617 | **0.0 %** |
| B-scaled α=0.1     | 259  | 0.0 % |
| B-scaled α=0.2     | 259  | 0.0 % |
| B-scaled α=0.3     | 259  | 0.0 % |
| B-scaled α=0.5     | 259  | 0.0 % |
| B-scaled α=0.7     | 259  | 0.0 % |
| B-scaled α=1.0     | 259  | 0.0 % |

Even at α=0.1, where the model is mostly fluent (25% degenerate, mostly
hallucinated cond-C-style answers) and the TTT perturbation is tiny (layer 30
rel_dw 0.037), it produces **zero** fact retrievals. Cond C — the same model
with no memory channel at all — gets 1.6% by chance. So magnitude tame or
not, past_w simply has no conversation content to retrieve.

This is conclusive: **the TTT did not encode memory**. Frozen-base
+ short-Q&A training gave the optimizer no incentive to push fast weights
toward "encode something useful." It pushed them toward "produce noise of
some magnitude." At full magnitude, that noise is large enough to crash the
base; at small magnitude, it's small enough not to crash but still not
useful. There is no operating point where this trained TTT is a memory
channel.

The frozen-base / magnitude story from earlier in this file is real but
secondary — it explains why cond B's outputs *look* broken at α=1.0, but it
doesn't explain why retrieval is zero at α=0.1 too. Hypothesis #2 from
RESULTS.md ("Frozen base + TTT-only training gives the wrong loss
landscape") is the dominant cause. Hypothesis #1 (training scale) and the
later "frozen base ≠ perturbation-robust" framing are both real but downstream.

### What it would take to flip the verdict

Iteration-2 has done as much as a frozen-base + Q&A-data setup can teach us.
Any "iteration-3" worth running has to attack the loss landscape, not the
mechanism:

- **Joint base+TTT training** (LoRA rank 16 on base, full TTT, 5000 steps,
  long-context corpus, seq_len 4096). Estimated $20-30 — beyond remaining
  free Modal credit. Predicted outcome: dw magnitudes stay reasonable and
  *some* retrieval emerges if the corpus has long-range dependencies. But
  this is a substantial commitment, and the upside is "maybe TTT-as-memory
  works at small scale," not "definitely solves Patrick's problem."
- **Magnitude controls during training** (smaller `ttt_proj` init scale, weight
  decay 0.01 on TTT params, grad clip). Standalone these would have prevented
  the magnitude blow-up in iteration-1 but, per the data above, would not
  have produced retrieval — they'd just have moved cond B from "gibberish"
  to "fluent hallucination." Worth pairing with the joint training; not
  worth running alone.
- **Pivot the question.** The literal question "does TTT replace context for
  conversational memory at affordable training scale" has a clean negative
  answer here. Further work should either (a) accept that and move on, or
  (b) reframe as "what *can* a small TTT learn?" — e.g. style transfer,
  formatting, single-turn instruction tuning — which are easier loss
  landscapes that don't require long-range memory.

## Iteration-2 step 2 — Phase A noise sweep (frozen-base sub-hypothesis)

Phase A finished. Random Gaussian noise of matching magnitude reproduces cond B's
failure mode, so there is nothing special about the direction TTT learned —
any perturbation of that magnitude breaks the frozen base.

| condition           | n   | degenerate% |
|---------------------|-----|-------------|
| C (vanilla)         | 2617| 11.6 |
| A' noise 0.01       | 259 | 6.6 |
| A' noise 0.05       | 259 | 14.3 |
| A' noise 0.10       | 259 | 20.1 |
| A' noise 0.20       | 259 | 38.2 |
| A' noise 0.50       | 259 | 29.7 |
| A' noise **1.02**   | 259 | **64.1** |
| A' noise 3.19       | 259 | 99.6 |
| B (TTT memory)      | 2617| 74.0 |

The vanilla base handles ≤5% relative noise gracefully (deg% basically equal to
its own intrinsic 11.6% baseline). It enters a transition zone around 10-20%.
By 100% noise it is mostly broken; by 320% it is entirely token-soup. TTT's
trained dw lands at 8% (layer 0) → 38% (layer 12) → **319%** (layer 30). The
deep-layer perturbations are well past the base's absorption budget.

So the bottleneck is **TTT magnitude blow-up at depth**, not "TTT learned the
wrong direction" and not "frozen base is intrinsically too fragile." The
direction is roughly random-equivalent; the base is reasonably robust *up to a
budget*; TTT just overspends that budget by 30× at the deepest layer.

Cheap follow-up experiments, in order of decisiveness per dollar:

1. **Inference-time TTT scaling** (~$1, 30 min). Load the same trained
   checkpoint, multiply `ttt_proj.weight` and `ttt_conv.weight` by a series of
   factors (e.g. 0.05, 0.1, 0.2, 0.3) before running cond B. If a smaller dw
   produces coherent (even if wrong) cond B output, magnitude alone is the
   gating issue and we can ship a magnitude-controlled retrain. If outputs are
   still gibberish at every scale, something deeper is wrong with what TTT
   learned.

2. **Per-layer ablation** (~$1, 30 min). Keep trained TTT for layers 0/6/12 but
   zero out layers 18/24/30 (the magnitude offenders). Run cond B. If outputs
   become coherent, the deep layers are specifically where the encoding fails;
   the shallow ones may even be carrying useful signal.

3. **Retrain with magnitude controls** (~$5, 1 hr). Reduce `ttt_proj` init
   scale (default Linear init at norm 81.5 → ~10), add weight decay 0.01 on
   ttt_proj/ttt_conv, optionally clip grad norm at 1.0. 2000 steps, lr 1e-5.
   Re-run all four conditions. Predicted outcome: dw magnitudes stay <10%,
   cond B becomes fluent, and we get a real read on whether *information* is
   encoded in the fast weights (vs just "magnitudes were too big").

Experiment 1 was run; see "Iteration-2 final verdict" above. Result:
the magnitude story alone does not explain failure — past_w contains no
information at any scale.

## Iteration-2 step 1 — dw magnitude measurement

Before running Phase A's noise sweep we measured ||dw||_F / ||W||_F per TTT layer
(output-diff method on a real benchmark conversation, against the unchanged
down_proj.weight). The result invalidated my "5–10%" guess from this file's
original draft:

| Layer | rel_dw |
|---|---|
| 0  | 0.082 |
| 6  | 0.029 |
| 12 | 0.382 |
| 18 | 0.807 |
| 24 | **1.652** |
| 30 | **3.188** |

The trained TTT does not produce a small perturbation of `down_proj.weight` —
at layer 30 it produces an MLP output **3.2× the magnitude of what the vanilla
weight would produce on the same hidden state**. The deep layers are no longer
"vanilla MLP plus a nudge"; the TTT term dominates.

This shifts the diagnosis. Frozen-base fragility may still matter, but it is
not the deepest cause — even a robust base couldn't make sense of an MLP whose
output at layer 30 is 3× its baseline magnitude in a near-random direction.
The dominant cause is **TTT magnitude blow-up at depth**.

Implications for the iteration-2 plan:
- **Phase B as originally drafted (joint LoRA + TTT)** would not necessarily
  fix this. It might give the base capacity to absorb large dw, but it doesn't
  attack the magnitude problem at the source.
- **Cheaper, more targeted attempts** worth trying first:
  1. **Scale down `ttt_proj`/`ttt_conv` at inference** (no retraining, ~$1).
     If dw scales linearly with these tensors, halving them should roughly halve
     the rel_dw. Worth testing whether a smaller perturbation produces coherent
     (just wrong) cond B outputs.
  2. **Reduce `ttt_proj` init scale before retraining** (norm 81.5 → ~10).
     Smaller init = smaller dw throughout training = less exposed to magnitude
     blow-up.
  3. **Reduce `ttt_lr`** (currently 3.0 — the inner-loop fast-weight learning
     rate). Lower ttt_lr should make per-chunk dw updates smaller.
  4. **Add weight decay or gradient clipping on ttt_proj/ttt_conv** during
     training (would have prevented the blow-up in the first place).

Phase A still proceeds: it tells us at what relative magnitude a *random*
perturbation breaks the base in the same way the trained TTT does. That number
calibrates how much we need to shrink dw to give the model any chance of
producing coherent text.

## Original hypothesis (pre-measurement, kept for context)

**The frozen base is the deepest cause, not the small training scale.**

Mechanically: condition B's predictions are gibberish (`"由谁参与\n\n品牌重塑 由谁参与..."`, single-token loops, garbled subwords). They are not "wrong answers in fluent prose" — they are *the model breaking*. After TTT modifies `down_proj.weight` by a few percent (~5-10% relative), the **frozen base has no idea how to read its own perturbed weight**.

This is different from "TTT didn't learn memory." It might have learned *something* — but the rest of the model can't decode what's stored. With a frozen base, every layer downstream of a TTT layer sees a perturbed activation distribution it was never trained to handle. The model isn't *interpreting* TTT's output, it's *crashing* on it.

The paper avoids this by training `base + TTT jointly` over 21B+ tokens, so the base learns to be robust to (and to *use*) the fast-weight modifications its TTT layers produce. We skipped that step. Predictably, things broke.

**Test the hypothesis:** if we add random Gaussian noise of similar magnitude (~5% relative) to `down_proj.weight` of a frozen Qwen3-8B and ask it to generate, the output will look just like cond B. If yes → confirms the failure mode is "frozen base ≠ perturbation-robust", not "TTT failed to encode information." Easy ablation, ~5 min.

## What's *less* likely the dominant cause (but still real)

| Cause | Why probably not the main one |
|---|---|
| Training too short (2000 steps) | `ttt_conv` already moved off init (norm 0.111 layer 0). Sanity 4 confirms `past_w` is input-dependent (diff 0.0015). The mechanism is encoding *something*. The problem is downstream-readability, not encoding-amount. |
| Wrong training data (English short Q&A) | Same reason — data shape affects *what's encoded*, but our predictions aren't "wrong answers", they're broken text. Suggests downstream collapse, not retrieval failure. |
| `ttt_chunk` mismatch with paper | We picked 64 specifically to fire 1-2 updates per conversation. That part worked (verified by sanity). |
| `ttt_proj` random init at norm 81.5 | At that magnitude, `dw` could be ~10x larger than ideal. Possible contributor — making the perturbation *too big* for a frozen base to absorb. Worth testing init scale. |

## Concrete iteration-2 plan

Goal: test the "frozen base is the dominant issue" hypothesis and, if confirmed, fix it. Two phases.

### Phase A — cheap ablation (~$1, 30 min)

Run a fifth condition: **A'** = vanilla Qwen3-8B + N(0, 0.05·||W||) Gaussian noise added to `down_proj.weight` of the same 6 layers (no TTT, no training, just random noise of similar magnitude). Drop conversation from context. Ask the same probes.

If A' produces the same gibberish as B → **frozen base is the dominant failure mode** (TTT could be encoding fine; downstream model can't read perturbed weights).

If A' is coherent (just wrong) → **TTT specifically is the issue**; the trained ttt_proj/ttt_conv produce a *worse* signal than random noise, suggesting they actively learned to confuse the model.

This is a 5-line modification to `run_experiment.py`. Cheap and decisive.

### Phase B — joint train base + TTT (LoRA on base) (~$10, 1.5 hr training + 4 hr conditions)

If Phase A confirms frozen-base hypothesis:

1. **Add LoRA adapters to base** (rank 16, all attention + MLP linears). ~90M extra trainable params.
2. **Joint train base-LoRA + TTT** for 5000 steps. Both move together — base learns to handle TTT's modifications.
3. **Use long-context data**: switch from `HuggingFaceH4/no_robots` (short instruction Q&A) to something like `togethercomputer/RedPajama-Data-1T-Sample` or similar long-form text. Long sequences naturally reward memory across chunks.
4. **seq_len=4096** instead of 512. With chunk=64, that's 64 chunks per forward → 63 TTT updates per backward, vs 7 we had. Much more gradient signal per step.
5. **Keep ttt_chunk=64, lr=1e-5** (we know they're stable).
6. Memory budget on A100-40G: 8.2B base bf16 (16GB) + 90M LoRA fp32 grads (0.4GB) + 100M TTT fp32 (0.4GB) + AdamW state ~5GB + activations w/ grad-ckpt ~2GB + KV cache ~3GB = ~27GB. Fits.

Expected per-step time at seq 4096 vs current 512: ~5-8x slower. 5000 steps × 5x slower than current 0.32s/step = 8000s = 2.2 hr. ~$5 training.

Then re-run all 4 conditions. ~$13.

Total iteration-2 cost: ~$18-20. Within remaining $16 of free credit (close — may need $5 paid spillover).

### Other things worth changing in iteration-2

- **Reduce `ttt_proj` init scale.** Currently random init at norm 81.5 (default Linear init). A smaller init (norm ~10-20) would make `dw` smaller initially and let training scale it up if useful. Cheap change in `model_utils.py`.
- **Train base+TTT joint with `ttt_target=input_embed`** (one of the two valid settings). README says this is for from-scratch; may give different gradient flow than `hidden_states`.
- **Add 5th condition A' permanently** (random-noise control). It pins down whether any future "B works!" is real signal vs just "noise happens to align."

## What I won't try (and why)

- **Paper-scale 5000 steps × seq 65536 × batch 64 joint training**: $1k+, many H100-days. Out of stated budget.
- **Switching to a different mechanism (LoRA-only, RAG, etc.)**: user explicitly asked for literal validation of TTT, not "make memory work somehow."
- **Hand-engineering a dataset of fact+probe pairs for training**: that's training-on-test-distribution, defeats the experimental question.

## How to read this file in iteration-2

The TL;DR for the next session: run Phase A first (5 minutes, cheap, decisive). The result of Phase A determines whether Phase B is worth $20.
