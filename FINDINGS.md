# FINDINGS — what went wrong, and what to try next

Iteration 1 (this commit) ended with `EM(B)/EM(A) = 0/0.929 = 0.000`. Below: what
I think actually broke, ranked by how confident I am, and a concrete iteration-2
plan with cost estimates.

## What I'm now most confident broke the experiment

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
