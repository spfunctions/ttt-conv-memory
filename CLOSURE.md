# CLOSURE — repo abandoned 2026-04-28

This repo answered the literal question Patrick asked
("does TTT fast-weight update substitute for context window as conversational
memory mechanism?") with **NEGATIVE** under realistic single-GPU budget,
across two iterations that together cost ~$17 of free Modal credit.

The deciding piece of evidence is in `FINDINGS.md`:

> Even at α=0.1 inference scaling — where the model is mostly fluent and the
> TTT perturbation is small (layer-30 rel_dw 0.037) — the gold answer appears
> in the prediction 0/259 times. Cond A (full context) gets 92.2 %; cond C
> (vanilla, no memory) gets 1.6 % by chance. Past_w simply contains no
> conversation memory at any operating point we can produce with a frozen
> base trained on short Q&A.

Flipping this would require joint base+TTT training (~$20-30, beyond
remaining free credit) on a long-context corpus, and even that is not
guaranteed to produce a positive result — it would just give the loss
landscape a chance to push fast weights toward "memory" instead of "noise".

## What lives on after this repo

These are the reusable bits worth keeping in mind for future work:

- `model_utils.py:kv_stripped_clone` — TTTDynamicCache primitive that drops
  KV but preserves `past_w` (turn fast-weight memory on/off independently of
  attention KV).
- `model_utils.py:zero_ttt_params` / `snapshot_ttt_params` /
  `restore_ttt_params` — toggle TTT effect by parameter zeroing because
  `config.ttt_mode = False` at runtime is a no-op.
- `model_utils.py:measure_relative_dw` — output-diff method for measuring
  ||dw||_F / ||W||_F on real conversations via MLP forward hook (handles
  In-Place-TTT's `(hidden_states, present_w)` tuple return).
- `model_utils.py:apply_gaussian_noise_to_down_proj` — primitive for
  random-noise control conditions.
- `modal_app.py` patterns — debian_slim Python 3.11 + cu128 torch + sdpa
  (no flash-attn 30-min compile), `add_local_dir` last, `vol.commit()` after
  every result write.

## What we did NOT test (still open as research questions)

- Joint base+TTT training at any scale (paper does this at ~21B tokens; we
  did not even try a small joint run).
- Long-context training data (we used short Q&A — no_robots).
- Other test-time-training implementations (LoRA-online-update, etc.).
- Larger conversational corpora in Chinese (we trained on English, evaluated
  in Chinese).

## Total cost

| iteration | description | $ |
|---|---|---|
| iter-1 | scaffolding, train, sanity, 4 conditions, eval | 13.67 |
| iter-2 step 1 | dw measurement | 0.01 |
| iter-2 step 2 | Phase A noise sweep | 1.10 |
| iter-2 step 3 | inference-time TTT scaling | 1.50 |
| **total** | | **~16.30** |

Within the $30/mo Modal free credit; no paid spillover.

## Final commit + push

This file is the last commit. After this, the repo is left as-is. If
anyone (incl. future me) ever wants to revisit, read in this order:

1. `FINDINGS.md` (top section is the iteration-2 final verdict)
2. `RESULTS.md` (iteration-1 numbers)
3. `STATE.md` (chronological log)
4. `DECISIONS.md` (research-level forks)
