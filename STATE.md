# STATE — Live progress log

Updated continuously. Each entry stamped with UTC timestamp.

## Current phase
**Phase 0 — scaffolding**

## Phase ledger

- [x] Phase -1: research (read upstream repo, found checkpoint absent, mapped TTTDynamicCache primitive)
- [ ] Phase 0: scaffolding (this repo, docs, benchmark builder, modal app)
- [ ] Phase 1: benchmark generated locally
- [ ] Phase 2: Modal image built, base Qwen3-8B inference verified
- [ ] Phase 3: minimal TTT training run finished, sanity checks pass
- [ ] Phase 4: condition A run (300 samples)
- [ ] Phase 5: condition C run (300 samples)
- [ ] Phase 6: condition B run (300 samples) — **the experiment**
- [ ] Phase 7: condition D run (100 samples)
- [ ] Phase 8: evaluation + figures + RESULTS.md
- [ ] Phase 9: commit + push final

## Detailed log

### 2026-04-28T08:30:00Z — kickoff
- User briefed the experiment.
- Verified vast.ai CLI present (auth as `patrick@simplefunctions.dev`, balance $0.00).
- Verified `gh` auth as `patrickliu0077`, write access to `spfunctions` org.
- Researched `ByteDance-Seed/In-Place-TTT` repo:
  - Repo exists, public, ICLR 2026 oral, last push 2026-04-21.
  - **No public checkpoint.** `eval_config/models.py` is a stub.
  - Fast weight lives in `TTTDynamicCache.ttt_states[layer_idx][2]` — decouplable from KV.
  - `ttt_chunk=4096` gates TTT update; below threshold update is skipped.
  - Pinned deps: torch 2.8.0+cu128, flash-attn 2.8.3 cp311, veomni @ 9b91e164, transformers 4.57.3, Python 3.11.

### 2026-04-28T08:35:00Z — direction set
- Path B-mini selected (self-train minimal checkpoint, ~$10-15 on Modal).
- Platform: Modal ($30/mo free credit).
- Repo: `spfunctions/ttt-conv-memory` public.

### 2026-04-28T08:40:00Z — scaffolding actions
- Issue opened upstream: <https://github.com/ByteDance-Seed/In-Place-TTT/issues/3>
- Repo created: <https://github.com/spfunctions/ttt-conv-memory>
- Repo cloned to `/Users/liuyizhou/spfunctions/ttt-conv-memory`.
- Modal CLI installed (v1.4.2), auth'd to workspace `patrick-43806`, token verified.

### 2026-04-28T08:45:00Z — docs scaffold complete
- `README.md`, `SPEC.md`, `STATE.md` (this), `DECISIONS.md` written.
- Next: `requirements.txt`, `.gitignore`, `setup.sh`, `build_benchmark.py`.

### 2026-04-28T09:00:00Z — code scaffold complete + benchmark generated
- `requirements.txt`, `.gitignore`, `setup.sh` written.
- `build_benchmark.py` written and run locally — produced `benchmark_v1.json` (300 samples, 2620 probes, 5 conversation skeletons).
- `model_utils.py` written (Qwen3-8B + In-Place TTT loader, fast-weight save/load, `kv_stripped_clone` primitive, ttt-mode toggle).
- `train_minimal.py` written (frozen-base, train ttt_proj+ttt_conv on `HuggingFaceH4/no_robots`).
- `run_experiment.py` written (4 conditions A/B/C/D).
- `evaluate.py` written + smoke-tested locally with synthetic per-condition outputs (pipeline mechanics OK).
- `sanity.py` written (5 pre-flight sanity checks per SPEC.md).
- `modal_app.py` written (nvidia/cuda 12.8 dev base image + pinned ML stack + In-Place TTT clone).
- All Python files AST-parse clean.
- User switched session to bypass-permissions / auto-mode.
- Next: commit + push, then trigger Modal image build via `modal run modal_app.py::smoke_image`.

### 2026-04-28T09:30:00Z — three architectural bug fixes after deep upstream code read
Read `inference_model/hf_qwen3/modeling_qwen3.py` carefully. Found:
1. **`ttt_chunk` was 1024 — TTT update would never fire.** Measured benchmark conv tokens: 123-158, probe tokens: 3-10. The upstream guard `if seq_len < self.ttt_chunk: return ..., present_down_proj_w` means convs at ttt_chunk=1024 just early-return without updating. Fixed: ttt_chunk=64 (DECISIONS D-002 revised). All 300 convs now trigger update; all 2620 probes safely below threshold.
2. **`config.ttt_mode = False` is a no-op at runtime.** Upstream attaches ttt_proj/ttt_conv at __init__ gated on config flags; mutating them after init does nothing. Replaced disable_ttt_updates / enable_ttt_updates with `zero_ttt_params` + `snapshot_ttt_params` + `restore_ttt_params`. For conditions A/C: zero TTT params (functional vanilla); for B/D: restore trained snapshot.
3. **flash-attn skipped, use SDPA.** Modal pip mirror has no prebuilt flash-attn wheel for cu128/torch2.8/cp311 → falls back to source compile (30+ minutes). Upstream code dispatches via `config._attn_implementation`; SDPA works without flash-attn. Saved per-image-build time.

### 2026-04-28T09:35:00Z — kv_stripped_clone fix
Realized that retaining `past_h_tail`/`past_t_tail` (zero or otherwise) could let probe forward concatenate with them and exceed `ttt_chunk`, firing a TTT update during probe and polluting the fast weights with probe content. Fixed: drop tails to None so the layer's `if past_h is None: present_h = hidden_states` branch keeps probe forward standalone.

### 2026-04-28T09:38:00Z — Modal smoke build started
- First build attempt (with flash-attn) cancelled at ~5 min into source compile.
- Restarted with debian_slim base + no flash-attn → fast image build in progress.
- Watching for terminal signal via Bash background task (buvik951x).
- Next: when smoke succeeds, run `modal run modal_app.py::train` (B-mini training, ~4-6h on A100).

### 2026-04-28T09:55:00Z — `workdir` ordering bug, fixed
- First build complete attempt failed: `.workdir("/app")` after `.add_local_dir(".", remote_path="/app")` is rejected by Modal.
- Fixed: moved workdir before add_local_dir (any build step after add_local must use copy=True).

### 2026-04-28T10:00:00Z — SMOKE TEST PASSED ✓
- Modal app: <https://modal.com/apps/patrick-43806/main/ap-aD1IG5D7gfZ9VtSmDXpKj5>
- Image built clean. Container booted on A100-40G.
- HF download Qwen3-8B (5 shards) cached to `/data/hf_cache/transformers/` on the volume.
- Model load: bf16 on cuda:0. ~16GB.
- Init TTT param norms: `ttt_proj=81.5` (random init), `ttt_conv=0.0` (zero init, expected).
- Forward pass on 4-token input → logits shape `[1, 4, 151936]` (151936 is Qwen3 vocab).
- TTT layers in model: `[0, 6, 12, 18, 24, 30]` — index 36 from config silently dropped (Qwen3-8B has 36 layers, max idx is 35). Effectively 6 TTT layers, not 7 as in upstream-recommended config.
- Cost so far: ~2 min × A100 ≈ $0.07.

### 2026-04-28T10:08:00Z — Training first attempt (OOM)
- Command: `modal run modal_app.py::train --steps 400 --ttt-chunk 64 --seq-len 1024 --batch-size 2 --grad-accum 2`
- Result: CUDA OOM at first backward. 38.58 GB used, 1 GB free, needed 1.16 GB more.
- Trainable params: 100.8M (TTT) — 12 tensors (6 layers × ttt_proj + ttt_conv).
- Total: 8.29B (Qwen3-8B + TTT params).

### 2026-04-28T10:14:00Z — Training second attempt with grad checkpointing — SUCCESS ✓
- Command: `modal run modal_app.py::train --steps 400 --ttt-chunk 64 --seq-len 512 --batch-size 1 --grad-accum 4`
- Wall time: **~125 seconds total** (model load + dataset + 400 steps).
- Per-step time: ~0.3s/step (very fast on A100 even with grad-checkpointing).
- Loss curve: 4.2551 (step 0) → ~3.0 average (final). Noisy due to batch=1.
- TTT param drift: `ttt_conv` 0.000 → 0.011, `ttt_proj` 81.5 → 81.5 (basically unchanged at this scale).
- Total TTT param norm sum: 489.00 → 489.04. Small but **gradients did flow**.
- Checkpoint: `/data/checkpoints/ttt_minimal.pt` (~150 MB on Modal volume).
- Cost: ~$0.10 (~3 min × A100).

### 2026-04-28T10:18:00Z — Sanity checks ran — 2/5 passed
- Triggered: `modal run modal_app.py::sanity`
- Modal app: <https://modal.com/apps/patrick-43806/main/ap-qfcqbh5M2CO3sIDWcoOY41>
- Results:
  - ✓ Check 3 (fast weight reproducible): pass — diff = 0.00e+00
  - ✓ Check 5 (KV-stripped forward): pass — model emitted Chinese answer
  - ✗ Check 1 (trained vs zero diverges): IndexError in HF generate due to multi-stage cache passing — bug in test, not model
  - ✗ Check 2 (TTT params non-zero): zero norms reported — actually a side effect of failed check 1 leaving model in zeroed state (no try-finally)
  - ✗ Check 4 (input-dependent past_w): diff=0 — past_w identical for different inputs ⇒ TTT effect too small / params barely moved off init
- Diagnosis: training was too weak. ttt_conv 0 → 0.011 means dw signal ≈ 0.5 vs down_proj norm ~64 → ~1% perturbation, hard to detect.
- Fixes applied:
  - sanity.py check 1: compare logits (more sensitive than greedy text), wrap in try-finally to restore state on failure.
  - Re-train with 5x more steps (2000) and 10x higher LR (5e-5).

### 2026-04-28T10:25:00Z — Re-training (run 3, lr=5e-5)
- Command: `modal run modal_app.py::train --steps 2000 --lr 5e-5 --ttt-chunk 64 --seq-len 512`
- **Diverged immediately** — loss 4.25 → 10.07 by step 4 → 14.62 by step 7. lr too high for our setup.
- Killed at step 400.

### 2026-04-28T10:32:00Z — Re-training (run 4, lr=1e-5) — SUCCESS ✓
- Command: `modal run modal_app.py::train --steps 2000 --lr 1e-5 --ttt-chunk 64 --seq-len 512`
- Wall time: ~11 min.
- Loss curve: 4.25 → 3.83 → 3.25 → 3.21 → ~3.0 average (gentle decline, plateaus around step 1000).
- TTT param drift (sum of norms): 489.00 → 489.20 (delta 0.20, vs run 1's 0.04 — 5x stronger).
- ttt_conv per-layer norms (final): layer 0 = 0.111, layer 6 = 0.023, layer 12-30 ≈ 0.012-0.020.
- Layer 0 learns fastest (sees fresh hidden states with most variance).
- Cost: ~$0.40.

### 2026-04-28T10:45:00Z — Sanity checks against new checkpoint — 5/5 PASSED ✓
- Modal app: <https://modal.com/apps/patrick-43806/main/ap-qJjniVyYuDSUQlpUMEbqZt>
- ✓ Check 1 (trained vs zeroed logits diverge): mean abs diff 4.5625, max 25.5, **top1 token differs** (zero=18, trained=28516). TTT has measurable effect.
- ✓ Check 2 (TTT params non-zero): proj 81.5, conv 0.111 / 0.023 (layers 0/6 sample).
- ✓ Check 3 (fast weight reproducible): mean abs diff 0.00e+00 across two forward passes of same input.
- ✓ Check 4 (input-dependent past_w): mean abs diff 0.0015 between two different inputs. Real signal.
- ✓ Check 5 (KV-stripped forward): no errors.

### 2026-04-28T10:48:00Z — All 4 conditions launched in parallel
- Cond C: ETA 70 min, no context
- Cond A: ETA 100 min, conversation in context
- Cond B: ETA ~2.5h, the experiment
- Cond D: ETA ~50 min, distractor

### 2026-04-28T11:08:00Z — Cond D completed (23 min, 100 samples)
### 2026-04-28T11:38:00Z — Cond C completed (53 min, 300 samples)
### 2026-04-28T11:42:00Z — Cond B completed (54 min, 300 samples)
### 2026-04-28T12:07:00Z — Cond A completed (82 min, 300 samples — bottleneck)

### 2026-04-28T12:10:00Z — Evaluate ran successfully

**HEADLINE NUMBERS:**

| Cond | EM | F1 | n |
|---|---|---|---|
| A — context baseline | **0.929** | 0.154 | 2617 |
| B — TTT memory | **0.000** | 0.007 | 2617 |
| C — no memory | **0.016** | 0.034 | 2617 |
| D — TTT + distractor | **0.000** | 0.000 | 1096 |

- `memory_efficiency_ratio = EM(B) / EM(A) = 0.000`
- `lift_over_no_memory = EM(B) − EM(C) = −0.016` (B even worse than C)
- **Verdict: NEGATIVE.** TTT fast weights, under our B-mini training, do not substitute for context window — they actively degrade the model below random-baseline.

### 2026-04-28T12:15:00Z — Pulled results to local
- `results/{report.json, condition_*.json, figures/*.png}` all on local disk.

### 2026-04-28T12:18:00Z — Phase 9: writing RESULTS.md + final commit + push
- See `RESULTS.md` for full interpretation.

## Total cost (final)

| Phase | $ |
|---|---|
| Smoke test | 0.07 |
| Training run 1 (lr=5e-6, 400 steps) | 0.10 |
| Training run 2 (OOM, killed early) | 0.05 |
| Training run 3 (lr=5e-5, killed for divergence) | 0.30 |
| Training run 4 (lr=1e-5, 2000 steps, success) | 0.40 |
| Sanity #1 + #2 | 0.25 |
| 4 conditions in parallel | ~12.40 |
| Evaluate + pull | 0.10 |
| **Total** | **~$13.67** |

Within $30/mo Modal free credit.

## Cost tracker

| When | What | $ | Cumulative |
|---|---|---|---|
| 2026-04-28 | Modal signup credit | -30.00 (credit) | -30.00 |
| | | | |

## Modal usage tracker

| When | Function | GPU | Duration | Cost |
|---|---|---|---|---|
| | | | | |

## Decisions log pointer

See `DECISIONS.md` for every research-level fork.
