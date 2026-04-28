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
