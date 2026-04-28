# DECISIONS — every fork in the road

Each decision is timestamped, names the alternatives considered, and explains why we chose what we chose. **Decisions, not opinions.** When something turns out wrong, append a follow-up entry rather than rewriting history.

---

## D-001 — Use B-mini self-trained checkpoint (not paper-scale, not LoRA fallback)

**Decided:** 2026-04-28T08:35:00Z
**Status:** active

### Context

The In-Place-TTT repo does not ship a pretrained checkpoint. Without trained `ttt_proj` and `ttt_conv` parameters, the TTT mechanism is non-functional (zero-init `ttt_conv` produces zero updates). Three real options:

| Option | Cost | Faithfulness | Time |
|---|---|---|---|
| A — paper-scale continual-pretrain | $1k+ | ★★★★★ | days |
| B-mini — self-train at minimum scale | $10-15 | ★★★ | 6-8h |
| C — PEFT LoRA online update fallback | $5-10 | ★★ (different vehicle) | 4-6h |
| D — wait for upstream | $0 | ★★★★★ | unknown |

### Decision

**B-mini.** Train just enough that `ttt_proj` and `ttt_conv` are out of init.

### Why

User's literal goal is "validate whether TTT fast-weight updates can replace context window for conversational memory" using the In-Place TTT vehicle. C changes the vehicle. A is over-budget for a gating experiment. D is uncontrolled timing.

B-mini is the minimum thing that preserves the original vehicle. We accept that absolute numbers will be lower than paper-scale would give; we care about the **direction** (does TTT memory exist at all? does it scale with training?), not the absolute peak.

We also opened upstream issue #3 in parallel — if the authors release a checkpoint within the experiment window, we re-run with theirs as a free upgrade.

---

## D-002 — `ttt_chunk = 64` for both training and eval (deviates from paper's 4096)

**Decided:** 2026-04-28T09:30:00Z (revised from initial 1024)
**Status:** active

### Context

Paper sets `ttt_chunk = 4096`. The TTT update only fires when `present_h.shape[1] >= ttt_chunk`. Empirically measured token counts of our generated benchmark:
- L1/L3 conversation: mean 140, min 123, max 158 tokens
- L2 conversation (same as L1): mean 140
- Probe full prompt (system + question + "回答："): ≈40-60 tokens
- Distractor (L3): ≈2000-4000 tokens

Initially set `ttt_chunk = 1024` (compromise between paper's 4096 and conversation length). Then realized: with 140-token conversations and 1024-chunk, the TTT update **never fires**. The whole experiment would measure noise.

### Decision

`ttt_chunk = 64` for both training and eval.

### Why

- **All 300 conversations trigger ≥1 update** (123 > 64): the TTT mechanism actually engages. Most conversations get 1-2 chunk updates per forward pass.
- **All 2620 probes are too short to trigger updates** (probe full prompt ≈ 40-60 tokens, < 64). Probes don't pollute the conversation-modified fast weights — clean A/B separation.
- **Distractor adds many updates** (2000-4000 tokens / 64 ≈ 30-60 updates). Tests the distractor effect at scale.
- **Training at seq_len=1024 with chunk=64 = 16 chunks per sample = 15 updates per backward**. Plenty of update signal during training.

### Risk

- This is much smaller than paper's 4096. The TTT layers may behave qualitatively differently at this chunk size — for example, the per-update gradient is smaller (less context per chunk), and accumulation across many tiny chunks may dominate.
- Mitigation: matched chunk size between train and eval (`ttt_chunk=64` everywhere) to avoid distribution shift.
- If the experiment shows weak signal, the natural next iteration is to extend conversation length in benchmark_v2 (multi-paragraph, multi-topic dialogues) and increase chunk back toward 256-512.

### Numbers we computed

```
L1/L3 conv tokens: mean=140, min=123, max=158
L1/L3 probe tokens (question only): mean=6, min=3, max=10
L2 probe tokens: mean=7, min=5, max=10

at ttt_chunk=64:
  conversations triggering update: 300/300
  L1/L3 probe questions triggering: 0/2192
  L2 probe questions triggering: 0/425
```

---

## D-003 — Template-based dataset generation (no LLM API)

**Decided:** 2026-04-28T08:42:00Z
**Status:** active

### Context

Brief allows either LLM API (GPT-4o / Claude) or template-based dataset generation. Templates are more reproducible; LLM-generated has more naturalness.

### Decision

Templates with a fact pool of ~200 names, ~200 numbers, ~100 places, ~50 projects, ~50 preferences, ~20 conversation skeletons.

### Why

- Reproducibility (anyone with `build_benchmark.py` and the fact pool gets the same dataset)
- No external API cost or rate-limit risk
- Tighter control over fact distribution and probe ambiguity
- Probes have clean gold answers because they were generated alongside the conversation, not extracted

### Risk

Synthetic conversations may not look like real human conversations — a model trained on natural dialogue might handle them differently. We accept this; the experiment is about the *mechanism*, not natural-conversation ecological validity. If condition A scores poorly, that's a data-quality flag.

---

## D-004 — Train only `ttt_proj + ttt_conv`, freeze base

**Decided:** 2026-04-28T08:50:00Z
**Status:** active

### Context

Paper trains everything jointly (full continual-pretrain). For B-mini we have constrained budget.

### Decision

Freeze all original Qwen3-8B parameters. Train only the new TTT-specific parameters: `ttt_proj` (Linear hidden→hidden) and `ttt_conv` (depth-wise Conv1d). For Qwen3-8B with 7 TTT layers this is ~7 × (hidden² + hidden×kernel) ≈ ~150M params (vs 8.2B base) — ~50× param-efficient.

### Why

- Preserves Qwen3-8B's world knowledge intact (so condition A — context baseline — gives clean upper bound)
- Drastically reduces VRAM for backward pass (only ~150M params accumulate gradients)
- Matches the spirit of "test-time training" — the *fast* weights are what move, base stays put

### Risk

Paper trains base + TTT layers jointly, possibly because TTT layers need to coordinate with base updates. Our frozen-base setup may give weaker TTT params than joint training would. This is acceptable for B-mini; we're checking mechanism existence, not chasing peak performance.

---

## D-005 — Use Modal serverless (not vast.ai or other persistent box)

**Decided:** 2026-04-28T08:32:00Z
**Status:** active

### Context

GPU options:
- vast.ai: $0.61/hr A100, but $0 balance (needs deposit)
- Modal: $2.10/hr A100 with $30/mo free credit
- Lambda Labs / Runpod: similar to vast, needs deposit

### Decision

Modal.

### Why

- $30 free credit covers full experiment budget (12-15 GPU-hours expected)
- Serverless: no idle billing during dev iteration
- Already used elsewhere in the SF stack — incremental learning curve
- No deposit / wire transfer friction

### Trade-off

Cold start (~30-60s) per Modal Function call. Mitigated by batching: each condition runs all 300 samples in one Function invocation, not 300 invocations.

---

## D-006 — Greedy decoding, bf16, max_new_tokens=64

**Decided:** 2026-04-28T08:55:00Z
**Status:** active

### Context

Eval generation hyperparameters.

### Decision

- `do_sample=False` (greedy)
- `torch_dtype=bfloat16`
- `max_new_tokens=64`
- `temperature=None`, `top_p=None`

### Why

- Greedy → deterministic, reproducible
- bf16 → matches training, fits in 40G with KV cache for 4096+ tokens
- 64 tokens → all factual answers fit (most are <10 tokens; 64 absorbs verbose phrasings without inflating runtime)

---

---

## D-007 — SDPA attention, no flash-attn (skip 30+ minute compile)

**Decided:** 2026-04-28T09:35:00Z
**Status:** active

### Context

Upstream pins `flash-attn==2.8.3`. On Modal's pip mirror with cu128 wheels there's no matching prebuilt wheel, so install falls back to `setup.py bdist_wheel` — observed ~5+ minutes into the build with no end in sight (typical flash-attn source compile is 20-40 minutes on a 16-core machine).

### Decision

Skip flash-attn entirely. Use PyTorch SDPA (`attn_implementation="sdpa"`).

### Why

- The In-Place TTT modeling code dispatches attention through `config._attn_implementation` — SDPA is a fully supported branch (`Qwen3PreTrainedModel._supports_sdpa = True`).
- Upstream's `from transformers.modeling_flash_attention_utils import FlashAttentionKwargs` imports a typed-dict from transformers, not from flash_attn — so this import works without the package.
- SDPA on A100 is within ~1.2-1.5x of flash-attn for our seq lengths (≤4096). For our 12-15 GPU-hour total budget the difference is ~3-4 GPU-hours, but skipping flash-attn build saves ~30+ minutes of every cold image build.
- We're not using upstream's training framework (VeOmni); we're using HF transformers directly. VeOmni might have hard flash-attn deps but we don't touch it.

### Risk

- A code path inside upstream's `inference_model/` that we haven't read could `import flash_attn` directly. If so, `import inference_model` will fail at import time. Workaround: install flash-attn lazily on first crash.
- SDPA may have small numerical differences vs flash-attn at large scale. Not relevant for our experiment scope.

---

## (template — copy this for new decisions)

## D-XXX — short title

**Decided:** YYYY-MM-DDTHH:MM:SSZ
**Status:** active | superseded-by-D-YYY | reverted

### Context

What forced the decision.

### Decision

What we chose.

### Why

The reasoning, including alternatives we rejected.

### Risk

What could go wrong with this choice and how we'd notice.
