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

## D-002 — `ttt_chunk = 1024` for both training and eval (deviates from paper's 4096)

**Decided:** 2026-04-28T08:40:00Z
**Status:** active

### Context

Paper sets `ttt_chunk = 4096`. The TTT update only fires when `present_h.shape[1] >= ttt_chunk`. Our conversational samples are 200-1500 tokens. At paper's setting, **TTT update never fires on a single conversation**, defeating the experiment.

Options:
1. Keep `ttt_chunk=4096`, pad/repeat conversations to ≥4096 tokens at inference time. (Mismatches training distribution; padding is artificial.)
2. Lower `ttt_chunk` to match expected conversation length, train and eval at the same setting.
3. Lower `ttt_chunk` to a small value (e.g. 256) so multiple updates fire per conversation.

### Decision

`ttt_chunk = 1024` for both training and eval. Conversations <1024 tokens get one update at end-of-input; longer conversations get one update per 1024-token block.

### Why

We're training from scratch (B-mini), so the chunk size is a free parameter. 1024 covers most conversation lengths in a single chunk while still being short enough that long dialogues see multiple updates. Matching train and eval is critical — paper notes that TTT-train at one chunk size and TTT-infer at a different chunk size degrades.

### Risk

If the trained TTT mechanism is sensitive to chunk size, our results aren't directly comparable to paper-scale training at 4096. We document this and treat the absolute numbers as our setup's, not the paper's.

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
