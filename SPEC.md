# SPEC — TTT Conversation Memory Experiment

Last updated: 2026-04-28
Authoritative spec. Updated as design choices crystallize. Original user brief preserved at the end.

## Research question

Can test-time training (TTT) fast-weight updates retain factual content from a conversation well enough to substitute for the context window? Concretely, after a model ingests a dialogue with TTT enabled, and the dialogue is then **dropped from context**, how many embedded facts can the model still answer correctly using only its updated fast weights?

The verdict bar from the original brief:

| Memory efficiency ratio = EM(B) / EM(A) | Interpretation |
|---|---|
| > 0.7 | "This path is viable in principle" |
| 0.3–0.7 | "Mechanism partly works; do ablations" |
| < 0.3 | "Fast-weight memory is not a context-window substitute under this setup" |

## Vehicle

[ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT) on Qwen3-8B.

Key code-level facts (from a 2026-04-28 reading of the upstream repo):

1. **Fast-weight state** lives in `TTTDynamicCache` (subclass of HF `DynamicCache`). It carries an extra parallel array `self.ttt_states[layer_idx] = (past_h, past_t, past_w)` alongside the KV slots. **The fast weight `past_w` is in the cache, not in the model parameters.**

2. **TTT update gate** per-layer: fires only if `config.ttt_mode == True` AND `layer_idx in config.ttt_layers` AND `present_h.shape[1] >= config.ttt_chunk` (default 4096). Below the chunk threshold, `present_down_proj_w` is read but **not** updated.

3. **Per-layer learnable additions** (zero-init / default-init at scratch): `ttt_proj` (Linear, hidden→hidden, no bias) and `ttt_conv` (depth-wise Conv1d, kernel=5). These need training to be useful — at init `ttt_conv` is zero, so its contribution is zero.

4. **Recommended TTT layers**: `[0, 6, 12, 18, 24, 30, 36]` (every 6th layer of a 36-layer Qwen3-8B = 7 layers).

5. **`ttt_target ∈ {hidden_states, input_embed}`** — README says `input_embed` for from-scratch and `hidden_states` for continual. We use `hidden_states` (default).

6. **Disabling TTT at inference**: set `config.ttt_mode = False` before generate, OR pass an empty `config.ttt_layers`. There is no CLI/eval-script flag.

7. **No public pretrained checkpoint exists.** HF org `ByteDance-Seed` has 55 models, none with `ttt`/`place`/`test` in name. `eval_config/models.py` is a stub.

## Architectural assumption being tested

Because `(past_h, past_t, past_w)` lives in `TTTDynamicCache`, we can construct a cache where:
- The KV slots `(key, value)` are zeroed/dropped → the model sees no past tokens in its attention
- The TTT state `ttt_states[layer_idx][2] = past_w` is preserved → the model's MLP down-proj has the conversation-modified fast weight

This is the technical primitive that makes the experiment possible. **If this assumption breaks** (e.g. fast weight is unusable without its co-stored hidden-state context, or the inference forward path requires both), the experimental design must change. We verify this assumption explicitly in a sanity check before running the full 300-sample suite.

## Dataset (`benchmark_v1.json`)

300 samples, 100 per layer.

### Layer 1 — direct fact recall (100 samples)

Each sample = a multi-turn natural conversation with 5–10 facts naturally embedded. Six fact categories:

| # | Category | Example |
|---|---|---|
| 1 | Person binding | `"我叫张伟，我同事叫李明"` |
| 2 | Numeric binding | `"我的工号是7742"`, `"预算是340万"` |
| 3 | Temporal binding | `"会议在周三下午两点"` |
| 4 | Relational binding | `"李明是市场部负责人，向王总汇报"` |
| 5 | Preference binding | `"我不吃辣，偏好清淡"` |
| 6 | Spatial binding | `"办公室在3号楼12层"` |

Each fact has one paired probe (question + gold answer). Probes are factual, single-fact, and have unambiguous gold answers.

### Layer 2 — reasoning recall (100 samples)

Built on layer 1 conversations. Probes require composing two facts (e.g. conversation has "Zhang oversees project A" and "project A deadline is June 15"; probe asks "when is Zhang's project deadline?"). 5 such composed probes per sample.

### Layer 3 — distractor robustness (100 samples)

Same conversations as layer 1, then **2000–4000 tokens of unrelated text** (technical articles, news) appended. Same probes as layer 1. Tests whether subsequent unrelated input overwrites the fact-encoding fast weights.

### Generation strategy

**Template-based with a fact pool.** No external LLM API dependency. We define ~200 names, ~200 numbers, ~100 places/projects, etc., and procedurally assemble dialogues from a small set of conversation skeletons. This is more controllable and reproducible than LLM-generated content.

### Sample format

```json
{
  "sample_id": "L1-0042",
  "level": 1,
  "conversation": "...",
  "facts": [
    {"fact_id": "F1", "fact_text": "工号是7742", "category": "numeric"},
    ...
  ],
  "probes": [
    {
      "probe_id": "P1",
      "question": "我的工号是多少？",
      "gold_answer": "7742",
      "required_facts": ["F1"]
    },
    ...
  ],
  "distractor": null  // or string, only for level 3
}
```

## Experimental conditions

All four conditions run on the same trained TTT-enabled Qwen3-8B checkpoint, greedy decoding, bf16, `max_new_tokens=64`.

### Condition A — context baseline (TTT off, conversation in context)

```
input  = system_prompt + conversation + probe
output = model.generate(input, ttt_off=True)
```

Standard inference. Conversation visible in the attention context. **TTT updates explicitly disabled** so this is a vanilla Qwen3-8B inference. Establishes the upper bound — no method should beat this.

### Condition B — TTT memory (the experiment)

```
1. Reset model: load fresh TTT checkpoint, instantiate fresh TTTDynamicCache
2. Forward conversation through model with ttt_mode=True
   → fast weights now encode the conversation
3. Strip KV slots from cache, keep ttt_states[i][2] (past_w)
4. Forward each probe with ttt_mode=False, the stripped cache as past_key_values
5. Greedy-decode answer
```

**Critical**: between samples we fully reset the fast-weight state to checkpoint values to prevent cross-sample contamination. Either we reload the checkpoint or we save/restore the original `past_w` snapshots.

### Condition C — no memory baseline

```
input  = system_prompt + probe        (no conversation, no TTT)
output = model.generate(input, ttt_off=True)
```

The probe alone, with a vanilla model. Establishes the lower bound — pure world knowledge / random guess.

### Condition D — TTT + distractor (level 3 only)

```
1. Reset
2. Forward conversation + distractor through model with ttt_mode=True
3. Strip KV, keep fast weights
4. Probe (same as B)
```

Quantifies how much subsequent unrelated input degrades the fact-encoding fast weight.

## Self-training (B-mini)

Because no public TTT checkpoint exists, we self-train at minimal scale.

### Goal

Bring `ttt_proj` and `ttt_conv` out of init (especially `ttt_conv`, which is zero-init) so the TTT mechanism is functional. We are **not** trying to match the paper's continual-pretrain quality — that's $1k+ of GPU. We need just enough training that the fast-weight delta is meaningfully non-zero and structured.

### Configuration

| Hyperparameter | Paper | Ours (B-mini) | Why |
|---|---|---|---|
| Optimizer steps | 5000 | 300–500 | Just enough to move TTT params off init |
| Global batch size | 64 | 4 | Single GPU |
| Max seq length | 65536 | 4096 | Match `ttt_chunk` exactly |
| `ttt_chunk` | 4096 | **1024** | Match expected eval input length (see DECISIONS D-002) |
| Learning rate | 5e-6 | 5e-6 | Unchanged |
| Trained params | All | `ttt_proj` + `ttt_conv` only (base frozen) | Param-efficient; preserves Qwen3-8B world knowledge |
| Data | Long-context corpus | Conversational (OpenAssistant or ShareGPT subset) | Distribution match to eval |
| GPU | 8×H100 | 1×A100-40G | Cost |

### Sanity checks (gate before running 300 samples)

1. **TTT-on vs TTT-off output diverges.** Same prompt, same seed, two settings: `ttt_mode=True` and `ttt_mode=False` after 4096+ tokens of conversation. Outputs must differ. If identical, TTT did nothing.
2. **`ttt_proj` weight norm > 0**, `ttt_conv` weight norm > 0 after training.
3. **Fast weight is reproducible.** Same input twice → same `past_w` (deterministic forward).
4. **Fast weight is input-dependent.** Two different inputs → two different `past_w`.
5. **KV-stripped cache forward works.** Construct a cache with zeroed KV but kept `past_w`, forward a probe through it, no shape errors.

If any sanity check fails, debug before running conditions.

## Evaluation metrics

### Primary

- **EM (exact match)**: normalize (lowercase, strip punctuation, strip articles, collapse whitespace) and check `gold_answer ⊆ generated`. Per-probe binary.
- **F1 (token-level)**: standard SQuAD-style precision/recall over normalized tokens. Match if F1 > 0.5.

### Headline metrics

- `memory_efficiency_ratio = EM(B) / EM(A)` — the verdict number
- `distractor_decay_rate = (EM(D) - EM(B)) / EM(B)` — typically negative

### Breakdowns

- **By category** (6 dims): does TTT favor numeric over preference, etc.
- **By position in conversation** (front 1/3, middle 1/3, back 1/3): is there a forgetting gradient?
- **By layer** (1/2/3): does reasoning hurt vs direct recall? does distractor hurt?

### Figures

1. EM bar chart, 4 conditions × 3 layers
2. EM heatmap, conditions × 6 categories (layer 1)
3. Position decay curve, EM vs position percentile (layer 1, condition B)
4. Distractor effect plot, EM(B) vs EM(D) per sample (layer 3)

## Risks and mitigations

| Risk | Mitigation | Cost if hit |
|---|---|---|
| Architectural assumption fails (KV-strip breaks fast-weight forward) | Sanity check #5 catches it before main runs. If broken, modify model code to decouple. | +4-8h dev |
| Self-trained TTT params still functionally untrained | Sanity check #1+#2 catch it. If hit, double train steps to 1000 (~+$5). | +$5, +6h |
| `ttt_chunk=1024` mismatch with paper-trained models | We're training from scratch, so chunk choice is ours to make. | 0 |
| Probe ambiguity (multiple valid answers but only one in gold) | F1 metric absorbs this. EM is strict; F1 is the soft metric. | accept noise |
| Cross-sample contamination (fast weights leak between samples) | Explicit reset between samples; verified by sanity check that fresh cache gives baseline output. | discovered = abort and fix |
| Modal image build fails | Reproduce on vast.ai with raw Linux GPU host (`setup.sh` covers this) | +1h |

## Original user brief

(Preserved verbatim for traceability.)

> 项目名称： TTT对话记忆实验（ttt-conv-memory）
> 目标： 验证测试时训练（TTT）的快速权重更新能否替代上下文窗口作为多轮对话记忆机制。具体测量：模型通过TTT处理一段包含事实的对话后，在不将原始对话放入上下文的条件下，能回忆多少事实。
>
> [... full brief in commit history of SPEC.md, see initial commit ...]

The original brief is in the commit history as the seed — see git log for `SPEC.md`. This document supersedes it where they conflict. All deviations are recorded in `DECISIONS.md`.
