# ttt-conv-memory

Empirical validation of the question:

> **Can test-time training (TTT) fast-weight updates serve as a multi-turn conversation memory mechanism, replacing the context window?**

We feed a model a dialogue containing facts, let TTT update its fast weights during ingestion, then **drop the conversation from context** and ask probe questions. The fast-weight state is the only place the facts could persist. We measure how many of them the model can still answer.

The vehicle is the [In-Place TTT](https://github.com/ByteDance-Seed/In-Place-TTT) architecture from ByteDance-Seed (ICLR 2026 oral) on Qwen3-8B.

## Status

- See [`STATE.md`](STATE.md) for live progress.
- See [`SPEC.md`](SPEC.md) for the experimental design (refined from the original brief with research findings).
- See [`DECISIONS.md`](DECISIONS.md) for every research-level fork in the road and why we took the side we took.
- See [`RESULTS.md`](RESULTS.md) (if it exists yet) for the final report.

## Layout

```
ttt-conv-memory/
├── README.md             # this file
├── SPEC.md               # detailed experimental spec
├── STATE.md              # live state — updated after every step
├── DECISIONS.md          # every fork + rationale
├── RESULTS.md            # final report (written at end)
├── requirements.txt      # pinned Python deps for the GPU host
├── setup.sh              # bare-metal Linux GPU host install
├── modal_app.py          # Modal serverless entry point
├── build_benchmark.py    # generate the 300-sample benchmark
├── benchmark_v1.json     # the generated benchmark (committed)
├── model_utils.py        # model loading + TTT cache / fast-weight control
├── train_minimal.py      # minimal continual-pretrain to bring TTT params out of init
├── run_experiment.py     # main 4-condition pipeline
├── evaluate.py           # metrics + figures
├── results/              # per-condition outputs + final report
│   ├── condition_a.json
│   ├── condition_b.json
│   ├── condition_c.json
│   ├── condition_d.json
│   ├── report.json
│   └── figures/
└── logs/                 # raw run logs (training, inference, sanity checks)
```

## Quick reference

| Question | Answer |
|---|---|
| Hardware | Single A100-40G via Modal |
| Cost budget | ~$25-30 (covered by Modal $30/mo free credit) |
| Model | Qwen3-8B (base) + In-Place TTT layers |
| Checkpoint | Self-trained minimal (see `DECISIONS.md` D-001) |
| Dataset | 300 synthetic conversations × 5-10 facts each |
| Conditions | A: context baseline / B: TTT memory / C: no memory / D: TTT + distractor |
| Verdict bar | Memory efficiency ratio = EM(B) / EM(A) > 0.7 ⇒ "this path works" |

## Reproduce

```bash
# 1. Build the benchmark (CPU-only, ~30s)
python build_benchmark.py

# 2. End-to-end on Modal
modal run modal_app.py::full_pipeline

# OR step-by-step on a Linux GPU host
bash setup.sh
python train_minimal.py
python run_experiment.py --condition all
python evaluate.py
```

## License

MIT (see `LICENSE`).
