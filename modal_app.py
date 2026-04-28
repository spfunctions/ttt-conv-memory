"""
modal_app.py — Modal serverless entry point.

Each phase of the pipeline is a Modal Function. They share an A100-40G image
with the In-Place TTT repo + pinned ML stack and a single named Volume.

Run from the local machine:

    modal run modal_app.py::full_pipeline             # train + 4 conditions + eval
    modal run modal_app.py::full_pipeline --steps 800 # bigger training run

    modal run modal_app.py::quick_smoke               # tiny smoke test
    modal run modal_app.py::sanity_checks             # mechanism sanity tests
    modal run modal_app.py::run_one --cond b --limit 5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

app = modal.App("ttt-conv-memory")

# ---------------------------------------------------------------------------
# Image: debian_slim Python 3.11 + torch 2.8 cu128 + transformers stack +
#        In-Place TTT clone. NO flash-attn (we use PyTorch SDPA for ~equivalent
#        speed without a 30-minute source compile).
#
# The upstream In-Place TTT modeling code dispatches attention via
# `config._attn_implementation`; we default to "sdpa" in model_utils.py.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ca-certificates", "curl")
    # Torch first (cu128 wheel)
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==4.57.3",
        "datasets",
        "tiktoken",
        "einops",
        "tqdm",
        "matplotlib",
        "huggingface_hub",
        "blobfile",
        "opt_einsum",
        "liger-kernel",
        "numpy",
        "safetensors",
        "accelerate",
        "sentencepiece",
    )
    # VeOmni is only needed for upstream training; we don't actually use it for
    # our train_minimal.py (which just uses HF transformers). Keeping it for
    # parity with their import paths in case a code path touches it.
    .pip_install(
        "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"
    )
    .run_commands(
        "git clone https://github.com/ByteDance-Seed/In-Place-TTT.git /opt/repos/In-Place-TTT",
    )
    .workdir("/app")
    .env({
        "TTT_REPO": "/opt/repos/In-Place-TTT",
        "PYTHONPATH": "/opt/repos/In-Place-TTT:/app",
        "HF_HOME": "/data/hf_cache",
        "TRANSFORMERS_CACHE": "/data/hf_cache/transformers",
    })
    # add_local_dir must be LAST (any build step after it requires copy=True)
    .add_local_dir(".", remote_path="/app")
)

vol = modal.Volume.from_name("ttt-conv-memory-vol", create_if_missing=True)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": vol},
    timeout=4 * 3600,
)
def smoke_load_qwen3():
    """Verify base Qwen3-8B + In-Place TTT loads and a forward pass works."""
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/opt/repos/In-Place-TTT")
    import torch
    from model_utils import load_ttt_qwen3, ttt_param_norms
    print("[smoke] loading model...")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=64)
    print(f"[smoke] device: {next(model.parameters()).device}")
    print(f"[smoke] dtype: {next(model.parameters()).dtype}")
    print(f"[smoke] init TTT param norms (first 3): "
          f"{list(ttt_param_norms(model).items())[:3]}")
    # Quick forward
    inp = tokenizer("Hello, world.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=inp["input_ids"])
    print(f"[smoke] logits shape: {out.logits.shape}")
    print("[smoke] OK")
    vol.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": vol},
    timeout=2 * 3600,
)
def sanity_checks(checkpoint: str | None = None):
    """Run the 5 sanity checks from SPEC.md (gate before main runs)."""
    import sys
    sys.path.insert(0, "/app")
    import subprocess
    cmd = ["python", "/app/sanity.py"]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    subprocess.run(cmd, check=True)
    vol.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": vol},
    timeout=4 * 3600,
)
def train_minimal(
    steps: int = 400,
    ttt_chunk: int = 64,
    seq_len: int = 1024,
    batch_size: int = 2,
    grad_accum: int = 2,
    lr: float = 5e-6,
):
    import subprocess, os
    os.makedirs("/data/checkpoints", exist_ok=True)
    cmd = [
        "python", "/app/train_minimal.py",
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--lr", str(lr),
        "--ttt-chunk", str(ttt_chunk),
        "--seq-len", str(seq_len),
        "--out", "/data/checkpoints/ttt_minimal.pt",
    ]
    print(f"[train_minimal] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    vol.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": vol},
    timeout=3 * 3600,
)
def run_one(cond: str, limit: int | None = None, ttt_chunk: int = 64):
    """Run a single condition (a/b/c/d)."""
    import subprocess, os
    os.makedirs("/data/results", exist_ok=True)
    cmd = [
        "python", "/app/run_experiment.py",
        "--checkpoint", "/data/checkpoints/ttt_minimal.pt",
        "--condition", cond,
        "--out-dir", "/data/results",
        "--ttt-chunk", str(ttt_chunk),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[run_one] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    vol.commit()


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
)
def evaluate():
    import subprocess
    cmd = [
        "python", "/app/evaluate.py",
        "--results-dir", "/data/results",
        "--out", "/data/results/report.json",
    ]
    print(f"[evaluate] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    vol.commit()


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
)
def fetch_results() -> dict:
    p = Path("/data/results/report.json")
    if not p.exists():
        return {"error": "no report"}
    return json.loads(p.read_text())


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
)
def fetch_file(path: str) -> bytes:
    """Read a file from the volume — used to pull figures or logs back to local."""
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# Local entry points
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke_image():
    """Just verify the image builds and Qwen3-8B loads."""
    smoke_load_qwen3.remote()


@app.local_entrypoint()
def sanity():
    """Run sanity checks against the trained checkpoint."""
    sanity_checks.remote(checkpoint="/data/checkpoints/ttt_minimal.pt")


@app.local_entrypoint()
def train(steps: int = 400, ttt_chunk: int = 64, seq_len: int = 1024,
          batch_size: int = 2, grad_accum: int = 2, lr: float = 5e-6):
    train_minimal.remote(
        steps=steps, ttt_chunk=ttt_chunk, seq_len=seq_len,
        batch_size=batch_size, grad_accum=grad_accum, lr=lr,
    )


@app.local_entrypoint()
def run_condition_remote(cond: str = "all", limit: int | None = None):
    if cond == "all":
        for c in ("a", "c", "b", "d"):
            run_one.remote(cond=c, limit=limit)
    else:
        run_one.remote(cond=cond, limit=limit)


@app.local_entrypoint()
def full_pipeline(steps: int = 400, ttt_chunk: int = 64):
    """End-to-end: train -> sanity -> conditions -> eval -> print headline."""
    print("=== train ===")
    train_minimal.remote(steps=steps, ttt_chunk=ttt_chunk)
    print("=== sanity ===")
    sanity_checks.remote(checkpoint="/data/checkpoints/ttt_minimal.pt")
    print("=== conditions ===")
    for c in ("a", "c", "b", "d"):
        print(f"  cond {c.upper()}")
        run_one.remote(cond=c)
    print("=== evaluate ===")
    evaluate.remote()
    print("=== headline ===")
    rep = fetch_results.remote()
    print(json.dumps(rep.get("headline", {}), indent=2, ensure_ascii=False))


@app.local_entrypoint()
def quick_smoke(limit: int = 3, steps: int = 20):
    """Smoke test: tiny train + tiny eval. Sanity-check the whole pipeline cheaply."""
    print("=== smoke train ===")
    train_minimal.remote(steps=steps, ttt_chunk=64)
    print("=== smoke conditions ===")
    for c in ("a", "c", "b"):
        run_one.remote(cond=c, limit=limit)
    print("=== smoke evaluate ===")
    evaluate.remote()
    rep = fetch_results.remote()
    print(json.dumps(rep, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def pull_results(out_dir: str = "results"):
    """Copy report + figures from the volume to local."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    files = [
        "/data/results/report.json",
        "/data/results/condition_a.json",
        "/data/results/condition_b.json",
        "/data/results/condition_c.json",
        "/data/results/condition_d.json",
        "/data/results/figures/fig1_em_by_condition_level.png",
        "/data/results/figures/fig2_em_heatmap.png",
        "/data/results/figures/fig3_position_decay.png",
        "/data/results/figures/fig4_distractor_effect.png",
    ]
    for remote in files:
        try:
            data = fetch_file.remote(remote)
            local = out / Path(remote).name
            if "figures" in remote:
                local = out / "figures" / Path(remote).name
            local.write_bytes(data)
            print(f"  -> {local}")
        except Exception as e:
            print(f"  skip {remote}: {e}")
