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
    seq_len: int = 512,
    batch_size: int = 1,
    grad_accum: int = 4,
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
    gpu="A100-40GB",
    volumes={"/data": vol},
    timeout=2 * 3600,
)
def phase_a(
    sample_limit: int = 30,
    extra_scales: str = "0.01,0.05,0.20,0.50",
):
    """Phase A: ablation that tests whether the frozen base is the dominant
    failure mode of cond B.

    Step 1: with trained TTT params loaded, forward one conversation through the
    cond B path and measure ||dw||_F / ||W||_F per TTT-bearing layer (output-diff
    method against the unchanged down_proj.weight).

    Step 2: zero TTT params (functional vanilla Qwen3-8B) and sweep Gaussian noise
    of given relative Frobenius magnitudes added to down_proj.weight on the same
    layers. For each magnitude, run cond A' (no context, no TTT, just noisy
    weights) on a subset of benchmark samples.

    If a noise magnitude near the measured dw produces the same gibberish as
    cond B, the frozen-base hypothesis is confirmed: any perturbation of similar
    size breaks the model, regardless of whether it was 'learned' by TTT.
    """
    import sys, os, json
    from pathlib import Path
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/opt/repos/In-Place-TTT")
    import torch
    from model_utils import (
        load_ttt_qwen3, load_ttt_params, snapshot_ttt_params,
        zero_ttt_params, restore_ttt_params,
        snapshot_down_proj, restore_down_proj,
        apply_gaussian_noise_to_down_proj, measure_relative_dw,
    )
    from run_experiment import run_condition_a_prime, select_samples

    os.makedirs("/data/results", exist_ok=True)

    print("[phase_a] loading model + trained TTT checkpoint")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=64)
    load_ttt_params(model, "/data/checkpoints/ttt_minimal.pt")
    trained = snapshot_ttt_params(model)
    model.eval()

    layers = [0, 6, 12, 18, 24, 30]

    print("[phase_a] step 1: measuring ||dw||_F / ||W||_F (output-diff method)")
    bench = json.loads(Path("/app/benchmark_v1.json").read_text())
    sample0 = bench["samples"][0]
    measured_mean = None
    measured_max = None
    diag: dict = {
        "sample_id": sample0["sample_id"],
        "conv_chars": len(sample0["conversation"]),
    }
    try:
        rel_dw = measure_relative_dw(
            model, tokenizer, sample0["conversation"], layers,
        )
        for i, v in rel_dw.items():
            print(f"    layer {i}: rel_dw = {v:.4f}")
        measured_mean = sum(rel_dw.values()) / len(rel_dw)
        measured_max = max(rel_dw.values())
        print(f"[phase_a] mean = {measured_mean:.4f}, max = {measured_max:.4f}")
        diag["per_layer_relative_dw"] = {str(k): v for k, v in rel_dw.items()}
        diag["mean"] = measured_mean
        diag["max"] = measured_max
    except Exception as e:
        import traceback
        print(f"[phase_a] measurement failed: {e}")
        traceback.print_exc()
        diag["measurement_error"] = repr(e)
    Path("/data/results/phase_a_diagnostic.json").write_text(
        json.dumps(diag, indent=2)
    )

    print("[phase_a] step 2: zeroing TTT params (functional vanilla)")
    zero_ttt_params(model)

    scale_set = {round(float(s), 4) for s in extra_scales.split(",") if s.strip()}
    if measured_mean is not None:
        scale_set.add(round(measured_mean, 4))
    if measured_max is not None:
        scale_set.add(round(measured_max, 4))
    scales = sorted(scale_set)
    print(f"[phase_a] sweep scales: {scales}")

    base_dp_snap = snapshot_down_proj(model, layers)
    samples = select_samples(bench["samples"], "a_prime", sample_limit)
    print(f"[phase_a] running cond A' on {len(samples)} samples per scale "
          f"(~{sum(len(s['probes']) for s in samples)} probes per scale)")

    runs: dict = {}
    for scale in scales:
        print(f"\n[phase_a] === scale {scale} ===")
        restore_down_proj(model, base_dp_snap)
        info = apply_gaussian_noise_to_down_proj(model, layers, scale, seed=0)
        for idx, d in info.items():
            print(f"    layer {idx}: ||W||={d['w_norm']:.2f} "
                  f"achieved={d['achieved_relative']:.4f}")
        res = run_condition_a_prime(model, tokenizer, samples)
        suffix = str(scale).replace(".", "p")
        out_path = Path(f"/data/results/condition_a_prime_{suffix}.json")
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"    -> {out_path}")
        runs[str(scale)] = {
            "suffix": suffix,
            "out": str(out_path),
            "n_samples": len(samples),
        }

    restore_down_proj(model, base_dp_snap)

    summary = {
        "diagnostic": diag,
        "scales_run": runs,
        "samples_per_scale": len(samples),
    }
    Path("/data/results/phase_a_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    vol.commit()
    print("[phase_a] done")
    return summary


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": vol},
    timeout=2 * 3600,
)
def inference_scaling(
    sample_limit: int = 30,
    scales: str = "0.05,0.1,0.2,0.3",
):
    """Test the magnitude hypothesis: re-run cond B with the trained TTT params
    multiplicatively scaled at inference time. If a smaller dw produces coherent
    (even if factually wrong) cond B output, magnitude is the dominant issue
    and we can ship by adding magnitude controls during training. If outputs
    remain gibberish at every scale, the TTT's *direction* is also bad.

    For each scale α in `scales`:
      ttt_proj.weight ← α · trained_ttt_proj.weight
      ttt_conv.weight ← α · trained_ttt_conv.weight
    Then forward conv + probe per cond B and save predictions.
    """
    import sys, os, json
    from pathlib import Path
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/opt/repos/In-Place-TTT")
    import torch
    from model_utils import (
        load_ttt_qwen3, load_ttt_params, snapshot_ttt_params,
        restore_ttt_params, measure_relative_dw,
    )
    from run_experiment import run_condition_b, select_samples

    os.makedirs("/data/results", exist_ok=True)

    print("[scaling] loading model + trained TTT")
    model, tokenizer = load_ttt_qwen3(ttt_chunk=64)
    load_ttt_params(model, "/data/checkpoints/ttt_minimal.pt")
    trained = snapshot_ttt_params(model)
    model.eval()

    layers = [0, 6, 12, 18, 24, 30]
    bench = json.loads(Path("/app/benchmark_v1.json").read_text())
    sample0 = bench["samples"][0]
    samples = select_samples(bench["samples"], "b", sample_limit)
    print(f"[scaling] {len(samples)} samples per scale, "
          f"~{sum(len(s['probes']) for s in samples)} probes per scale")

    scale_list = [float(s) for s in scales.split(",") if s.strip()]
    summary: dict = {"scales": {}}

    for scale in scale_list:
        print(f"\n[scaling] === scale α={scale} ===")
        # Reset to trained params, then multiply
        restore_ttt_params(model, trained)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "ttt_proj" in name or "ttt_conv" in name:
                    p.data.mul_(scale)

        # Measure post-scale rel_dw to confirm scaling worked
        try:
            rel_dw = measure_relative_dw(
                model, tokenizer, sample0["conversation"], layers,
            )
            print(f"[scaling]   post-scale rel_dw: "
                  f"{ {i: round(v, 4) for i, v in rel_dw.items()} }")
        except Exception as e:
            print(f"[scaling]   measurement failed: {e}")
            rel_dw = {}

        res = run_condition_b(model, tokenizer, samples)
        suffix = str(scale).replace(".", "p")
        out_path = Path(f"/data/results/condition_b_scaled_{suffix}.json")
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"[scaling]   -> {out_path}")
        summary["scales"][str(scale)] = {
            "suffix": suffix,
            "out": str(out_path),
            "rel_dw": {str(k): v for k, v in rel_dw.items()},
        }

    Path("/data/results/inference_scaling_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    vol.commit()
    print("[scaling] done")
    return summary


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
def train(steps: int = 400, ttt_chunk: int = 64, seq_len: int = 512,
          batch_size: int = 1, grad_accum: int = 4, lr: float = 5e-6):
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
def phase_a_run(sample_limit: int = 30, extra_scales: str = "0.01,0.05,0.20,0.50"):
    """Run Phase A: ||dw|| measurement + Gaussian-noise sweep."""
    summary = phase_a.remote(sample_limit=sample_limit, extra_scales=extra_scales)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def inference_scaling_run(sample_limit: int = 30, scales: str = "0.05,0.1,0.2,0.3"):
    """Run inference-time TTT scaling: shrink ttt_proj/ttt_conv by α and re-run cond B."""
    summary = inference_scaling.remote(sample_limit=sample_limit, scales=scales)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def pull_inference_scaling(out_dir: str = "results"):
    """Copy inference-scaling summary + per-scale prediction files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        b = fetch_file.remote("/data/results/inference_scaling_summary.json")
        (out / "inference_scaling_summary.json").write_bytes(b)
        summary = json.loads(b.decode())
        print(f"  -> {out / 'inference_scaling_summary.json'}")
    except Exception as e:
        print(f"  no summary: {e}")
        summary = {"scales": {}}
    for scale, info in summary.get("scales", {}).items():
        suffix = info["suffix"]
        remote = f"/data/results/condition_b_scaled_{suffix}.json"
        try:
            b = fetch_file.remote(remote)
            (out / f"condition_b_scaled_{suffix}.json").write_bytes(b)
            print(f"  -> {out / f'condition_b_scaled_{suffix}.json'}")
        except Exception as e:
            print(f"  skip {remote}: {e}")


@app.local_entrypoint()
def pull_phase_a(out_dir: str = "results"):
    """Copy Phase A diagnostic + per-scale prediction files from the volume."""
    import os
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # First fetch summary to know which suffixes exist
    try:
        summary_bytes = fetch_file.remote("/data/results/phase_a_summary.json")
        (out / "phase_a_summary.json").write_bytes(summary_bytes)
        summary = json.loads(summary_bytes.decode())
        print(f"  -> {out / 'phase_a_summary.json'}")
    except Exception as e:
        print(f"  no summary: {e}")
        summary = {"scales_run": {}}
    # diagnostic
    try:
        b = fetch_file.remote("/data/results/phase_a_diagnostic.json")
        (out / "phase_a_diagnostic.json").write_bytes(b)
        print(f"  -> {out / 'phase_a_diagnostic.json'}")
    except Exception as e:
        print(f"  no diagnostic: {e}")
    # per-scale
    for scale, info in summary.get("scales_run", {}).items():
        suffix = info["suffix"]
        remote = f"/data/results/condition_a_prime_{suffix}.json"
        try:
            b = fetch_file.remote(remote)
            (out / f"condition_a_prime_{suffix}.json").write_bytes(b)
            print(f"  -> {out / f'condition_a_prime_{suffix}.json'}")
        except Exception as e:
            print(f"  skip {remote}: {e}")


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
