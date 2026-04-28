"""
model_utils.py — Qwen3-8B + In-Place TTT loading and fast-weight controls.

This module is only meant to run inside the GPU container where
$TTT_REPO points at a clone of ByteDance-Seed/In-Place-TTT.

Key primitives:

  load_ttt_qwen3(...)         -> (model, tokenizer)
  freeze_base_train_ttt_only  -> only ttt_proj + ttt_conv get gradients
  save_ttt_params / load_ttt_params  -> persist just our trainable subset
  fresh_ttt_cache()           -> empty TTTDynamicCache
  kv_stripped_clone(cache)    -> clone with empty KV but kept past_w (THE primitive)
  enable_ttt_updates / disable_ttt_updates(model)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

# Add upstream In-Place TTT repo to PYTHONPATH so we can import their modeling code.
TTT_REPO = os.environ.get("TTT_REPO", "/opt/repos/In-Place-TTT")
if TTT_REPO and Path(TTT_REPO).exists() and TTT_REPO not in sys.path:
    sys.path.insert(0, TTT_REPO)


def _import_ttt():
    """Lazy import — will fail outside GPU container, that's fine."""
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config  # type: ignore
    from inference_model.hf_qwen3.modeling_qwen3 import (  # type: ignore
        Qwen3ForCausalLM,
        TTTDynamicCache,
    )
    return Qwen3Config, Qwen3ForCausalLM, TTTDynamicCache


def load_ttt_qwen3(
    model_id: str = "Qwen/Qwen3-8B",
    ttt_layers: tuple[int, ...] = (0, 6, 12, 18, 24, 30, 36),
    ttt_chunk: int = 64,
    ttt_lr: float = 3.0,
    ttt_target: str = "hidden_states",
    ttt_proj: bool = True,
    attn_impl: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Load Qwen3-8B base + In-Place TTT layers attached.

    `ttt_layers` is the list of transformer-block indices that get a TTT
    fast-weight head. The default `(0, 6, 12, 18, 24, 30, 36)` matches the
    upstream config (every 6th layer of a 36-layer model = 7 layers).

    `attn_impl` defaults to "sdpa" (PyTorch native) so we don't need flash-attn.
    Upstream README assumes flash-attn but the modeling code dispatches via
    `config._attn_implementation` — sdpa works for our use case.

    We deviate from the paper's `ttt_chunk=4096` to a much smaller value (64)
    to match our measured dialogue length distribution (see DECISIONS.md D-002).
    """
    Qwen3Config, Qwen3ForCausalLM, _ = _import_ttt()
    from transformers import AutoTokenizer

    config = Qwen3Config.from_pretrained(model_id)
    config.ttt_mode = True
    config.ttt_layers = list(ttt_layers)
    config.ttt_chunk = ttt_chunk
    config.ttt_lr = ttt_lr
    config.ttt_target = ttt_target
    config.ttt_proj = ttt_proj
    config._attn_implementation = attn_impl

    model = Qwen3ForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def freeze_base_train_ttt_only(model) -> tuple[int, int]:
    """Freeze base, train only ttt_proj + ttt_conv. Returns (n_total, n_trainable)."""
    n_total, n_trainable = 0, 0
    trainable_names = []
    for name, param in model.named_parameters():
        n_total += param.numel()
        if "ttt_proj" in name or "ttt_conv" in name:
            param.requires_grad = True
            n_trainable += param.numel()
            trainable_names.append(name)
        else:
            param.requires_grad = False
    print(f"[freeze_base_train_ttt_only] {len(trainable_names)} trainable tensors")
    print(f"[freeze_base_train_ttt_only] examples: {trainable_names[:4]}")
    return n_total, n_trainable


def save_ttt_params(model, path: str | Path) -> None:
    """Save just ttt_proj + ttt_conv weights."""
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "ttt_proj" in k or "ttt_conv" in k
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[save_ttt_params] {len(state)} tensors -> {path}")


def load_ttt_params(model, path: str | Path) -> None:
    """Load ttt_proj + ttt_conv weights into model in-place (strict=False)."""
    if not Path(path).exists():
        print(f"[load_ttt_params] no checkpoint at {path}, skipping (using init weights)")
        return
    state = torch.load(path, map_location="cpu")
    # Move to model device
    for k in list(state.keys()):
        state[k] = state[k].to(next(model.parameters()).device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `missing` will be huge (all base weights) — we expect that. We care about unexpected.
    if unexpected:
        print(f"[load_ttt_params] WARNING: unexpected keys: {unexpected[:5]}")
    print(f"[load_ttt_params] loaded {len(state)} tensors from {path}")


def fresh_ttt_cache():
    """Brand new empty cache."""
    _, _, TTTDynamicCache = _import_ttt()
    return TTTDynamicCache()


def kv_stripped_clone(cache):
    """The experimental primitive.

    Build a new TTTDynamicCache that:
      - has empty KV slots (as if no past tokens were attended)
      - preserves each layer's `past_w` (the conversation-modified fast weight)
      - sets `past_h` and `past_t` to None so the probe forward starts at a
        clean chunk boundary (NOT concatenated with the conversation's tail)

    The `past_h_tail` / `past_t_tail` saved by the upstream layer logic represent
    the unfinished last chunk of the conversation. If we kept them, the probe
    forward would do `torch.cat([past_h_tail, probe_hidden_states])` and could
    exceed `ttt_chunk`, firing a TTT update that pollutes the fast weights with
    probe content (the very leak we're trying to prevent).

    Setting them to None forces the layer's `if past_h is None: present_h = hidden_states`
    branch, which keeps the probe forward as a standalone short input. With our
    `ttt_chunk=64` and probe length ≈40-60 tokens, the inner
    `if seq_len < ttt_chunk: return ...` guard then skips the update entirely —
    `past_w` is read for the down-projection but not modified. Exactly what we want.
    """
    _, _, TTTDynamicCache = _import_ttt()
    new = TTTDynamicCache()
    if hasattr(cache, "ttt_states") and cache.ttt_states:
        new.ttt_states = []
        for st in cache.ttt_states:
            past_h, past_t, past_w = st
            new.ttt_states.append((
                None,       # drop past_h tail to prevent probe-time update
                None,       # drop past_t tail
                past_w,     # preserve fast weight (the whole point)
            ))
    # KV slots intentionally left empty — DynamicCache.key_cache and .value_cache
    # are empty lists by default in fresh instance.
    return new


def snapshot_ttt_params(model) -> dict:
    """CPU snapshot of all ttt_proj + ttt_conv params, for restore later."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "ttt_proj" in k or "ttt_conv" in k
    }


def zero_ttt_params(model) -> None:
    """Functionally disable TTT: zero ttt_proj.weight and ttt_conv.weight in place.

    Why: the upstream Qwen3 modeling code attaches ttt_proj/ttt_conv at construction
    time (gated on `config.ttt_mode` and `config.ttt_layers`). Once attached, the
    forward path branches into the TTT logic regardless of any runtime flag —
    `config.ttt_mode = False` set after init is a no-op for inference. To make the
    model behave as vanilla Qwen3-8B at inference time we have to zero the TTT
    parameters so that:
      - dw = 0  (because dw involves ttt_proj.weight or ttt_conv-processed t)
      - present_w stays equal to original down_proj.weight
      - the linear projection result is identical to vanilla down_proj(h)
    The TTT code path still executes (we pay extra compute), but produces identical
    numerical output to vanilla Qwen3-8B.
    """
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "ttt_proj" in name or "ttt_conv" in name:
                p.data.zero_()
                n += 1
    print(f"[zero_ttt_params] zeroed {n} TTT params (functional vanilla mode)")


def restore_ttt_params(model, snapshot: dict) -> None:
    """Restore ttt_proj + ttt_conv params from a snapshot dict (from snapshot_ttt_params)."""
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.data.copy_(snapshot[name].to(p.device))
                n += 1
    print(f"[restore_ttt_params] restored {n} TTT params")


def ttt_param_norms(model) -> dict[str, float]:
    """Diagnostic: L2 norm of each TTT parameter."""
    out = {}
    for name, param in model.named_parameters():
        if "ttt_proj" in name or "ttt_conv" in name:
            out[name] = float(param.detach().norm().item())
    return out


# ---------------------------------------------------------------------------
# Phase A helpers — random Gaussian noise control on down_proj.weight
# ---------------------------------------------------------------------------

def snapshot_down_proj(model, layer_indices) -> dict:
    """CPU snapshot of down_proj.weight for the given transformer-block indices."""
    snap = {}
    for idx in layer_indices:
        w = model.model.layers[idx].mlp.down_proj.weight
        snap[idx] = w.detach().cpu().clone()
    return snap


def restore_down_proj(model, snapshot: dict) -> None:
    """Restore down_proj.weight from a snapshot dict (in place)."""
    with torch.no_grad():
        for idx, w_cpu in snapshot.items():
            target = model.model.layers[idx].mlp.down_proj.weight
            target.data.copy_(w_cpu.to(target.device, dtype=target.dtype))


def apply_gaussian_noise_to_down_proj(
    model,
    layer_indices,
    relative_scale: float,
    seed: int = 0,
) -> dict[int, dict]:
    """Add Gaussian noise to down_proj.weight scaled to a target relative Frobenius norm.

    ||noise||_F / ||W||_F = relative_scale, applied per layer. Returns a dict of
    diagnostic info (||W||, ||noise||, achieved relative scale) per layer.
    """
    info = {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for idx in layer_indices:
            w = model.model.layers[idx].mlp.down_proj.weight
            w_norm = w.detach().float().norm().item()
            noise = torch.randn(w.shape, generator=g, dtype=torch.float32)
            noise = noise.to(w.device)
            noise_norm_raw = noise.norm().item()
            scale = (relative_scale * w_norm) / max(noise_norm_raw, 1e-12)
            noise = noise * scale
            noise_norm = noise.norm().item()
            w.add_(noise.to(w.dtype))
            info[idx] = {
                "w_norm": w_norm,
                "noise_norm": noise_norm,
                "achieved_relative": noise_norm / w_norm,
            }
    return info


@torch.no_grad()
def measure_relative_dw(
    model,
    tokenizer,
    conversation_text: str,
    layer_indices,
    max_seq_len: int = 2048,
) -> dict[int, float]:
    """Estimate ||dw||_F / ||W||_F per TTT layer by comparing the MLP output against
    a recomputed vanilla output that uses the unchanged down_proj.weight.

    Method: forward the conversation through the TTT path (which mutates the cache
    and produces a TTT-modified MLP output). Hook each TTT-bearing MLP and capture
    its (input, output). Recompute the vanilla output by running gate_proj/up_proj
    explicitly and applying the unchanged down_proj.weight via F.linear. The
    relative difference at the MLP output is `||x · dw^T||_F / ||x · W^T||_F`,
    which is a tight proxy for `||dw||_F / ||W||_F` for typical hidden states.

    Assumes the TTT layer leaves `mlp.down_proj.weight` unchanged at the parameter
    level and bypasses it via a local matmul against `W + dw`. This is true for
    In-Place-TTT (where the .weight tensor is the base, never mutated).
    """
    import torch.nn.functional as F

    captured: dict[int, tuple] = {}

    def make_hook(layer_idx, mlp_module):
        def hook(_module, inputs, output):
            x = inputs[0]
            # In-Place-TTT MLP returns (hidden_states, present_w); fall back to
            # the tensor if it's a plain Module.
            if isinstance(output, tuple):
                mlp_out = output[0]
            else:
                mlp_out = output
            captured[layer_idx] = (x.detach(), mlp_out.detach())
        return hook

    handles = []
    for idx in layer_indices:
        mlp = model.model.layers[idx].mlp
        handles.append(mlp.register_forward_hook(make_hook(idx, mlp)))

    try:
        inp = tokenizer(
            conversation_text, return_tensors="pt",
            truncation=True, max_length=max_seq_len,
        ).to(model.device)
        cache = fresh_ttt_cache()
        _ = model(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            past_key_values=cache,
            use_cache=True,
        )
    finally:
        for h in handles:
            h.remove()

    out: dict[int, float] = {}
    for idx, (x, y_ttt) in captured.items():
        mlp = model.model.layers[idx].mlp
        # Vanilla recompute: re-derive intermediate, apply unchanged down_proj.weight
        intermediate = F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
        y_vanilla = F.linear(intermediate, mlp.down_proj.weight)
        if mlp.down_proj.bias is not None:
            y_vanilla = y_vanilla + mlp.down_proj.bias
        diff = (y_ttt - y_vanilla).float()
        denom = y_vanilla.float().norm().item()
        out[idx] = float(diff.norm().item() / max(denom, 1e-12))
    return out
