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
    ttt_chunk: int = 1024,
    ttt_lr: float = 3.0,
    ttt_target: str = "hidden_states",
    ttt_proj: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Load Qwen3-8B base + In-Place TTT layers attached.

    `ttt_layers` is the list of transformer-block indices that get a TTT
    fast-weight head. The default `(0, 6, 12, 18, 24, 30, 36)` matches the
    upstream config (every 6th layer of a 36-layer model = 7 layers).

    We deviate from the paper's `ttt_chunk=4096` to `1024` to match our
    expected dialogue length distribution (see DECISIONS.md D-002).
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

    model = Qwen3ForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map=device,
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
      - resets `past_h` and `past_t` to zero so the next forward starts at a clean
        chunk boundary

    After this strip the model is queried with no attention context but with
    the TTT-modified down-projection weights. This is the whole point of the
    experiment.
    """
    _, _, TTTDynamicCache = _import_ttt()
    new = TTTDynamicCache()
    if hasattr(cache, "ttt_states") and cache.ttt_states:
        new.ttt_states = []
        for st in cache.ttt_states:
            past_h, past_t, past_w = st
            new.ttt_states.append((
                torch.zeros_like(past_h) if past_h is not None else None,
                torch.zeros_like(past_t) if past_t is not None else None,
                past_w,  # preserve fast weight
            ))
    # KV slots intentionally left empty — DynamicCache.key_cache and .value_cache
    # are empty lists by default in fresh instance.
    return new


def disable_ttt_updates(model) -> None:
    """In-place: stop fast-weight updates during forward (use during inference probes)."""
    model.config.ttt_mode = False


def enable_ttt_updates(model) -> None:
    model.config.ttt_mode = True


def ttt_param_norms(model) -> dict[str, float]:
    """Diagnostic: L2 norm of each TTT parameter. Used for sanity-check #2."""
    out = {}
    for name, param in model.named_parameters():
        if "ttt_proj" in name or "ttt_conv" in name:
            out[name] = float(param.detach().norm().item())
    return out
