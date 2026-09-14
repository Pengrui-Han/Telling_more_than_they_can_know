"""Build a concept direction from contrast pairs and inject it into the residual stream.

Conventions (Qwen2/Llama-style decoder, `model.model.layers[i]`):
  * "layer L" means the output of zero-indexed decoder block L, before final norm.
    Both extraction and injection hook that block. hidden_states[L + 1] is not
    reliable at the last block because it can contain the final-normalised state.
  * The hook adds `scale * direction` at every position on every forward pass
    (prompt and each generated token alike).
"""

from contextlib import contextmanager
import math

import torch


def decoder_block(model, layer: int):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a Qwen2/Llama-style model with model.model.layers")
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer must be in [0, {len(layers) - 1}], got {layer}")
    return layers[layer]


@torch.no_grad()
def last_token_residual(tokenizer, model, text: str, layer: int) -> torch.Tensor:
    """Residual stream at the last token of `text` (raw, no chat template), after block `layer`."""
    block = decoder_block(model, layer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    residual = None

    def capture(_module, _inputs, output):
        nonlocal residual
        hidden = output[0] if isinstance(output, tuple) else output
        residual = hidden[0, -1].detach().float().cpu().clone()

    handle = block.register_forward_hook(capture)
    try:
        # Skip the vocabulary head and retain only the requested last-token vector.
        model.model(**inputs, use_cache=False, output_hidden_states=False)
    finally:
        handle.remove()
    if residual is None:
        raise RuntimeError("The selected decoder block did not execute")
    return residual


def build_direction(tokenizer, model, pairs, layer: int):
    """Mean difference (positive - negative) over contrast pairs, L2-normalised.

    Returns (unit_direction, raw_norm). The raw norm is kept so injection scales can be
    compared with the natural magnitude of the difference.
    """
    diffs = [
        last_token_residual(tokenizer, model, pos, layer)
        - last_token_residual(tokenizer, model, neg, layer)
        for pos, neg in pairs
    ]
    if not diffs:
        raise ValueError("At least one contrast pair is required")
    mean_diff = torch.stack(diffs).mean(0)
    norm = mean_diff.norm()
    if not torch.isfinite(norm) or norm.item() == 0:
        raise ValueError("Contrast direction must have a finite, nonzero norm")
    return mean_diff / norm, norm.item()


@contextmanager
def steer(model, layer: int, direction: torch.Tensor, scale: float):
    """Add `scale * direction` to the residual stream after block `layer` while active.

    scale == 0 registers nothing, so the same code path gives an exact baseline.
    Handles both tuple and tensor block outputs (transformers changed this across versions).
    """
    block = decoder_block(model, layer)
    if direction.ndim != 1 or direction.numel() != model.config.hidden_size:
        raise ValueError("direction must be a vector matching model.config.hidden_size")
    if not torch.isfinite(direction).all() or not math.isfinite(scale):
        raise ValueError("direction and scale must be finite")
    if scale == 0:
        yield
        return

    addend = None

    def hook(_module, _inputs, output):
        nonlocal addend
        hidden = output[0] if isinstance(output, tuple) else output
        if addend is None:
            addend = (scale * direction).to(device=hidden.device, dtype=hidden.dtype)
        hidden.add_(addend)
        return output

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
