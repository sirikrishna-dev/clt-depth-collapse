"""Explicit forward pass for BitVectorTransformer with intervention points.

Why this file exists
--------------------
`nn.TransformerEncoderLayer` in eval mode with frozen parameters uses PyTorch's
fused "fast path". That path does not call the attention / feed-forward
submodules separately, so forward hooks on them silently never fire. Any
ablation built on hooks would then measure nothing.

This module re-implements the layer computation explicitly (post-LN, which is
PyTorch's default `norm_first=False`):

    a   = SelfAttn(x)                 # attention output       -> site ("attn", L)
    h1  = LayerNorm1(x + a)
    m   = Linear2(GELU(Linear1(h1)))  # MLP (feed-forward) out -> site ("mlp", L)
    out = LayerNorm2(h1 + m)          # block output           -> site ("resid", L)

`tests/test_gt_forward.py` checks that, with no intervention, the output equals
`model(x)` to within 1e-5 for every checkpoint in the zoo.

An intervention is a dict mapping a site (kind, layer) to a function
`f(activation, ctx) -> new_activation`; `ctx` carries the batch index of each
row so resample-style ablations can look up donor activations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_layer(layer):
    if getattr(layer, "norm_first", False):
        raise NotImplementedError("gt_forward assumes post-LN (norm_first=False)")
    act = getattr(layer, "activation", None)
    if act is not F.gelu and not isinstance(act, torch.nn.GELU):
        raise NotImplementedError(f"gt_forward assumes GELU activation, got {act!r}")


def attn_out(layer, x):
    out, _ = layer.self_attn(x, x, x, need_weights=False)
    return out


def mlp_out(layer, h1):
    return layer.linear2(F.gelu(layer.linear1(h1)))


@torch.no_grad()
def forward(model, x, interventions=None, return_cache=False):
    """Run the model on x = [batch, operands, bits].

    Returns logits [batch, bits] (and a cache of every site if return_cache).
    """
    interventions = interventions or {}
    cache = {}
    h = model.input_proj(x.transpose(1, 2))
    if ("resid", -1) in interventions:
        h = interventions[("resid", -1)](h)
    cache[("resid", -1)] = h
    for L, layer in enumerate(model.encoder.layers):
        _check_layer(layer)
        a = attn_out(layer, h)
        if ("attn", L) in interventions:
            a = interventions[("attn", L)](a)
        cache[("attn", L)] = a
        h1 = layer.norm1(h + a)
        m = mlp_out(layer, h1)
        if ("mlp", L) in interventions:
            m = interventions[("mlp", L)](m)
        cache[("mlp", L)] = m
        h = layer.norm2(h1 + m)
        if ("resid", L) in interventions:
            h = interventions[("resid", L)](h)
        cache[("resid", L)] = h
    if model.encoder.norm is not None:
        h = model.encoder.norm(h)
    logits = model.head(h).squeeze(-1)
    return (logits, cache) if return_cache else logits


def n_layers(model):
    return len(model.encoder.layers)
