"""Capture MLP replacement targets plus optional whole-block diagnostics.

Primary PLT/CLT interface (agreed with Person 3):
  input  = input to encoder.layers[i].linear1 (after norm2 in the default
           norm_first=False TransformerEncoderLayer)
  target = output of encoder.layers[i].linear2 (before dropout2 and the
           residual addition; dropout is disabled in eval mode)

Diagnostic-only interface:
  block_input  = residual stream entering encoder.layers[i]
  block_output = residual stream leaving encoder.layers[i]

Block-boundary captures are supplementary diagnostics. They are not the
reconstruction target for the MLP-level PLT/CLT experiment.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model_adapter import load_task_split


def capture_mlp_io(
    model: nn.Module,
    data_path: str,
    task: str,
    split: str,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    capture_block_io: bool = False,
) -> dict[str, Any]:
    """Capture MLP inputs and writes; optionally capture residual block borders.

    The model is run in evaluation mode so dropout is disabled. All returned
    activation tensors are detached to CPU memory. Intermediate dataset
    columns (u/v) are never passed to the original model.

    Args:
        capture_block_io: If true, also capture each TransformerEncoderLayer's
            input and output. This is intended for activation inspection and
            diagnostics; it is disabled in the training cache by default to
            reduce memory use.
    """
    x, y = load_task_split(data_path, task, split)
    device = torch.device(device)
    model = model.to(device).eval()
    try:
        layers = list(model.encoder.layers)  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("Expected model.encoder.layers, compatible with BitVectorTransformer") from exc

    mlp_inputs: list[list[torch.Tensor]] = [[] for _ in layers]
    mlp_writes: list[list[torch.Tensor]] = [[] for _ in layers]
    block_inputs: list[list[torch.Tensor]] = [[] for _ in layers]
    block_outputs: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = []

    for layer_idx, layer in enumerate(layers):
        def linear1_pre_hook(module, args, idx=layer_idx):
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError(f"linear1 pre-hook at layer {idx} received no tensor input")
            mlp_inputs[idx].append(args[0].detach().to("cpu"))

        def linear2_hook(module, args, output, idx=layer_idx):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"linear2 hook at layer {idx} received non-tensor output")
            mlp_writes[idx].append(output.detach().to("cpu"))

        handles.append(layer.linear1.register_forward_pre_hook(linear1_pre_hook))
        handles.append(layer.linear2.register_forward_hook(linear2_hook))

        if capture_block_io:
            def block_pre_hook(module, args, idx=layer_idx):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError(f"block pre-hook at layer {idx} received no tensor input")
                block_inputs[idx].append(args[0].detach().to("cpu"))

            def block_post_hook(module, args, output, idx=layer_idx):
                # TransformerEncoderLayer returns a tensor for this model's
                # default configuration. Fail loudly if that contract changes.
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"block hook at layer {idx} returned a non-tensor output")
                block_outputs[idx].append(output.detach().to("cpu"))

            handles.append(layer.register_forward_pre_hook(block_pre_hook))
            handles.append(layer.register_forward_hook(block_post_hook))

    logits_batches = []
    try:
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for xb, _yb in loader:
                logits_batches.append(model(xb.to(device)).detach().cpu())
    finally:
        for handle in handles:
            handle.remove()

    inputs = [torch.cat(parts, dim=0) for parts in mlp_inputs]
    writes = [torch.cat(parts, dim=0) for parts in mlp_writes]
    block_in = [torch.cat(parts, dim=0) for parts in block_inputs] if capture_block_io else []
    block_out = [torch.cat(parts, dim=0) for parts in block_outputs] if capture_block_io else []
    logits = torch.cat(logits_batches, dim=0)
    preds = (torch.sigmoid(logits) >= 0.5).float()
    bit_accuracy = (preds == y).float().mean().item()
    exact_vector_accuracy = (preds == y).all(dim=1).float().mean().item()

    if not inputs or len(inputs) != len(writes) or any(a.shape != b.shape for a, b in zip(inputs, writes)):
        raise RuntimeError("Captured MLP input/write shapes are inconsistent.")
    diagnostic_tensors = block_in + block_out
    if capture_block_io and (
        len(block_in) != len(layers) or len(block_out) != len(layers)
        or any(a.shape != b.shape for a, b in zip(block_in, block_out))
    ):
        raise RuntimeError("Captured block-input/block-output shapes are inconsistent.")
    all_activations = inputs + writes + diagnostic_tensors
    if any(not torch.isfinite(t).all() for t in all_activations):
        raise ValueError("Captured activations contain NaN or infinite values.")

    return {
        "inputs": inputs,
        "writes": writes,
        "block_inputs": block_in,
        "block_outputs": block_out,
        "block_io_captured": capture_block_io,
        "logits": logits,
        "bit_accuracy": bit_accuracy,
        "exact_vector_accuracy": exact_vector_accuracy,
        "task": task,
        "split": split,
        "n_examples": len(x),
        "n_layers": len(layers),
        "model_input_shape": tuple(x.shape),
        "activation_shapes": [tuple(t.shape) for t in inputs],
        "mlp_write_shapes": [tuple(t.shape) for t in writes],
        "block_input_shapes": [tuple(t.shape) for t in block_in],
        "block_output_shapes": [tuple(t.shape) for t in block_out],
    }
