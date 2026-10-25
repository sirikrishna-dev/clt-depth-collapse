"""Standalone smoke test for original model loading and activation capture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .activation_cache import capture_mlp_io
from .model_adapter import load_original_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["2hop", "3hop"], default="2hop")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = p.parse_args()
    data_path = args.data or f"data/boolean_{args.task}_improved.csv"
    device = torch.device(args.device)
    model, metadata = load_original_checkpoint(args.checkpoint, device)
    result = capture_mlp_io(model, data_path, args.task, args.split, args.batch_size, device, capture_block_io=True)
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_task": metadata.get("task"),
        "requested_task": args.task,
        "split": args.split,
        "n_examples": result["n_examples"],
        "model_input_shape": result["model_input_shape"],
        "mlp_input_shapes": result["activation_shapes"],
        "mlp_write_shapes": [tuple(t.shape) for t in result["writes"]],
        "residual_stream_block_input_shapes": result["block_input_shapes"],
        "residual_stream_block_output_shapes": result["block_output_shapes"],
        "primary_reconstruction_interface": "MLP input -> MLP write",
        "diagnostic_interface": "residual stream block input -> block output",
        "logits_shape": tuple(result["logits"].shape),
        "test_bit_accuracy" if args.split == "test" else "split_bit_accuracy": result["bit_accuracy"],
        "test_exact_vector_accuracy" if args.split == "test" else "split_exact_vector_accuracy": result["exact_vector_accuracy"],
        "finite_activations": all(bool(torch.isfinite(t).all()) for t in result["inputs"] + result["writes"]),
        "model_inputs_exclude_u_v": True,
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
