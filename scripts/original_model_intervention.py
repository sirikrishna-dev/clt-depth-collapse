
"""Test whether original Transformer MLP writes affect predictions.

This is a layer-write intervention, not a direct PLT/CLT feature intervention.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_person2_src"))

from person2_loader.model_adapter import (
    load_original_checkpoint,
    load_task_split,
)


def evaluate(model, x, y, batch_size=128, intervene_layer=None, intervention="none"):
    """Evaluate predictions, optionally changing one layer's MLP write."""
    saved_outputs = []
    handle = None

    if intervene_layer is not None:
        layer = model.encoder.layers[intervene_layer]

        def replace_write(module, inputs, output):
            if intervention == "zero":
                return torch.zeros_like(output)
            if intervention == "mean":
                # A constant mean write, computed from this evaluation batch.
                return output.mean(dim=(0, 1), keepdim=True).expand_as(output)
            return output

        handle = layer.linear2.register_forward_hook(replace_write)

    try:
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=False,
        )

        all_logits = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                all_logits.append(model(xb).cpu())
                all_targets.append(yb.cpu())

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        predictions = (torch.sigmoid(logits) >= 0.5).float()

        return {
            "bit_accuracy": (predictions == targets).float().mean().item(),
            "exact_vector_accuracy": (
                (predictions == targets).all(dim=1).float().mean().item()
            ),
            "changed_bits_vs_targets": int((predictions != targets).sum().item()),
            "logits": logits,
            "predictions": predictions,
        }
    finally:
        if handle is not None:
            handle.remove()


def main():
    checkpoint = ROOT / "checkpoints" / "3hop_L2_seed0.pt"
    dataset = ROOT / "data" / "boolean_3hop.csv"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset missing: {dataset}")

    model, metadata = load_original_checkpoint(checkpoint)
    x, y = load_task_split(dataset, task="3hop", split="test")

    print("Checkpoint:", metadata)
    print("Test examples:", len(x))

    baseline = evaluate(model, x, y)
    print("\nBASELINE")
    print(f"Bit accuracy:          {baseline['bit_accuracy']:.4f}")
    print(f"Exact-vector accuracy: {baseline['exact_vector_accuracy']:.4f}")

    for layer_index in range(len(model.encoder.layers)):
        for intervention in ("zero", "mean"):
            result = evaluate(
                model,
                x,
                y,
                intervene_layer=layer_index,
                intervention=intervention,
            )

            changed_predictions = (
                result["predictions"] != baseline["predictions"]
            ).any(dim=1).sum().item()

            logit_difference = (
                # Use the same evaluation order and original baseline logits.
                result["logits"] - baseline["logits"]
            ).abs().mean().item()

            print(f"\nLayer {layer_index} — {intervention.upper()} MLP write")
            print(f"Bit accuracy:          {result['bit_accuracy']:.4f}")
            print(f"Exact-vector accuracy: {result['exact_vector_accuracy']:.4f}")
            print(f"Examples with changed predictions: {changed_predictions}")
            print(f"Mean absolute logit change: {logit_difference:.8f}")

    print(
        "\nInterpretation: this measures sensitivity to interventions on "
        "whole MLP writes. It does not isolate a PLT/CLT feature or prove "
        "that an attribution graph captures the true computational depth."
    )


if __name__ == "__main__":
    main()
