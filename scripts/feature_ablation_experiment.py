
"""Ablate every PLT/CLT feature and measure reconstruction changes.

This measures replacement-model write changes only. It does not establish
causal effects on the original Transformer's predictions.
"""

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_person2_src"))

from person2_loader.activation_cache import capture_mlp_io
from person2_loader.checkpoint_io import load_replacement_checkpoint
from person2_loader.model_adapter import load_original_checkpoint


def reconstruct_with_features(model, features, model_type):
    """Decode fixed feature activations into reconstructed writes."""
    if model_type == "plt":
        return [
            transcoder.decoder(features[layer])
            for layer, transcoder in enumerate(model.transcoders)
        ]

    outputs = []
    for destination in range(model.n_layers):
        contributions = []

        for source in range(model.n_layers):
            key = f"{source}_{destination}"
            if key in model.decoders:
                contributions.append(model.decoders[key](features[source]))

        output = torch.stack(contributions).sum(dim=0)
        outputs.append(output + model.destination_bias[destination])

    return outputs


def main():
    original_path = ROOT / "checkpoints" / "3hop_L2_seed0.pt"
    data_path = ROOT / "data" / "boolean_3hop.csv"

    checkpoints = {
        "PLT": (ROOT.parent / "3hop_plt_seed0_features128_span1.pt", "plt"),
        "CLT": (ROOT.parent / "3hop_clt_seed0_features128_span2.pt", "clt"),
    }

    required = [original_path, data_path]
    required.extend(path for path, _ in checkpoints.values())

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Capturing test-set MLP inputs...")
    original_model, _ = load_original_checkpoint(original_path)

    captured = capture_mlp_io(
        model=original_model,
        data_path=str(data_path),
        task="3hop",
        split="test",
        batch_size=128,
    )
    inputs = captured["inputs"]
    print(f"Captured {len(inputs)} layers.")

    results = []

    for label, (checkpoint_path, model_type) in checkpoints.items():
        print(f"\nRunning {label} ablation experiment...")
        model, metadata = load_replacement_checkpoint(checkpoint_path)

        with torch.no_grad():
            baseline_writes, baseline_features = model(inputs)
            baseline_decoded = reconstruct_with_features(
                model, baseline_features, model_type
            )

        # Verify that our manual decoder reproduces the normal forward pass.
        for layer, (expected, actual) in enumerate(
            zip(baseline_writes, baseline_decoded)
        ):
            if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-4):
                raise RuntimeError(
                    f"{label} manual reconstruction mismatch at layer {layer}"
                )

        n_layers = len(baseline_features)
        total_features = 0

        for source_layer, activations in enumerate(baseline_features):
            n_features = activations.shape[-1]

            for feature_index in range(n_features):
                modified_features = [
                    feature.clone() for feature in baseline_features
                ]
                modified_features[source_layer][..., feature_index] = 0

                with torch.no_grad():
                    modified_writes = reconstruct_with_features(
                        model, modified_features, model_type
                    )

                for destination, (before, after) in enumerate(
                    zip(baseline_decoded, modified_writes)
                ):
                    difference = before - after
                    mean_abs = difference.abs().mean().item()
                    l2 = torch.linalg.vector_norm(difference).item()
                    baseline_l2 = torch.linalg.vector_norm(before).item()
                    relative_l2 = l2 / max(baseline_l2, 1e-12)

                    results.append({
                        "model": label,
                        "source_layer": source_layer,
                        "feature": feature_index,
                        "destination_layer": destination,
                        "mean_abs_change": mean_abs,
                        "l2_change": l2,
                        "relative_l2_change": relative_l2,
                        "changed": l2 > 1e-10,
                    })

                total_features += 1

        print(f"Tested {total_features} features across {n_layers} layers.")

    output_path = ROOT / "feature_ablation_results.csv"
    fieldnames = [
        "model",
        "source_layer",
        "feature",
        "destination_layer",
        "mean_abs_change",
        "l2_change",
        "relative_l2_change",
        "changed",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} result rows to: {output_path}")
    print(
        "Important: these results measure changes to reconstructed writes, "
        "not causal effects on the original Transformer's predictions."
    )


if __name__ == "__main__":
    main()
