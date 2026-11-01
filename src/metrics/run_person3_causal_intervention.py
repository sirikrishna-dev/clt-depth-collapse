
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.replacements.activation_cache import capture_mlp_io
from src.replacements.checkpoint_io import load_replacement_checkpoint
from src.replacements.model_adapter import (
    load_original_checkpoint,
    load_task_split,
)


def get_encoder(replacement, model_type, layer):
    if model_type == "clt":
        return replacement.encoders[layer]
    return replacement.transcoders[layer].encoder


def get_decoder(replacement, model_type, source, target):
    if model_type == "clt":
        return replacement.decoders[f"{source}_{target}"]

    if source != target:
        raise ValueError("PLT supports same-layer writes only.")

    return replacement.transcoders[source].decoder


def get_targets(replacement, model_type, source):
    if model_type == "plt":
        return [source]

    return sorted(
        int(key.split("_")[1])
        for key in replacement.decoders.keys()
        if int(key.split("_")[0]) == source
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument(
        "--model-type",
        choices=["plt", "clt"],
        required=True,
    )
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--feature-id", type=int, default=22)

    parser.add_argument(
        "--target-layers",
        type=str,
        default="all",
        help="Comma-separated destination layers, or all",
    )

    parser.add_argument(
        "--output-dir",
        default="results/person3/causal_pilot",
    )

    args = parser.parse_args()

    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original Transformer and replacement model.
    original, checkpoint_meta = load_original_checkpoint(
        args.checkpoint,
        device=device,
    )

    replacement, replacement_meta = load_replacement_checkpoint(
        args.replacement
    )

    replacement = replacement.to(device).eval()

    task = checkpoint_meta["task"]

    # Capture baseline MLP inputs on the test split.
    baseline = capture_mlp_io(
        original,
        args.data,
        task,
        "test",
        batch_size=128,
        device=device,
    )

    x, y = load_task_split(args.data, task, "test")
    x = x.to(device)
    y = y.to(device)

    n_layers = len(original.encoder.layers)
    source = args.source_layer
    feature_id = args.feature_id

    if not 0 <= source < n_layers:
        raise ValueError(f"Invalid source layer: {source}")

    if len(baseline["inputs"]) != n_layers:
        raise RuntimeError("Unexpected number of captured layers.")

    # Calculate original feature activations.
    baseline_features = []

    with torch.no_grad():
        for layer in range(n_layers):
            encoder = get_encoder(
                replacement,
                args.model_type,
                layer,
            )

            features = torch.relu(
                encoder(baseline["inputs"][layer])
            )

            baseline_features.append(features)

    source_features = baseline_features[source]

    if not 0 <= feature_id < source_features.shape[-1]:
        raise ValueError(
            f"Feature {feature_id} is out of range. "
            f"Available features: {source_features.shape[-1]}"
        )

    # Remove one feature.
    ablated_features = source_features.clone()
    ablated_features[..., feature_id] = 0.0

    # Determine permitted destination layers.
    targets = get_targets(
        replacement,
        args.model_type,
        source,
    )

    # Apply optional destination filtering.
    if args.target_layers.lower() != "all":
        selected_targets = [
            int(value.strip())
            for value in args.target_layers.split(",")
        ]

        if len(selected_targets) != len(set(selected_targets)):
            raise ValueError("Duplicate destination layers supplied.")

        invalid = set(selected_targets) - set(targets)

        if invalid:
            raise ValueError(
                f"Invalid destinations {sorted(invalid)}. "
                f"Available destinations: {targets}"
            )

        targets = selected_targets

    if not targets:
        raise ValueError("No destination layers selected.")

    # Calculate decoder-write changes.
    write_deltas = {}

    with torch.no_grad():
        for target in targets:
            decoder = get_decoder(
                replacement,
                args.model_type,
                source,
                target,
            )

            original_write = decoder(source_features)
            ablated_write = decoder(ablated_features)

            write_deltas[target] = (
                ablated_write - original_write
            )

    # Baseline model output.
    with torch.no_grad():
        baseline_logits = original(x)

    # Capture modified MLP inputs and inject write changes.
    modified_inputs = [[] for _ in range(n_layers)]
    handles = []

    for layer_idx, layer in enumerate(original.encoder.layers):

        def input_hook(module, inputs, idx=layer_idx):
            modified_inputs[idx].append(
                inputs[0].detach().cpu()
            )

        handles.append(
            layer.linear1.register_forward_pre_hook(input_hook)
        )

        if layer_idx in write_deltas:
            delta = write_deltas[layer_idx]

            def write_hook(module, inputs, output, d=delta):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "Expected tensor output from linear2."
                    )

                return output + d.to(
                    device=output.device,
                    dtype=output.dtype,
                )

            handles.append(
                layer.linear2.register_forward_hook(write_hook)
            )

    try:
        with torch.no_grad():
            intervened_logits = original(x)
    finally:
        for handle in handles:
            handle.remove()

    modified_inputs = [
        torch.cat(parts, dim=0)
        for parts in modified_inputs
    ]

    # Measure downstream feature changes.
    feature_rows = []

    with torch.no_grad():
        for layer in range(n_layers):
            encoder = get_encoder(
                replacement,
                args.model_type,
                layer,
            )

            changed_features = torch.relu(
                encoder(modified_inputs[layer])
            )

            original_features = baseline_features[layer]

            difference = (
                changed_features - original_features
            ).abs()

            feature_rows.append({
                "layer": layer,
                "mean_absolute_feature_change": difference.mean().item(),
                "max_absolute_feature_change": difference.max().item(),
                "changed_feature_fraction": (
                    difference > 1e-6
                ).float().mean().item(),
            })

    # Measure final prediction changes.
    logit_difference = (
        intervened_logits - baseline_logits
    ).abs()

    baseline_predictions = (
        torch.sigmoid(baseline_logits) >= 0.5
    ).float()

    intervened_predictions = (
        torch.sigmoid(intervened_logits) >= 0.5
    ).float()

    prediction_change_fraction = (
        baseline_predictions != intervened_predictions
    ).float().mean().item()

    baseline_accuracy = (
        baseline_predictions == y
    ).float().mean().item()

    intervened_accuracy = (
        intervened_predictions == y
    ).float().mean().item()

    # Record direct decoder contributions.
    write_rows = []

    for target, delta in write_deltas.items():
        write_rows.append({
            "source_layer": source,
            "target_layer": target,
            "feature_id": feature_id,
            "mean_absolute_write_change": delta.abs().mean().item(),
            "max_absolute_write_change": delta.abs().max().item(),
        })

    feature_df = pd.DataFrame(feature_rows)
    write_df = pd.DataFrame(write_rows)

    seed = checkpoint_meta.get("seed", "unknown")

    prefix = (
        f"{task}_seed{seed}_{args.model_type}"
        f"_L{source}_F{feature_id}"
        f"_targets_{'-'.join(map(str, targets))}"
    )
    feature_csv = (
        output_dir / f"{prefix}_downstream_features.csv"
    )

    write_csv = (
        output_dir / f"{prefix}_write_changes.csv"
    )

    report_json = (
        output_dir / f"{prefix}_intervention.json"
    )

    feature_df.to_csv(feature_csv, index=False)
    write_df.to_csv(write_csv, index=False)

    report = {
        "experiment": "single_feature_decoder_write_intervention",
        "model_type": args.model_type,
        "source_layer": source,
        "feature_id": feature_id,
        "test_examples": len(x),
        "direct_destination_layers": targets,
        "feature_effects": feature_rows,
        "write_effects": write_rows,
        "baseline_bit_accuracy": baseline_accuracy,
        "intervened_bit_accuracy": intervened_accuracy,
        "prediction_change_fraction": prediction_change_fraction,
        "mean_absolute_logit_change": logit_difference.mean().item(),
        "max_absolute_logit_change": logit_difference.max().item(),
        "causal_interpretation": (
            "Measures the effect of injecting the selected feature's "
            "decoder-write changes into the original Transformer. "
            "This is a controlled transcoder-derived write intervention, "
            "not proof that the feature is an intrinsic causal variable "
            "of the original Transformer."
        ),
        "graph_edges_are_causal": False,
    }

    report_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nINTERVENTION COMPLETED")
    print("---------------------")
    print(f"Model: {args.model_type.upper()}")
    print(f"Intervened feature: L{source}, F{feature_id}")
    print(f"Direct destination layers: {targets}")

    print("\nDownstream feature changes:")
    print(feature_df.to_string(index=False))

    print("\nPrediction change fraction:", prediction_change_fraction)
    print("Baseline bit accuracy:", baseline_accuracy)
    print("Intervened bit accuracy:", intervened_accuracy)

    print("\nSaved:")
    print(feature_csv)
    print(write_csv)
    print(report_json)


if __name__ == "__main__":
    main()
