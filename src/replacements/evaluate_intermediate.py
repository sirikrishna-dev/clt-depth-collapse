
"""Evaluate recovery of the known 2-hop XOR intermediate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model_adapter import load_original_checkpoint, load_task_split
from .models import CrossLayerTranscoder, PerLayerTranscoder
from .activation_cache import capture_mlp_io


def load_u_labels(data_path: Path, split: str) -> np.ndarray:
    """Load u labels separately; they are never model inputs."""
    df = pd.read_csv(data_path)
    df = df.loc[df["split"].astype(str) == split].copy()

    if "task" in df.columns:
        df = df.loc[
            df["task"].astype(str).isin(
                {"2hop", "boolean_2hop_xor_then_and"}
            )
        ]

    cols = [f"u_{i:02d}" for i in range(16)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing u columns: {missing}")

    labels = df[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all():
        raise ValueError(f"Missing u labels in {split} split.")

    return labels.astype(np.int64)


def load_replacement(path: Path, n_layers: int, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    model_type = saved["model_type"]
    n_features = int(metadata["n_features_per_layer"])

    if model_type == "plt":
        model = PerLayerTranscoder(
            n_layers=n_layers, d_model=64, n_features=n_features
        )
    elif model_type == "clt":
        model = CrossLayerTranscoder(
            n_layers=n_layers,
            d_model=64,
            n_features=n_features,
            write_span=int(metadata["write_span"]),
        )
    else:
        raise ValueError(f"Unsupported replacement type: {model_type}")

    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    return model, metadata


def get_features(model, inputs, device):
    collected = [[] for _ in inputs]

    with torch.no_grad():
        for start in range(0, inputs[0].shape[0], 256):
            batch = [
                x[start:start + 256].to(device)
                for x in inputs
            ]
            _, features = model(batch)

            for layer, values in enumerate(features):
                collected[layer].append(values.detach().cpu())

    return [torch.cat(parts, dim=0) for parts in collected]


def fit_probe(train_features, train_labels):
    """Fit a probe on training data only."""
    x = train_features.detach().cpu().numpy().reshape(
        len(train_labels), -1
    )
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        ),
    )
    probe.fit(x, train_labels)
    return probe


def evaluate_probe(probe, features, labels):
    """Evaluate a previously fitted probe without fitting on these rows."""
    x = features.detach().cpu().numpy().reshape(len(labels), -1)
    predictions = probe.predict(x)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "evaluation_examples": int(len(labels)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data", default="data/boolean_2hop_improved.csv"
    )
    parser.add_argument(
        "--plt",
        default="results/person2/2hop_plt_seed0_features128_span1.pt",
    )
    parser.add_argument(
        "--clt",
        default="results/person2/2hop_clt_seed0_features128_span2.pt",
    )
    parser.add_argument(
        "--output",
        default="results/person2/intermediate_recovery.json",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    data_path = Path(args.data)

    train_labels = load_u_labels(data_path, "train")
    val_labels = load_u_labels(data_path, "validation")
    test_labels = load_u_labels(data_path, "test")

    original, _ = load_original_checkpoint(args.checkpoint, device)

    captured = {}
    for split in ("train", "validation", "test"):
        captured[split] = capture_mlp_io(
            original, str(data_path), "2hop", split, device=device
        )

    for split, labels in (
        ("train", train_labels),
        ("validation", val_labels),
        ("test", test_labels),
    ):
        if len(labels) != captured[split]["n_examples"]:
            raise ValueError(
                f"{split}: label count does not match activation count"
            )

    # Check Person 1's claimed SIIT representation after encoder block 0.
    siit_features = {}
    for split in ("train", "validation", "test"):
        with torch.no_grad():
            x, _ = load_task_split(data_path, "2hop", split)
            h = original.input_proj(x.transpose(1, 2))
            for layer_index, layer in enumerate(original.encoder.layers):
                h = layer(h)
                if layer_index == 0:
                    siit_features[split] = (
                        h[:, :, 0].detach().cpu()
                    )

    siit_results = []
    for bit in range(16):
        probe = fit_probe(
            siit_features["train"][:, bit].reshape(-1, 1),
            train_labels[:, bit],
        )
        siit_results.append({
            "bit": bit,
            "validation": evaluate_probe(
                probe,
                siit_features["validation"][:, bit].reshape(-1, 1),
                val_labels[:, bit],
            ),
            "test": evaluate_probe(
                probe,
                siit_features["test"][:, bit].reshape(-1, 1),
                test_labels[:, bit],
            ),
        })

    results = {
        "task": "2hop",
        "intermediate": "u = XOR(a, b)",
        "train_examples": len(train_labels),
        "validation_examples": len(val_labels),
        "test_examples": len(test_labels),
        "original_model_test_bit_accuracy":
            captured["test"]["bit_accuracy"],
        "siit_hidden_dimension_0_probe_by_bit": siit_results,
        "replacement_models": {},
        "interpretation_warning": (
            "Probes were fitted on the training split. Validation and test "
            "splits were not used to fit the probes. Predictability alone "
            "does not prove causal faithfulness or that a feature represents "
            "the true XOR mechanism."
        ),
    }

    for model_type, checkpoint in (
        ("plt", Path(args.plt)),
        ("clt", Path(args.clt)),
    ):
        model, metadata = load_replacement(
            checkpoint, captured["train"]["n_layers"], device
        )

        split_features = {}
        for split in ("train", "validation", "test"):
            split_features[split] = get_features(
                model, captured[split]["inputs"], device
            )

        layer_results = []
        n_layers = len(split_features["train"])

        for layer_index in range(n_layers):
            bit_results = []

            for bit in range(16):
                probe = fit_probe(
                    split_features["train"][layer_index][:, bit, :],
                    train_labels[:, bit],
                )
                validation_score = evaluate_probe(
                    probe,
                    split_features["validation"][layer_index][:, bit, :],
                    val_labels[:, bit],
                )
                test_score = evaluate_probe(
                    probe,
                    split_features["test"][layer_index][:, bit, :],
                    test_labels[:, bit],
                )
                bit_results.append({
                    "bit": bit,
                    "validation": validation_score,
                    "test": test_score,
                })

            layer_results.append({
                "layer": layer_index,
                "mean_validation_accuracy": float(np.mean([
                    r["validation"]["accuracy"] for r in bit_results
                ])),
                "mean_test_accuracy": float(np.mean([
                    r["test"]["accuracy"] for r in bit_results
                ])),
                "mean_validation_f1": float(np.mean([
                    r["validation"]["f1"] for r in bit_results
                ])),
                "mean_test_f1": float(np.mean([
                    r["test"]["f1"] for r in bit_results
                ])),
                "per_bit": bit_results,
            })

        results["replacement_models"][model_type] = {
            "checkpoint": str(checkpoint),
            "base_checkpoint": metadata["base_checkpoint"],
            "layers": layer_results,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nSaved intermediate recovery results to: {output_path}")


if __name__ == "__main__":
    main()
