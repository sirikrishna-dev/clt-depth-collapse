from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.replacements.activation_cache import capture_mlp_io
from src.replacements.checkpoint_io import load_replacement_checkpoint
from src.replacements.model_adapter import load_original_checkpoint


def get_encoder(replacement, model_type, layer):
    if model_type == "clt":
        return replacement.encoders[layer]
    return replacement.transcoders[layer].encoder


def find_u_columns(frame, variable="u"):
    cols = [c for c in frame.columns if str(c).lower().startswith(f"{variable}_")]
    if len(cols) != 16:
        raise ValueError(
            f"Expected 16 binary {variable} columns in the CSV; "
            f"found {len(cols)}: {cols}"
        )
    def numeric_suffix(name):
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        return int(digits) if digits else -1
    return sorted(cols, key=numeric_suffix)


def load_u_labels(data_path, variable="u"):
    frame = pd.read_csv(data_path)
    columns = find_u_columns(frame, variable)
    values = frame[columns].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"The {variable} columns must contain only 0/1 values.")
    return torch.tensor(values, dtype=torch.float32)


def capture_features(checkpoint, data, replacement_path, model_type, split, device):
    original, meta = load_original_checkpoint(checkpoint, device=device)
    replacement, _ = load_replacement_checkpoint(replacement_path)
    replacement = replacement.to(device).eval()
    captured = capture_mlp_io(
        original, data, meta["task"], split,
        batch_size=128, device=device
    )
    features = []
    with torch.no_grad():
        for layer in range(len(original.encoder.layers)):
            encoder = get_encoder(replacement, model_type, layer)
            features.append(torch.relu(encoder(captured["inputs"][layer])).cpu())
    return features


def fit_and_score(train_feature, train_labels, eval_feature, eval_labels):
    """Fit one scalar logistic probe per token position; return mean accuracy."""
    scores = []
    for position in range(train_labels.shape[1]):
        x_train = train_feature[:, position].reshape(-1, 1).numpy()
        y_train = train_labels[:, position].numpy().astype(int)
        x_eval = eval_feature[:, position].reshape(-1, 1).numpy()
        y_eval = eval_labels[:, position].numpy().astype(int)

        if np.unique(y_train).size < 2:
            predictions = np.full_like(y_eval, y_train[0])
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42),
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_eval)
        scores.append(accuracy_score(y_eval, predictions))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate how well each individual PLT/CLT feature predicts an intermediate variable (u or v)."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plt", required=True)
    parser.add_argument("--clt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--variable", choices=["u", "v"], default="u",
                        help="Intermediate variable to decode (v exists only in the 3-hop task)")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    device = torch.device("cpu")
    labels = load_u_labels(args.data, args.variable)

    # Dataset convention used in this project: 5,000 train / 1,000 val / 1,000 test.
    train_y = labels[:5000]
    val_y = labels[5000:6000]
    test_y = labels[6000:7000]

    all_rows = []
    for model_type, replacement_path in (
        ("plt", args.plt),
        ("clt", args.clt),
    ):
        train_reps = capture_features(
            args.checkpoint, args.data, replacement_path,
            model_type, "train", device
        )
        val_reps = capture_features(
            args.checkpoint, args.data, replacement_path,
            model_type, "validation", device
        )
        test_reps = capture_features(
            args.checkpoint, args.data, replacement_path,
            model_type, "test", device
        )

        for layer, (train_rep, val_rep, test_rep) in enumerate(
            zip(train_reps, val_reps, test_reps)
        ):
            if train_rep.ndim != 3:
                raise ValueError("Expected features shaped [examples, positions, features].")
            if train_rep.shape[1] != train_y.shape[1]:
                raise ValueError(
                    f"Feature positions={train_rep.shape[1]}, u label width={train_y.shape[1]}"
                )

            for feature_id in range(train_rep.shape[-1]):
                val_acc = fit_and_score(
                    train_rep[:, :, feature_id], train_y,
                    val_rep[:, :, feature_id], val_y
                )
                test_acc = fit_and_score(
                    train_rep[:, :, feature_id], train_y,
                    test_rep[:, :, feature_id], test_y
                )
                all_rows.append({
                    "variable": args.variable,
                    "model_type": model_type,
                    "layer": layer,
                    "feature_id": feature_id,
                    "validation_accuracy": val_acc,
                    "test_accuracy": test_acc,
                })

    results = pd.DataFrame(all_rows)
    # Use deterministic first-rank tie-breaking so exactly top_k are selected per layer.
    results["validation_rank"] = results.groupby(
        ["model_type", "layer"]
    )["validation_accuracy"].rank(method="first", ascending=False)
    results["selected_top_k"] = results["validation_rank"] <= args.top_k

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Saved: {output}")
    for (model_type, layer), group in results.groupby(["model_type", "layer"]):
        print(f"\n{model_type.upper()} layer {layer}:")
        print(
            group.sort_values("validation_rank").head(args.top_k)[
                ["feature_id", "validation_accuracy", "test_accuracy"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
