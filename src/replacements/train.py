"""Train Person 2 PLT/CLT baselines to reconstruct original MLP writes."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .activation_cache import capture_mlp_io
from .model_adapter import load_original_checkpoint
from .models import CrossLayerTranscoder, PerLayerTranscoder


def metrics_for(preds: list[torch.Tensor], targets: list[torch.Tensor], features: list[torch.Tensor]) -> dict[str, float]:
    errors = [p.float() - y.float() for p, y in zip(preds, targets)]
    sum_sq_err = sum(float((e ** 2).sum()) for e in errors)
    sum_sq_target = sum(float((y.float() ** 2).sum()) for y in targets)
    all_target = torch.cat([y.float().reshape(-1) for y in targets])
    concat_errors = torch.cat([e.float().reshape(-1) for e in errors])
    target_var = float(all_target.var(unbiased=False))
    error_var = float(concat_errors.var(unbiased=False))
    total_elements = sum(e.numel() for e in errors)
    mse = sum_sq_err / max(1, total_elements)
    rel_l2 = (sum_sq_err / max(sum_sq_target, 1e-12)) ** 0.5
    explained_variance = 1.0 - error_var / max(target_var, 1e-12)
    concat_features = torch.cat([z.float().reshape(-1) for z in features])
    return {
        "mse": mse,
        "relative_l2": rel_l2,
        "explained_variance": explained_variance,
        "active_feature_fraction": float((concat_features > 1e-6).float().mean()),
        "mean_feature_activation": float(concat_features.mean()),
    }


def forward_model(model, xs):
    preds, features = model(xs)
    return preds, features


def evaluate(model, xs, ys, batch_size, device):
    model.eval()
    pred_all = [[] for _ in ys]
    feature_all = [[] for _ in xs]
    with torch.no_grad():
        ds = TensorDataset(*xs, *ys)
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            bx = [t.to(device) for t in batch[:len(xs)]]
            by = [t.to(device) for t in batch[len(xs):]]
            preds, features = forward_model(model, bx)
            for i, t in enumerate(preds): pred_all[i].append(t.detach().cpu())
            for i, t in enumerate(features): feature_all[i].append(t.detach().cpu())
    preds = [torch.cat(parts) for parts in pred_all]
    features = [torch.cat(parts) for parts in feature_all]
    return metrics_for(preds, ys, features), preds


def train(args):
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available() and args.device in {"auto", "cuda"}:
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    root = Path(args.root).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute(): checkpoint_path = root / checkpoint_path
    data_path = Path(args.data) if args.data else root / "data" / f"boolean_{args.task}_improved.csv"
    if not data_path.is_absolute(): data_path = root / data_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute(): out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    original, original_meta = load_original_checkpoint(checkpoint_path, device)
    if original_meta["task"] != args.task:
        raise ValueError(f"Checkpoint task {original_meta['task']!r} doesn't match requested task {args.task!r}")

    captured = {}
    for split in ("train", "validation", "test"):
        captured[split] = capture_mlp_io(
            original, str(data_path), args.task, split, args.capture_batch_size, device,
            capture_block_io=False,
        )
    n_layers = captured["train"]["n_layers"]
    if args.model_type == "plt":
        model = PerLayerTranscoder(n_layers, d_model=64, n_features=args.n_features)
        write_span = 1
    else:
        write_span = args.write_span
        model = CrossLayerTranscoder(n_layers, d_model=64, n_features=args.n_features, write_span=write_span)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_x = captured["train"]["inputs"]
    train_y = captured["train"]["writes"]
    val_x = captured["validation"]["inputs"]
    val_y = captured["validation"]["writes"]
    train_ds = TensorDataset(*train_x, *train_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = total_mse = total_sparse = n_batches = 0.0
        for batch in train_loader:
            xs = [t.to(device) for t in batch[:n_layers]]
            ys = [t.to(device) for t in batch[n_layers:]]
            optimizer.zero_grad(set_to_none=True)
            preds, features = forward_model(model, xs)
            mse = torch.stack([nn.functional.mse_loss(p, y) for p, y in zip(preds, ys)]).mean()
            sparse = torch.stack([z.abs().mean() for z in features]).mean()
            loss = mse + args.sparsity * sparse
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()); total_mse += float(mse.detach()); total_sparse += float(sparse.detach()); n_batches += 1
        val_metrics, _ = evaluate(model, val_x, val_y, args.batch_size, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, n_batches),
            "train_mse": total_mse / max(1, n_batches),
            "train_feature_l1": total_sparse / max(1, n_batches),
            "validation_mse": val_metrics["mse"],
            "validation_relative_l2": val_metrics["relative_l2"],
        })
        print(f"epoch={epoch+1:03d} train_mse={history[-1]['train_mse']:.6g} val_mse={val_metrics['mse']:.6g} val_rel_l2={val_metrics['relative_l2']:.6g}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training didn't produce a checkpoint.")
    model.load_state_dict(best_state)
    test_metrics, _ = evaluate(model, captured["test"]["inputs"], captured["test"]["writes"], args.batch_size, device)
    val_metrics, _ = evaluate(model, val_x, val_y, args.batch_size, device)

    run_id = f"{args.task}_{args.model_type}_seed{seed}_features{args.n_features}_span{write_span}"
    ckpt_path = out_dir / f"{run_id}.pt"
    metadata = {
        "run_id": run_id,
        "status": "completed",
        "model_type": args.model_type,
        "architecture_label": "experimental_sparse_transcoder_baseline",
        "task": args.task,
        "ground_truth_hop_depth": 2 if args.task == "2hop" else 3,
        "base_checkpoint": str(checkpoint_path),
        "base_seed": original_meta.get("seed"),
        "seed": seed,
        "n_layers": n_layers,
        "d_model": 64,
        "n_features_per_layer": args.n_features,
        "sparsity_coefficient": args.sparsity,
        "write_span": write_span,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_examples": captured["train"]["n_examples"],
        "validation_examples": captured["validation"]["n_examples"],
        "test_examples": captured["test"]["n_examples"],
        "activation_target": "MLP write: encoder.layers.{i}.linear2 output reconstructed from encoder.layers.{i}.linear1 input",
        "replacement_interface": {
            "primary_target": "mlp_input_to_mlp_write",
            "mlp_input_hook": "encoder.layers.{i}.linear1 pre-forward input (after norm2 for norm_first=False)",
            "mlp_write_hook": "encoder.layers.{i}.linear2 forward output (before dropout2 and residual addition; eval mode disables dropout)",
            "diagnostic_block_input_hook": "encoder.layers.{i} forward pre-hook input (residual stream entering block)",
            "diagnostic_block_output_hook": "encoder.layers.{i} forward output (residual stream leaving block)",
            "block_boundary_hooks_used_for_training_targets": False,
        },
        "graph_handoff": {
            "feature_ids_convention": "replacement_type_layer_index_feature_index",
            "feature_activations_saved_in_checkpoint": False,
            "decoder_weights_are_attribution_edges": False,
            "note": "Person 3 must define/validate the attribution method before converting transcoder weights or activations to graph edge weights.",
        },
        "interpretation_caveat": "Experimental baseline; not confirmed identical to a paper/library CLT implementation. Confirm graph interface and architecture with team before scientific claims.",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata, "model_type": args.model_type}, ckpt_path)
    (out_dir / f"{run_id}.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / f"{run_id}_history.json").write_text(json.dumps(history, indent=2))

    fidelity_path = out_dir / "fidelity_table.csv"
    row = {
        "run_id": run_id,
        "status": "completed",
        "model_type": args.model_type,
        "task": args.task,
        "base_checkpoint": str(checkpoint_path),
        "base_seed": original_meta.get("seed"),
        "replacement_seed": seed,
        "n_features_per_layer": args.n_features,
        "sparsity_coefficient": args.sparsity,
        "write_span": write_span,
        "validation_mse": val_metrics["mse"],
        "validation_relative_l2": val_metrics["relative_l2"],
        "validation_explained_variance": val_metrics["explained_variance"],
        "test_mse": test_metrics["mse"],
        "test_relative_l2": test_metrics["relative_l2"],
        "test_explained_variance": test_metrics["explained_variance"],
        "test_active_feature_fraction": test_metrics["active_feature_fraction"],
        "test_mean_feature_activation": test_metrics["mean_feature_activation"],
        "checkpoint_path": str(ckpt_path),
    }
    # Per-layer metrics are important because a pooled score can hide a poor layer.
    val_preds_by_layer = evaluate(model, val_x, val_y, args.batch_size, device)[1]
    test_preds_by_layer = evaluate(model, captured["test"]["inputs"], captured["test"]["writes"], args.batch_size, device)[1]
    for layer_idx, (vp, vy, tp, ty) in enumerate(zip(val_preds_by_layer, val_y, test_preds_by_layer, captured["test"]["writes"])):
        layer_key = f"layer_{layer_idx + 1}"
        val_sq_err = float(((vp.float() - vy.float()) ** 2).sum())
        val_sq_y = float((vy.float() ** 2).sum())
        test_sq_err = float(((tp.float() - ty.float()) ** 2).sum())
        test_sq_y = float((ty.float() ** 2).sum())
        row[f"validation_{layer_key}_mse"] = val_sq_err / vy.numel()
        row[f"validation_{layer_key}_relative_l2"] = (val_sq_err / max(val_sq_y, 1e-12)) ** 0.5
        row[f"test_{layer_key}_mse"] = test_sq_err / ty.numel()
        row[f"test_{layer_key}_relative_l2"] = (test_sq_err / max(test_sq_y, 1e-12)) ** 0.5
    existing = []
    if fidelity_path.exists():
        with fidelity_path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    existing = [r for r in existing if r.get("run_id") != run_id]
    existing.append({k: str(v) for k, v in row.items()})
    with fidelity_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader(); writer.writerows(existing)

    print("\nSaved replacement checkpoint:", ckpt_path)
    print("Saved metrics table:", fidelity_path)
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    print("Original-model accuracy on test split:", captured["test"]["bit_accuracy"], "bit;", captured["test"]["exact_vector_accuracy"], "exact-vector")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--task", choices=["2hop", "3hop"], default="2hop")
    parser.add_argument("--model-type", choices=["plt", "clt"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Original transformer checkpoint")
    parser.add_argument("--data", default=None, help="Task CSV; defaults to data/boolean_<task>_improved.csv")
    parser.add_argument("--out-dir", default="results/person2")
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=1e-4)
    parser.add_argument("--write-span", type=int, default=2, help="CLT-style downstream write span; 1 means no cross-layer writes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--capture-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.n_features < 1: parser.error("--n-features must be positive")
    if args.epochs < 1: parser.error("--epochs must be positive")
    if args.sparsity < 0: parser.error("--sparsity must be non-negative")
    train(args)


if __name__ == "__main__":
    main()
