"""Intermediate-variable recovery for one SIIT checkpoint and its PLT/CLT replacements.

2hop: u = XOR(a, b).  3hop: u = XOR(a, b) and v = AND(u, c).
Probes are fitted on TRAIN only. Control rows (raw MLP input, residual stream)
and a majority-class baseline are reported so 100% scores can be interpreted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .activation_cache import capture_mlp_io
from .checkpoint_io import load_replacement_checkpoint
from .model_adapter import load_original_checkpoint
from .siit_utils import (
    default_siit_sites,
    flatten_recovery,
    load_intermediate_labels,
    probe_representation,
)

SPLITS = ("train", "validation", "test")


def replacement_features(model, inputs, device, batch_size: int = 256):
    parts = [[] for _ in inputs]
    with torch.no_grad():
        for start in range(0, inputs[0].shape[0], batch_size):
            batch = [x[start:start + batch_size].to(device) for x in inputs]
            _, feats = model(batch)
            for layer, z in enumerate(feats):
                parts[layer].append(z.detach().cpu())
    return [torch.cat(p, dim=0).numpy() for p in parts]


def evaluate(base_checkpoint: Path, data_path: Path, plt_path: Path, clt_path: Path, device="cpu") -> dict:
    device = torch.device(device)
    original, saved = load_original_checkpoint(base_checkpoint, device)
    task = saved["task"]
    n_layers = int(saved["layers"])
    variables = ["u"] if task == "2hop" else ["u", "v"]
    sites = saved.get("siit_sites") or default_siit_sites(task, n_layers)
    designated_layer = {s["var"]: int(s["layer"]) for s in sites}

    captured = {
        split: capture_mlp_io(original, str(data_path), task, split, 256, device, capture_block_io=True)
        for split in SPLITS
    }
    labels = {var: {s: load_intermediate_labels(data_path, task, s, var) for s in SPLITS} for var in variables}
    for var in variables:
        for s in SPLITS:
            if len(labels[var][s]) != captured[s]["n_examples"]:
                raise ValueError(f"{var}/{s}: label count != activation count (row alignment broken)")

    reps = {}
    for site in sites:
        lo, hi, layer = int(site["start"]), int(site["end"]), int(site["layer"])
        reps[f"siit_site_{site['var']}_L{layer}"] = (
            "siit_site", layer,
            {s: captured[s]["block_outputs"][layer][:, :, lo:hi].numpy() for s in SPLITS},
            [site["var"]],
        )
    for i in range(n_layers):
        reps[f"mlp_input_L{i}"] = (
            "control_mlp_input", i, {s: captured[s]["inputs"][i].numpy() for s in SPLITS}, variables)
        reps[f"residual_after_block_L{i}"] = (
            "control_residual", i, {s: captured[s]["block_outputs"][i].numpy() for s in SPLITS}, variables)

    replacement_info = {}
    for kind, path in (("plt", plt_path), ("clt", clt_path)):
        model, meta = load_replacement_checkpoint(path, device)
        if int(meta["n_layers"]) != n_layers:
            raise ValueError(f"{kind}: replacement has {meta['n_layers']} layers, base has {n_layers}")
        feats = {s: replacement_features(model, captured[s]["inputs"], device) for s in SPLITS}
        for i in range(n_layers):
            reps[f"{kind}_L{i}"] = (kind, i, {s: feats[s][i] for s in SPLITS}, variables)
        replacement_info[kind] = {
            "checkpoint": str(path),
            "write_span": meta.get("write_span"),
            "validation_relative_l2": meta["validation_metrics"]["relative_l2"],
        }

    results = []
    for var in variables:
        for name, (kind, layer, arrays, applicable) in reps.items():
            if var not in applicable:
                continue
            probe = probe_representation(arrays, labels[var])
            results.append({
                "variable": var,
                "representation": name,
                "kind": kind,
                "layer": layer,
                "designated_layer": layer == designated_layer.get(var),
                "n_dims_per_token": int(arrays["train"].shape[-1]),
                **probe,
            })
            print(f"  {var} {name:<28} val={probe['mean_validation_accuracy']:.4f} "
                  f"test={probe['mean_test_accuracy']:.4f} "
                  f"majority={probe['mean_test_majority_baseline_accuracy']:.4f}")

    return {
        "task": task,
        "n_layers": n_layers,
        "base_checkpoint": str(base_checkpoint),
        "base_seed": saved.get("seed"),
        "siit_sites": sites,
        "original_model_test_bit_accuracy": captured["test"]["bit_accuracy"],
        "original_model_test_exact_vector_accuracy": captured["test"]["exact_vector_accuracy"],
        "train_examples": captured["train"]["n_examples"],
        "validation_examples": captured["validation"]["n_examples"],
        "test_examples": captured["test"]["n_examples"],
        "replacement_models": replacement_info,
        "results": results,
        "interpretation_warning": (
            "Probes are fitted on train only. High probe accuracy shows a variable is linearly "
            "decodable; it does not show a feature is causally faithful or that any single feature "
            "implements the variable. Compare against the control rows and the majority baseline."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plt", required=True)
    parser.add_argument("--clt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = evaluate(Path(args.checkpoint), Path(args.data), Path(args.plt), Path(args.clt), args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()