from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pandas as pd
import torch
from src.replacements.activation_cache import capture_mlp_io
from src.replacements.checkpoint_io import load_replacement_checkpoint
from src.replacements.model_adapter import load_original_checkpoint, load_task_split

def get_encoder(replacement, model_type, layer):
    return replacement.encoders[layer] if model_type == "clt" else replacement.transcoders[layer].encoder

def get_decoder(replacement, model_type, source, target):
    if model_type == "clt":
        return replacement.decoders[f"{source}_{target}"]
    if source != target:
        raise ValueError("PLT supports same-layer writes only.")
    return replacement.transcoders[source].decoder

def get_targets(replacement, model_type, source):
    if model_type == "plt":
        return [source]
    return sorted(int(k.split("_")[1]) for k in replacement.decoders if int(k.split("_")[0]) == source)

def main():
    parser = argparse.ArgumentParser(description="Ablate one replacement feature and inject its decoder-write delta.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--model-type", choices=["plt", "clt"], required=True)
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--feature-id", type=int, default=22)
    parser.add_argument("--target-layers", default="all", help="Comma-separated destination layers, or all")
    parser.add_argument("--write-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/person3/causal_pilot")
    args = parser.parse_args()
    if not math.isfinite(args.write_scale) or args.write_scale <= 0:
        parser.error("--write-scale must be finite and greater than zero.")

    device = torch.device("cpu")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    original, checkpoint_meta = load_original_checkpoint(args.checkpoint, device=device)
    replacement, _ = load_replacement_checkpoint(args.replacement)
    replacement = replacement.to(device).eval()
    task = checkpoint_meta["task"]

    cache = capture_mlp_io(original, args.data, task, "test", batch_size=128, device=device)
    x, y = load_task_split(args.data, task, "test")
    x, y = x.to(device), y.to(device)
    n_layers = len(original.encoder.layers)
    source, feature_id = args.source_layer, args.feature_id
    if not 0 <= source < n_layers:
        raise ValueError(f"Invalid source layer: {source}")
    if len(cache["inputs"]) != n_layers:
        raise RuntimeError("Unexpected number of captured layers.")

    baseline_features = []
    with torch.no_grad():
        for layer in range(n_layers):
            enc = get_encoder(replacement, args.model_type, layer)
            baseline_features.append(torch.relu(enc(cache["inputs"][layer])))
    source_features = baseline_features[source]
    if not 0 <= feature_id < source_features.shape[-1]:
        raise ValueError(f"Feature {feature_id} out of range; available={source_features.shape[-1]}")
    ablated = source_features.clone()
    ablated[..., feature_id] = 0.0

    targets = get_targets(replacement, args.model_type, source)
    if args.target_layers.lower() != "all":
        selected = [int(v.strip()) for v in args.target_layers.split(",") if v.strip()]
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate destination layers supplied.")
        invalid = set(selected) - set(targets)
        if invalid:
            raise ValueError(f"Invalid destinations {sorted(invalid)}; available={targets}")
        targets = selected
    if not targets:
        raise ValueError("No destination layers selected.")

    deltas = {}
    with torch.no_grad():
        for target in targets:
            dec = get_decoder(replacement, args.model_type, source, target)
            delta = dec(ablated) - dec(source_features)
            deltas[target] = delta * args.write_scale

    with torch.no_grad():
        baseline_logits = original(x)

    modified_inputs = [[] for _ in range(n_layers)]
    handles = []
    for layer_idx, layer in enumerate(original.encoder.layers):
        def input_hook(module, inputs, idx=layer_idx):
            modified_inputs[idx].append(inputs[0].detach().cpu())
        handles.append(layer.linear1.register_forward_pre_hook(input_hook))
        if layer_idx in deltas:
            delta = deltas[layer_idx]
            def write_hook(module, inputs, output, d=delta):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("Expected tensor output from linear2.")
                return output + d.to(device=output.device, dtype=output.dtype)
            handles.append(layer.linear2.register_forward_hook(write_hook))
    try:
        with torch.no_grad():
            intervened_logits = original(x)
    finally:
        for h in handles:
            h.remove()
    modified_inputs = [torch.cat(v, dim=0) for v in modified_inputs]

    layer_rows, individual_rows = [], []
    with torch.no_grad():
        for layer in range(n_layers):
            enc = get_encoder(replacement, args.model_type, layer)
            changed = torch.relu(enc(modified_inputs[layer]))
            diff = (changed - baseline_features[layer]).abs()
            layer_rows.append({
                "layer": layer,
                "mean_absolute_feature_change": diff.mean().item(),
                "max_absolute_feature_change": diff.max().item(),
                "changed_feature_fraction": (diff > 1e-6).float().mean().item(),
            })
            for target_feature_id in range(diff.shape[-1]):
                fd = diff[..., target_feature_id]
                individual_rows.append({
                    "source_layer": source, "source_feature_id": feature_id,
                    "target_layer": layer, "target_feature_id": target_feature_id,
                    "mean_absolute_change": fd.mean().item(),
                    "max_absolute_change": fd.max().item(),
                    "changed_fraction": (fd > 1e-6).float().mean().item(),
                })

    logit_diff = (intervened_logits - baseline_logits).abs()
    base_pred = (torch.sigmoid(baseline_logits) >= .5).float()
    int_pred = (torch.sigmoid(intervened_logits) >= .5).float()
    pred_change = (base_pred != int_pred).float().mean().item()
    base_acc = (base_pred == y).float().mean().item()
    int_acc = (int_pred == y).float().mean().item()

    write_rows = [{
        "source_layer": source, "source_feature_id": feature_id, "target_layer": target,
        "write_scale": args.write_scale,
        "mean_absolute_write_change": delta.abs().mean().item(),
        "max_absolute_write_change": delta.abs().max().item(),
    } for target, delta in deltas.items()]

    feature_df = pd.DataFrame(layer_rows)
    individual_df = pd.DataFrame(individual_rows)
    write_df = pd.DataFrame(write_rows)
    seed = checkpoint_meta.get("seed", "unknown")
    scale_label = f"{args.write_scale:g}".replace(".", "p")
    prefix = f"{task}_seed{seed}_{args.model_type}_L{source}_F{feature_id}_targets_{'-'.join(map(str, targets))}_scale_{scale_label}"
    feature_csv = outdir / f"{prefix}_downstream_features.csv"
    individual_csv = outdir / f"{prefix}_individual_feature_changes.csv"
    write_csv = outdir / f"{prefix}_write_changes.csv"
    report_json = outdir / f"{prefix}_intervention.json"
    feature_df.to_csv(feature_csv, index=False)
    individual_df.to_csv(individual_csv, index=False)
    write_df.to_csv(write_csv, index=False)
    report = {
        "experiment": "single_feature_decoder_write_intervention",
        "model_type": args.model_type, "source_layer": source,
        "source_feature_id": feature_id, "test_examples": len(x),
        "direct_destination_layers": targets, "write_scale": args.write_scale,
        "feature_effects": layer_rows,
        "individual_feature_changes_csv": str(individual_csv),
        "write_effects": write_rows,
        "baseline_bit_accuracy": base_acc, "intervened_bit_accuracy": int_acc,
        "prediction_change_fraction": pred_change,
        "mean_absolute_logit_change": logit_diff.mean().item(),
        "max_absolute_logit_change": logit_diff.max().item(),
        "causal_interpretation": "Controlled transcoder-derived write intervention; not proof that the feature is an intrinsic causal variable of the original Transformer.",
        "graph_edges_are_causal": False,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nINTERVENTION COMPLETED")
    print("---------------------")
    print(f"Model: {args.model_type.upper()}")
    print(f"Intervened feature: L{source}, F{feature_id}")
    print(f"Direct destination layers: {targets}")
    print(f"Write scale: {args.write_scale}")
    print("\nLayer-level downstream feature changes:")
    print(feature_df.to_string(index=False))
    print("\nIndividual target-feature changes:")
    print(individual_df.to_string(index=False))
    print("\nPrediction change fraction:", pred_change)
    print("Baseline bit accuracy:", base_acc)
    print("Intervened bit accuracy:", int_acc)
    print("\nSaved:", feature_csv, individual_csv, write_csv, report_json, sep="\n")

if __name__ == "__main__":
    main()
