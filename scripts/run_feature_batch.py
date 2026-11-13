"""Run Layer-0 decoder-write interventions for many selected features at once.

Reads an individual-feature analysis CSV (from evaluate_individual_features),
picks features in --source-layer (default 0) whose validation accuracy is
>= --min-val-acc, runs run_person3_causal_intervention for each PLT and CLT
feature (write injected only at the source layer, scale 1.0), and writes one
summary CSV with the change measured at every layer.

Because every run uses scale 1.0, PLT and CLT are compared fairly using
"layer-k change per unit of write" = layer-k mean change / mean write size.
"next_layer_*" columns refer to the layer directly after the source layer.
(The seed-0 runs showed the response is close to proportional to the write
scale, which is what makes this normalisation reasonable.)

Example (run from the repository root):
    python scripts/run_feature_batch.py --task 2hop --seed 1 ^
        --analysis results/person3/individual_feature_analysis_seed1.csv
    python scripts/run_feature_batch.py --task 3hop --seed 0 --source-layer 1 ^
        --analysis results/person3/3hop_v_features_seed0.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

TASKS = {
    # task: (folder prefix, data file, CLT write span)
    "2hop": ("2hop_L2_siit_k1_seed{seed}", "data/boolean_2hop_improved.csv", 2),
    "3hop": ("3hop_L3_siit_k1_seed{seed}", "data/boolean_3hop_improved.csv", 3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--analysis", required=True, help="CSV from evaluate_individual_features")
    parser.add_argument("--min-val-acc", type=float, default=1.0,
                        help="Select Layer-0 features with validation accuracy >= this (default 1.0)")
    parser.add_argument("--max-per-model", type=int, default=None,
                        help="Optional cap on features per model (best validation first)")
    parser.add_argument("--source-layer", type=int, default=0,
                        help="Layer whose features are ablated (0 for u; 1 for v in 3-hop)")
    parser.add_argument("--target-layers", default=None,
                        help="Comma-separated destination layers for the write (default: the source layer only). "
                             "CLT only for anything other than the source layer, e.g. 0,1 to switch on the 0->1 shortcut")
    parser.add_argument("--models", default="plt,clt",
                        help="Which models to run: plt,clt (default), plt, or clt")
    parser.add_argument("--output-dir", default=None,
                        help="Default: results/person3/causal_batch/<task>_seed<seed> "
                             "(with _L<k> appended when --source-layer is not 0)")
    args = parser.parse_args()

    folder, data, span = TASKS[args.task]
    folder = folder.format(seed=args.seed)
    checkpoint = f"checkpoints/{folder}.pt"
    replacements = {
        "plt": f"results/person2/siit/{folder}/{args.task}_plt_seed0_features128_span1.pt",
        "clt": f"results/person2/siit/{folder}/{args.task}_clt_seed0_features128_span{span}.pt",
    }
    src = args.source_layer
    targets = [int(t) for t in (args.target_layers or str(src)).split(",") if t.strip()]
    target_arg = ",".join(map(str, targets))
    target_tag = "-".join(map(str, targets))
    suffix = "" if src == 0 else f"_L{src}"
    if targets != [src]:
        suffix += f"_T{target_tag}"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if targets != [src] and "plt" in models:
        sys.exit("PLT can only write to its own layer; use --models clt with --target-layers.")
    outdir = Path(args.output_dir or f"results/person3/causal_batch/{args.task}_seed{args.seed}{suffix}")
    outdir.mkdir(parents=True, exist_ok=True)

    for path in [checkpoint, data, args.analysis, *replacements.values()]:
        if not Path(path).is_file():
            sys.exit(f"Missing file: {path}")

    analysis = pd.read_csv(args.analysis)
    layer0 = analysis[(analysis["layer"] == src) & (analysis["validation_accuracy"] >= args.min_val_acc)]
    layer0 = layer0.sort_values(["model_type", "validation_accuracy", "feature_id"],
                                ascending=[True, False, True])
    layer0 = layer0[layer0["model_type"].isin(models)]
    if args.max_per_model:
        layer0 = layer0.groupby("model_type").head(args.max_per_model)
    if layer0.empty:
        sys.exit(f"No Layer-{src} features pass --min-val-acc; try a lower value such as 0.99.")

    print(f"Selected {len(layer0)} features:")
    for model, group in layer0.groupby("model_type"):
        print(f"  {model.upper()}: {group['feature_id'].tolist()}")

    rows = []
    for i, rec in enumerate(layer0.itertuples(index=False), start=1):
        model, fid = rec.model_type, int(rec.feature_id)
        pattern = f"*_{model}_L{src}_F{fid}_targets_{target_tag}_scale_1_intervention.json"
        existing = sorted(outdir.glob(pattern))
        if existing:
            print(f"[{i}/{len(layer0)}] {model.upper()} F{fid}: already done, reusing")
        else:
            print(f"[{i}/{len(layer0)}] {model.upper()} F{fid}: running...", flush=True)
            cmd = [sys.executable, "-m", "src.metrics.run_person3_causal_intervention",
                   "--checkpoint", checkpoint, "--data", data,
                   "--replacement", replacements[model], "--model-type", model,
                   "--source-layer", str(src), "--feature-id", str(fid),
                   "--target-layers", target_arg, "--write-scale", "1.0",
                   "--output-dir", str(outdir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:])
                sys.exit(f"Intervention failed for {model.upper()} F{fid}")
            existing = sorted(outdir.glob(pattern))
            if not existing:
                sys.exit(f"Run finished but no JSON found matching {pattern} in {outdir}")

        report = json.loads(existing[0].read_text(encoding="utf-8"))
        effects = {e["layer"]: e for e in report["feature_effects"]}
        writes = {w["target_layer"]: w["mean_absolute_write_change"] for w in report["write_effects"]}
        write = writes.get(src, report["write_effects"][0]["mean_absolute_write_change"])
        row = {
            "task": args.task, "seed": args.seed, "source_layer": src, "target_layers": target_tag,
            "model_type": model, "feature_id": fid,
            "validation_accuracy": rec.validation_accuracy, "test_accuracy": rec.test_accuracy,
            "mean_write_size": write,
            "total_write_size": sum(writes.values()),
        }
        for layer in sorted(effects):
            e = effects[layer]
            mean_change = e["mean_absolute_feature_change"]
            row[f"layer{layer}_mean_change"] = mean_change
            row[f"layer{layer}_max_change"] = e["max_absolute_feature_change"]
            row[f"layer{layer}_changed_fraction"] = e["changed_feature_fraction"]
            row[f"layer{layer}_change_per_unit_write"] = mean_change / write if write else None
        nxt = src + 1
        row["next_layer_mean_change"] = row.get(f"layer{nxt}_mean_change")
        row["next_layer_change_per_unit_write"] = row.get(f"layer{nxt}_change_per_unit_write")
        row["mean_logit_change"] = report.get("mean_absolute_logit_change")
        row["logit_change_per_unit_write"] = (row["mean_logit_change"] / write
                                              if write and row["mean_logit_change"] is not None else None)
        row["prediction_change_fraction"] = report["prediction_change_fraction"]
        row["intervened_bit_accuracy"] = report["intervened_bit_accuracy"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = outdir / f"{args.task}_seed{args.seed}{suffix}_batch_summary.csv"
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    print(f"\nSaved per-feature results: {summary_path}\n")
    downstream = [c for c in summary.columns
                  if c.endswith("_change_per_unit_write") and c.startswith("layer")
                  and int(c[5:c.index("_")]) > src]
    downstream = downstream + ["logit_change_per_unit_write"]
    print(summary[["model_type", "feature_id", "mean_write_size", *downstream,
                   "prediction_change_fraction"]].to_string(index=False))
    for col in downstream:
        label = "logit" if col.startswith("logit") else col.split('_')[0].replace('layer', 'layer-')
        print(f"\nPLT vs CLT ({label} change per unit write):")
        print(summary.groupby("model_type")[col]
              .agg(["count", "mean", "std", "min", "max"]).to_string())
    print("\nPLT vs CLT (raw write size):")
    print(summary.groupby("model_type")["mean_write_size"]
          .agg(["count", "mean", "std", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
