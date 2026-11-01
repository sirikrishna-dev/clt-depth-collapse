import json
from pathlib import Path
import pandas as pd

folder = Path("results/person3/causal_multiseed")
records = []

for path in sorted(folder.glob("*_intervention.json")):
    report = json.loads(path.read_text(encoding="utf-8"))

    task = path.name.split("_seed")[0]
    seed = int(path.name.split("_seed")[1].split("_")[0])

    for effect in report["feature_effects"]:
        records.append({
            "task": task,
            "seed": seed,
            "model": report["model_type"],
            "source_layer": report["source_layer"],
            "feature_id": report["feature_id"],
            "layer": effect["layer"],
            "mean_absolute_feature_change": effect["mean_absolute_feature_change"],
            "max_absolute_feature_change": effect["max_absolute_feature_change"],
            "changed_feature_fraction": effect["changed_feature_fraction"],
            "prediction_change_fraction": report["prediction_change_fraction"],
            "baseline_accuracy": report["baseline_bit_accuracy"],
            "intervened_accuracy": report["intervened_bit_accuracy"],
            "mean_absolute_logit_change": report["mean_absolute_logit_change"],
        })

df = pd.DataFrame(records)
df.to_csv(folder / "causal_all_seed_results.csv", index=False)

summary = df.groupby(
    ["task", "model", "layer"]
).agg(
    seeds=("seed", "nunique"),
    mean_feature_change=("mean_absolute_feature_change", "mean"),
    std_feature_change=("mean_absolute_feature_change", "std"),
    mean_changed_fraction=("changed_feature_fraction", "mean"),
    mean_prediction_change=("prediction_change_fraction", "mean"),
    mean_accuracy=("intervened_accuracy", "mean"),
    mean_logit_change=("mean_absolute_logit_change", "mean"),
).reset_index()

summary.to_csv(folder / "causal_mean_std_summary.csv", index=False)

print("\nUNIQUE MODEL RUNS:", df[["task", "seed", "model"]].drop_duplicates().shape[0])
print("INTERVENTION REPORTS:", len(list(folder.glob("*_intervention.json"))))
print("\nMEAN AND STANDARD DEVIATION:")
print(summary.to_string(index=False))
print("\nSaved:")
print(folder / "causal_all_seed_results.csv")
print(folder / "causal_mean_std_summary.csv")
