import json
from pathlib import Path
import pandas as pd

folder = Path("results/person3/causal_matched")
rows = []

for path in sorted(folder.glob("*_intervention.json")):
    r = json.loads(path.read_text(encoding="utf-8"))
    task = path.name.split("_seed")[0]
    seed = int(path.name.split("_seed")[1].split("_")[0])

    for e in r["feature_effects"]:
        rows.append({
            "task": task,
            "seed": seed,
            "model": r["model_type"],
            "layer": e["layer"],
            "mean_feature_change": e["mean_absolute_feature_change"],
            "changed_fraction": e["changed_feature_fraction"],
            "prediction_change": r["prediction_change_fraction"],
            "accuracy": r["intervened_bit_accuracy"],
            "logit_change": r["mean_absolute_logit_change"]
        })

df = pd.DataFrame(rows)
df.to_csv(folder / "matched_clt_all_results.csv", index=False)

summary = df.groupby(["task", "model", "layer"]).agg(
    seeds=("seed", "nunique"),
    mean_feature_change=("mean_feature_change", "mean"),
    std_feature_change=("mean_feature_change", "std"),
    mean_changed_fraction=("changed_fraction", "mean"),
    mean_prediction_change=("prediction_change", "mean"),
    mean_accuracy=("accuracy", "mean"),
    mean_logit_change=("logit_change", "mean")
).reset_index()

summary.to_csv(folder / "matched_clt_summary.csv", index=False)
print(summary.to_string(index=False))
print("\nReports:", len(list(folder.glob("*_intervention.json"))))
