import json
from pathlib import Path
import pandas as pd

folders = {
    "clt": Path("results/person3/causal_matched"),
    "plt": Path("results/person3/causal_multiseed"),
}

rows = []

for model, folder in folders.items():
    for path in sorted(folder.glob("*_intervention.json")):
        r = json.loads(path.read_text(encoding="utf-8"))

        if r["model_type"] != model:
            continue

        task = path.name.split("_seed")[0]
        seed = int(path.name.split("_seed")[1].split("_")[0])

        for e in r["feature_effects"]:
            if e["layer"] == 0:
                continue

            rows.append({
                "task": task,
                "seed": seed,
                "model": model,
                "layer": e["layer"],
                "mean_feature_change": e["mean_absolute_feature_change"],
                "changed_fraction": e["changed_feature_fraction"],
                "prediction_change": r["prediction_change_fraction"],
                "accuracy": r["intervened_bit_accuracy"],
                "logit_change": r["mean_absolute_logit_change"]
            })

df = pd.DataFrame(rows)
df.to_csv("results/person3/causal_matched/plt_clt_matched_all_seeds.csv", index=False)

summary = df.groupby(["task", "model", "layer"]).agg(
    seeds=("seed", "nunique"),
    mean_feature_change=("mean_feature_change", "mean"),
    std_feature_change=("mean_feature_change", "std"),
    mean_changed_fraction=("changed_fraction", "mean"),
    mean_logit_change=("logit_change", "mean"),
    mean_accuracy=("accuracy", "mean")
).reset_index()

summary.to_csv("results/person3/causal_matched/plt_clt_matched_summary.csv", index=False)

print(summary.to_string(index=False))
print("\nTotal model runs:", df[["task", "seed", "model"]].drop_duplicates().shape[0])
