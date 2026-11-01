
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

out = Path("results/person3/final_figures")
out.mkdir(parents=True, exist_ok=True)

# Load matched causal intervention results
df = pd.read_csv(
    "results/person3/causal_matched/plt_clt_matched_all_seeds.csv"
)

# Figure 1: Mean downstream feature changes
summary = df.groupby(["task", "model", "layer"])[
    "mean_feature_change"
].mean().reset_index()

summary["label"] = (
    summary["task"] + " L" + summary["layer"].astype(str)
)

pivot = summary.pivot(
    index="label", columns="model", values="mean_feature_change"
)

ax = pivot.plot(kind="bar", figsize=(10, 6))
ax.set_title("PLT vs CLT: Downstream Feature Changes")
ax.set_xlabel("Task and destination layer")
ax.set_ylabel("Mean absolute feature change")
ax.legend(title="Replacement")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(out / "01_plt_vs_clt_feature_change.png", dpi=300)
plt.close()

# Figure 2: Seed-wise variation
fig, ax = plt.subplots(figsize=(10, 6))

for model in ["plt", "clt"]:
    subset = df[df["model"] == model]
    for (task, layer), group in subset.groupby(["task", "layer"]):
        label = f"{task} L{layer} {model.upper()}"
        ax.plot(
            group["seed"],
            group["mean_feature_change"],
            marker="o",
            label=label
        )

ax.set_title("Seed-wise Variation in Feature Changes")
ax.set_xlabel("Random seed")
ax.set_ylabel("Mean absolute feature change")
ax.set_xticks(range(5))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out / "02_seed_wise_variation.png", dpi=300)
plt.close()


# Figure 3: Attribution graph comparison - grouped bars with standard deviation
graph_file = Path(
    "results/person3/attribution_multiseed/mean_std_across_seeds.csv"
)

if graph_file.exists():
    graph = pd.read_csv(graph_file)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, task in zip(axes, ["2hop", "3hop"]):
        task_data = graph[graph["Task"] == task].copy()

        connections = sorted(task_data["Connection"].unique())
        x = np.arange(len(connections))
        width = 0.35

        for i, model in enumerate(["plt", "clt"]):
            model_data = (
                task_data[task_data["Model"] == model]
                .set_index("Connection")
            )

            means = [
                model_data.loc[c, "MeanEffect"]
                if c in model_data.index else np.nan
                for c in connections
            ]

            stds = [
                model_data.loc[c, "StdEffect"]
                if c in model_data.index else np.nan
                for c in connections
            ]

            positions = x + (i - 0.5) * width

            ax.bar(
                positions,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=model.upper()
            )

        ax.set_title(f"{task.upper()} Attribution Connections")
        ax.set_xlabel("Source-to-destination connection")
        ax.set_ylabel("Mean decoder contribution effect")
        ax.set_xticks(x)
        ax.set_xticklabels(connections, rotation=35, ha="right")
        ax.legend(title="Replacement", loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "PLT vs CLT Attribution Effects Across Five Seeds",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(
        out / "03_attribution_graph_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

else:
    print("Attribution summary CSV not found; skipping figure 3.")


print("\nVISUALIZATIONS CREATED")
for file in sorted(out.glob("*.png")):
    print(file)
