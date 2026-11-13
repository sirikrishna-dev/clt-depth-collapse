"""Shortcut test: does switching on a CLT feature's cross-layer writes restore
the downstream effect that is missing when it writes only to its own layer?

For each seed it reads
  * the normal batch runs (PLT and CLT, write only to the source layer), and
  * the CLT "shortcut" batch runs (same CLT features, written to --targets),
then compares, per unit of the source-layer write:
  PLT (no shortcuts possible)  vs  CLT without shortcut  vs  CLT with shortcut.

Example (3-hop, Layer-0 features, 0->1 shortcut switched on):
    python scripts/shortcut_compare.py --task 3hop --source-layer 0 --targets 0,1

Note on what can be seen where: layer-k features are read from the INPUT of
layer k's MLP. A shortcut that writes into layer k's MLP OUTPUT is therefore
invisible to layer-k features, but visible to every later layer and to the
final logits. For --source-layer 0 --targets 0,1 the 0->1 shortcut is visible
at layer 2 and in the logits. For --source-layer 1 --targets 1,2 it is visible
only in the logits.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(root: Path, pattern: str) -> pd.DataFrame:
    rx = re.compile(pattern)
    files = sorted(f for f in root.glob("*/*_batch_summary.csv") if rx.fullmatch(f.parent.name))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_unit(df, metric):
    if metric == "logit":
        return df["mean_logit_change"] / df["mean_write_size"]
    return df[f"{metric}_mean_change"] / df["mean_write_size"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["2hop", "3hop"], default="3hop")
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--targets", default="0,1", help="Shortcut target layers used for the CLT runs")
    parser.add_argument("--root", default="results/person3/causal_batch")
    args = parser.parse_args()

    src = args.source_layer
    tag = "-".join(t.strip() for t in args.targets.split(","))
    lsuf = "" if src == 0 else f"_L{src}"
    root = Path(args.root)
    base = load(root, rf"{args.task}_seed\d+{lsuf}")
    short = load(root, rf"{args.task}_seed\d+{lsuf}_T{re.escape(tag)}")
    if base.empty or short.empty:
        raise SystemExit("Missing runs: need both the normal batch and the shortcut batch (--models clt --target-layers ...).")

    n_layers = 3 if args.task == "3hop" else 2
    metrics = [f"layer{k}" for k in range(src + 1, n_layers)
               if f"layer{k}_mean_change" in short.columns] + ["logit"]
    if "mean_logit_change" not in base.columns or base["mean_logit_change"].isna().all():
        raise SystemExit("Normal batch summaries have no logit column. Rerun the normal batch command for each seed "
                         "with the updated run_feature_batch.py (it reuses saved runs and only rebuilds the summary).")

    pd.set_option("display.width", 200)
    print(f"Task {args.task}, source layer {src}, CLT shortcut targets {tag}")
    print("Values are change per unit of the source-layer write.\n")

    for metric in metrics:
        print(f"=== {metric} ===")
        rows, feat_rows = [], []
        for seed in sorted(short["seed"].unique()):
            b = base[base.seed == seed]
            plt_v = per_unit(b[b.model_type == "plt"], metric)
            clt_b = b[b.model_type == "clt"].set_index("feature_id")
            clt_s = short[(short.seed == seed) & (short.model_type == "clt")].set_index("feature_id")
            common = clt_b.index.intersection(clt_s.index)
            if len(common) == 0 or plt_v.empty:
                continue
            no_sc = per_unit(clt_b.loc[common], metric)
            with_sc = per_unit(clt_s.loc[common], metric)
            rows.append({"seed": seed, "n_plt": len(plt_v), "n_clt": len(common),
                         "plt": plt_v.mean(), "clt_no_shortcut": no_sc.mean(),
                         "clt_with_shortcut": with_sc.mean(),
                         "gap_plt_minus_clt_no_shortcut": plt_v.mean() - no_sc.mean(),
                         "gap_plt_minus_clt_with_shortcut": plt_v.mean() - with_sc.mean()})
            feat_rows.append(pd.DataFrame({"seed": seed, "no": no_sc.values, "with": with_sc.values}))
        table = pd.DataFrame(rows)
        if table.empty:
            print("   no overlapping runs\n")
            continue
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        feats = pd.concat(feat_rows)
        d = feats["with"] - feats["no"]
        if np.allclose(d, 0, atol=1e-9):
            print("   Shortcut makes no difference here (expected if the shortcut lands after this measurement point).")
        else:
            t, p = stats.ttest_rel(feats["with"], feats["no"])
            print(f"   CLT features, with vs without shortcut (paired over {len(feats)} features): "
                  f"mean change {d.mean():+.4f}, t = {t:.2f}, p = {p:.3g}")
        if len(table) >= 2:
            for col in ["gap_plt_minus_clt_no_shortcut", "gap_plt_minus_clt_with_shortcut"]:
                t, p = stats.ttest_1samp(table[col], 0.0)
                print(f"   Seed-level {col}: mean {table[col].mean():+.4f}, "
                      f"positive in {(table[col] > 0).sum()}/{len(table)} seeds, t = {t:.2f}, p = {p:.3f}")
        print()

    out = root / f"{args.task}{lsuf}_shortcut_T{tag}_features.csv"
    short.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
