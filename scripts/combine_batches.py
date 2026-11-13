"""Combine all per-seed batch summaries for one task and compare PLT vs CLT.

Reads results/person3/causal_batch/<task>_seed*/<task>_seed*_batch_summary.csv
and reports, for the chosen metric (default: layer1_change_per_unit_write):

  1. Per-seed table: mean for PLT and CLT, difference, Welch t-test p-value.
  2. Seed-level paired test: one PLT mean and one CLT mean per seed, paired
     t-test and Wilcoxon signed-rank across seeds. This is the main test,
     because features from the same seed are not independent.
  3. Pooled (all features together) Welch t-test and Mann-Whitney U, reported
     for completeness only - they ignore the seed grouping.
  4. Prediction-change check across every run.
  5. Push-size-adjusted comparison. Per-unit effects shrink as the injected
     write gets bigger, so a model that happens to push harder looks weaker
     per unit. This section fits, by ordinary least squares,
         metric ~ CLT + log(mean_write_size) [+ one intercept per seed]
     and reports the CLT coefficient: the PLT-vs-CLT difference at the SAME
     push size. Per-seed adjusted differences are also tested across seeds.

Example (run from the repository root):
    python scripts/combine_batches.py --task 2hop
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def ols(y, X, names):
    """Plain OLS with classical standard errors. Returns {name: (coef, se, t, p)}."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n <= k:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    out = {}
    for i, name in enumerate(names):
        t = beta[i] / se[i] if se[i] > 0 else float("nan")
        p = 2 * stats.t.sf(abs(t), dof) if np.isfinite(t) else float("nan")
        out[name] = (beta[i], se[i], t, p)
    return out


def adjusted_section(data, metric):
    print("\n5) Push-size-adjusted comparison (same push size for PLT and CLT)")
    if "mean_write_size" not in data.columns:
        print("   mean_write_size column missing; skipped.")
        return None
    d = data[(data["mean_write_size"] > 0)].copy()
    d["log_write"] = np.log(d["mean_write_size"])
    d["is_clt"] = (d["model_type"] == "clt").astype(float)
    r = np.corrcoef(d["log_write"], d[metric])[0, 1]
    print(f"   Correlation between log(push size) and {metric}: r = {r:.3f}")

    rows = []
    for seed, g in d.groupby("seed"):
        if g["is_clt"].nunique() < 2 or len(g) < 4:
            continue
        res = ols(g[metric], np.column_stack([np.ones(len(g)), g["is_clt"], g["log_write"]]),
                  ["intercept", "clt", "log_write"])
        if res:
            c = res["clt"]
            rows.append({"seed": seed, "n": len(g),
                         "adjusted_plt_minus_clt": -c[0], "se": c[1], "p": c[3],
                         "log_write_slope": res["log_write"][0]})
    per_seed = pd.DataFrame(rows)
    if not per_seed.empty:
        print("   Per seed (difference PLT - CLT at equal push size):")
        print(per_seed.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    seeds = sorted(d["seed"].unique())
    cols = [np.ones(len(d)), d["is_clt"].to_numpy(), d["log_write"].to_numpy()]
    names = ["intercept", "clt", "log_write"]
    for s_ in seeds[1:]:
        cols.append((d["seed"] == s_).astype(float).to_numpy())
        names.append(f"seed_{s_}")
    res = ols(d[metric], np.column_stack(cols), names)
    if res:
        c = res["clt"]
        print(f"   All seeds together (one intercept per seed): "
              f"PLT - CLT = {-c[0]:.4f} (se {c[1]:.4f}), t = {-c[2]:.3f}, p = {c[3]:.4f}")
        print(f"   log(push size) slope = {res['log_write'][0]:.4f} (p = {res['log_write'][3]:.4g})")

    if len(per_seed) >= 2:
        diffs = per_seed["adjusted_plt_minus_clt"]
        t, p = stats.ttest_1samp(diffs, 0.0)
        print(f"   Across seeds: mean adjusted difference = {diffs.mean():.4f}, "
              f"PLT higher in {(diffs > 0).sum()} of {len(diffs)} seeds, t = {t:.3f}, p = {p:.3f}")
        if len(diffs) >= 5:
            w = stats.wilcoxon(diffs)
            print(f"   Wilcoxon signed-rank on adjusted differences: p = {w.pvalue:.3f}")
    return per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["2hop", "3hop"], required=True)
    parser.add_argument("--metric", default="layer1_change_per_unit_write")
    parser.add_argument("--source-layer", type=int, default=0,
                        help="Which batch to combine: 0 (default) or e.g. 1 for 3-hop v features")
    parser.add_argument("--root", default="results/person3/causal_batch")
    parser.add_argument("--output", default=None,
                        help="Default: <root>/<task>_combined_summary.csv")
    args = parser.parse_args()

    suffix = "" if args.source_layer == 0 else f"_L{args.source_layer}"
    folder = re.compile(rf"{args.task}_seed\d+{suffix}")
    files = sorted(f for f in Path(args.root).glob("*/*_batch_summary.csv")
                   if folder.fullmatch(f.parent.name))
    if not files:
        raise SystemExit(f"No batch summaries found under {args.root} for {args.task}")
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data = data.drop_duplicates(subset=["seed", "model_type", "feature_id"])
    if args.metric not in data.columns:
        raise SystemExit(f"Metric {args.metric!r} not in columns: {list(data.columns)}")
    data = data.dropna(subset=[args.metric])

    pd.set_option("display.width", 200)
    print(f"Task: {args.task}   Source layer: {args.source_layer}   Metric: {args.metric}")
    print(f"Files: {len(files)}   Runs: {len(data)}\n")

    rows = []
    for seed, group in data.groupby("seed"):
        plt_v = group.loc[group.model_type == "plt", args.metric].to_numpy()
        clt_v = group.loc[group.model_type == "clt", args.metric].to_numpy()
        t, p = welch(plt_v, clt_v)
        rows.append({
            "seed": seed,
            "n_plt": len(plt_v), "n_clt": len(clt_v),
            "plt_mean": plt_v.mean() if len(plt_v) else np.nan,
            "clt_mean": clt_v.mean() if len(clt_v) else np.nan,
            "plt_std": plt_v.std(ddof=1) if len(plt_v) > 1 else np.nan,
            "clt_std": clt_v.std(ddof=1) if len(clt_v) > 1 else np.nan,
            "difference_plt_minus_clt": (plt_v.mean() - clt_v.mean()) if len(plt_v) and len(clt_v) else np.nan,
            "welch_t": t, "welch_p": p,
        })
    per_seed = pd.DataFrame(rows)
    print("1) Per-seed comparison")
    print(per_seed.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    paired = per_seed.dropna(subset=["plt_mean", "clt_mean"])
    print("\n2) Seed-level paired test (main result; one PLT and one CLT mean per seed)")
    print(f"   Seeds used: {len(paired)}")
    if len(paired) >= 2:
        diff = paired.plt_mean - paired.clt_mean
        t, p = stats.ttest_rel(paired.plt_mean, paired.clt_mean)
        print(f"   Mean difference (PLT - CLT): {diff.mean():.4f}  (sd across seeds {diff.std(ddof=1):.4f})")
        print(f"   PLT higher in {(diff > 0).sum()} of {len(diff)} seeds")
        print(f"   Paired t-test: t = {t:.3f}, p = {p:.3f}")
        if len(paired) >= 5:
            w = stats.wilcoxon(paired.plt_mean, paired.clt_mean)
            print(f"   Wilcoxon signed-rank: W = {w.statistic:.1f}, p = {w.pvalue:.3f}")
        else:
            print("   Wilcoxon skipped (needs at least 5 seeds)")
    else:
        print("   Need at least 2 seeds with both PLT and CLT results.")

    plt_all = data.loc[data.model_type == "plt", args.metric].to_numpy()
    clt_all = data.loc[data.model_type == "clt", args.metric].to_numpy()
    print("\n3) Pooled over all features (ignores seed grouping; for reference only)")
    print(f"   PLT: n={len(plt_all)}, mean={plt_all.mean():.4f}, sd={plt_all.std(ddof=1):.4f}")
    print(f"   CLT: n={len(clt_all)}, mean={clt_all.mean():.4f}, sd={clt_all.std(ddof=1):.4f}")
    t, p = welch(plt_all, clt_all)
    print(f"   Welch t-test: t = {t:.3f}, p = {p:.3f}")
    if len(plt_all) and len(clt_all):
        u = stats.mannwhitneyu(plt_all, clt_all, alternative="two-sided")
        print(f"   Mann-Whitney U: U = {u.statistic:.1f}, p = {u.pvalue:.3f}")

    print("\n4) Prediction check")
    changed = (data["prediction_change_fraction"] > 0).sum()
    print(f"   Runs with any prediction change: {changed} of {len(data)}")
    print(f"   Minimum intervened bit accuracy: {data['intervened_bit_accuracy'].min():.4f}")

    adjusted = adjusted_section(data, args.metric)

    tag = f"{args.task}{suffix}_{args.metric}"
    output = Path(args.output or Path(args.root) / f"{tag}_combined_summary.csv")
    per_seed.to_csv(output, index=False)
    all_path = output.with_name(f"{tag}_all_features.csv")
    data.to_csv(all_path, index=False)
    print(f"\nSaved: {output}")
    print(f"Saved: {all_path}")
    if adjusted is not None and not adjusted.empty:
        adj_path = output.with_name(f"{tag}_push_adjusted.csv")
        adjusted.to_csv(adj_path, index=False)
        print(f"Saved: {adj_path}")


if __name__ == "__main__":
    main()
