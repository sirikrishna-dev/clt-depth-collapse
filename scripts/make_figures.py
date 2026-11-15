#!/usr/bin/env python
"""Paper figures from the summary CSVs written by make_tables.py.

    python scripts/make_figures.py --results results/merged --out paper/figures

Palette: validated categorical slots (PLT blue, CLT orange, param-matched PLT aqua);
every series also has its own marker so identity never relies on colour alone.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COL = {"plt": "#2a78d6", "clt": "#eb6834", "plt_parammatched": "#1baf7a"}
MARK = {"plt": "o", "clt": "s", "plt_parammatched": "^"}
LAB = {"plt": "PLT", "clt": "CLT", "plt_parammatched": "PLT (param-matched)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_depth_vs_D(R, out, metric):
    ds = pd.read_csv(os.path.join(R, "summary_depth.csv"))
    per = pd.read_csv(os.path.join(R, "summary_depth_per_model.csv"))
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    Ds = sorted(ds.D.unique())
    ax.plot(Ds, Ds, color=INK2, lw=1, ls="--", label="ground truth ($D_\\mathrm{feat}=D$)")
    for i, kind in enumerate([k for k in ("plt", "clt", "plt_parammatched") if k in set(ds.kind)]):
        d = ds[ds.kind == kind].sort_values("D")
        off = (i - 0.5) * 0.08
        ax.errorbar(d.D + off, d[metric], yerr=[d[metric] - d[metric + "_lo"], d[metric + "_hi"] - d[metric]],
                    color=COL[kind], marker=MARK[kind], ms=6, capsize=2, label=LAB[kind])
        p = per[(per.kind == kind) & (per.n_layers == per.D)]
        ax.scatter(p.D + off, p[metric], color=COL[kind], s=8, alpha=0.35, lw=0)
    ax.set_xlabel("true depth $D$ (Boolean hops)")
    ax.set_ylabel("effective graph depth $D_\\mathrm{feat}$")
    ax.set_xticks(Ds)
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(os.path.join(out, "fig1_depth_vs_D.pdf")); fig.savefig(os.path.join(out, "fig1_depth_vs_D.png"))


def fig_hop_profile(R, out):
    per = pd.read_csv(os.path.join(R, "summary_depth_per_model.csv"))
    kcols = sorted([c for c in per.columns if c.startswith("root_signed_I") and c[13:].isdigit()], key=lambda c: int(c[13:]))
    Ds = sorted(per.D.unique())
    fig, axes = plt.subplots(1, len(Ds), figsize=(1.6 * len(Ds) + 0.6, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    blues = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    for ax, D in zip(axes, Ds):
        bottom = np.zeros(2)
        kinds = [k for k in ("plt", "clt") if k in set(per.kind)]
        for j, c in enumerate(kcols[:D + 2]):
            vals = np.array([per[(per.D == D) & (per.kind == k) & (per.n_layers == per.D)][c].mean() for k in kinds])
            vals = np.nan_to_num(vals)
            ax.bar(range(len(kinds)), vals, bottom=bottom[:len(kinds)], color=blues[min(j, 6)],
                   edgecolor="#fcfcfb", linewidth=1, label=f"{j} hops" if D == Ds[-1] else None)
            bottom[:len(kinds)] += vals
        ax.set_xticks(range(len(kinds))); ax.set_xticklabels([LAB[k] for k in kinds])
        ax.set_title(f"D = {D}", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of root-input contribution")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.savefig(os.path.join(out, "fig2_hop_profile.pdf")); fig.savefig(os.path.join(out, "fig2_hop_profile.png"))


def fig_sparsity(R, out, metric):
    per = pd.read_csv(os.path.join(R, "summary_depth_per_model.csv"))
    per = per[per.n_layers == per.D]
    if per.k.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for kind in [k for k in ("plt", "clt") if k in set(per.kind)]:
        d = per[per.kind == kind].groupby("k").DCR.agg(["mean", "sem"]).reset_index()
        ax.errorbar(d.k, d["mean"], yerr=d["sem"], color=COL[kind], marker=MARK[kind], ms=6, capsize=2, label=LAB[kind])
    ax.set_xscale("log", base=2); ax.set_xlabel("active features per token per layer ($k$)")
    ax.set_ylabel("depth-collapse ratio $D_\\mathrm{feat}/D$ (mean over $D$)")
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, "fig3_sparsity.pdf")); fig.savefig(os.path.join(out, "fig3_sparsity.png"))


def fig_skipdep(R, out, main_k):
    p = os.path.join(R, "summary_skip_dependence.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    s = s[s.k == main_k]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for kind in [k for k in ("plt", "clt") if k in set(s.kind)]:
        d = s[s.kind == kind].sort_values("D")
        ax.plot(d.D, d.skip_only_acc, color=COL[kind], marker=MARK[kind], ms=6, label=f"{LAB[kind]}: skip route only")
        ax.plot(d.D, d.stepwise_only_acc, color=COL[kind], marker=MARK[kind], ms=6, ls=":", label=f"{LAB[kind]}: stepwise route only")
    ax.set_xlabel("true depth $D$"); ax.set_ylabel("counterfactual accuracy (informative bits)")
    ax.set_ylim(0, 1.05); ax.set_xticks(sorted(s.D.unique()))
    ax.legend(fontsize=6.5)
    fig.savefig(os.path.join(out, "fig4_skip_dependence.pdf")); fig.savefig(os.path.join(out, "fig4_skip_dependence.png"))


def fig_ablation(R, out):
    p = os.path.join(R, "ablation_check.csv")
    if not os.path.exists(p):
        return
    ab = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for kind in [k for k in ("plt", "clt") if k in set(ab.kind)]:
        d = ab[ab.kind == kind]
        ax.scatter(d.pred, d.actual, s=10, color=COL[kind], marker=MARK[kind], alpha=0.6, lw=0, label=LAB[kind])
    lim = np.nanmax(np.abs(np.r_[ab.pred, ab.actual])) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--")
    ax.set_xlabel("graph-predicted $\\Delta$ logit"); ax.set_ylabel("actual $\\Delta$ logit (ablation)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, "fig5_ablation_check.pdf")); fig.savefig(os.path.join(out, "fig5_ablation_check.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/merged")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--metric", default="root_signed_D_feat")
    ap.add_argument("--main-k", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_depth_vs_D(a.results, a.out, a.metric)
    fig_hop_profile(a.results, a.out)
    fig_sparsity(a.results, a.out, a.metric)
    fig_skipdep(a.results, a.out, a.main_k)
    fig_ablation(a.results, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
