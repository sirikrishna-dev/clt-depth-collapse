"""Generate Person 1's paper tables (Markdown) from results/v2_matched/person1/*.csv.

    python scripts/make_person1_tables.py  >  results/v2_matched/person1/TABLES.md

No number in Section 4 of the paper is typed by hand; it comes from this output.
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "v2_matched" / "person1"


def group(mid):
    if "siit" in mid:
        return ("2-hop SIIT" if mid.startswith("2hop") else "3-hop SIIT")
    return "2-hop baseline" if mid.startswith("2hop") else "3-hop baseline"


def fmt_ci(r):
    return f"{r.score:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] (n={r.n})"


def main():
    san = pd.read_csv(R / "gt_sanity.csv")
    abl = pd.read_csv(R / "gt_ablation.csv")
    dec = pd.read_csv(R / "gt_decodability.csv")
    sii = pd.read_csv(R / "gt_siit_interchange_chain_bypass.csv")
    sho = pd.read_csv(R / "gt_y_shortcut.csv")
    dt = pd.read_csv(R / "gt_dtrue.csv")
    for df in (san, abl, dec, sii, sho, dt):
        df["group"] = df.model_id.map(group)

    print("## Table 4.1 Sanity and class balance (test split)\n")
    t = san.groupby("group").agg(models=("model_id", "count"), bit_acc_min=("bit_acc", "min"),
                                 balanced_acc_min=("balanced_acc", "min"),
                                 frac_y1=("frac_y_is_1", "mean"),
                                 majority_baseline=("majority_baseline_acc", "mean"))
    print(t.round(3).to_markdown(), "\n")

    print("## Table 4.2 MLP resample ablation (mean over 3 seeds, then range across models)\n")
    m = abl[(abl.site == "mlp") & (abl.method == "resample")].groupby(["group", "model_id", "layer"])[
        ["flip_rate", "ba_drop"]].mean().reset_index()
    t = m.groupby(["group", "layer"]).agg(flip_min=("flip_rate", "min"), flip_max=("flip_rate", "max"),
                                          ba_drop_min=("ba_drop", "min"), ba_drop_max=("ba_drop", "max"))
    print(t.round(3).to_markdown(), "\n")

    print("## Table 4.3 Decodable vs used: y after block 0\n")
    d0 = dec[(dec["var"] == "y") & (dec.layer == 0)].groupby("group").balanced_acc.agg(["min", "max"])
    s0 = sho[sho.protocol == "after_block0_probe_direction"].groupby("group").score.agg(["min", "max"])
    c0 = sho[sho.protocol == "after_block0_random_direction_control"].groupby("group").score.max()
    t = pd.DataFrame({"y probe balanced acc (min-max)": d0["min"].round(3).astype(str) + "-" + d0["max"].round(3).astype(str),
                      "output follows donor y (min-max)": s0["min"].round(3).astype(str) + "-" + s0["max"].round(3).astype(str),
                      "random-direction control (max)": c0.round(3)})
    print(t.to_markdown(), "\n")

    print("## Table 4.4 SIIT designated-site tests (range across seeds; score [95% Wilson CI] (n))\n")
    rows = []
    for (g, test, prot), x in sii.groupby(["group", "test", "protocol"]):
        lo, hi = x.loc[x.score.idxmin()], x.loc[x.score.idxmax()]
        rows.append(dict(group=g, test=test, protocol=prot, worst_seed=fmt_ci(lo),
                         best_seed=fmt_ci(hi)))
    print(pd.DataFrame(rows).to_markdown(index=False), "\n")

    print("## Table 4.5 Ground-truth depth per model\n")
    print(dt[["group", "model_id", "necessary_mlp_layers", "y_shortcut_scores", "D_true", "evidence",
              "exclusive_sites"]].to_markdown(index=False), "\n")


if __name__ == "__main__":
    main()
