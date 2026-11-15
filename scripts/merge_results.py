#!/usr/bin/env python
"""Merge the three parts of the run (configs/part_A_D1-3, part_B_D4, part_C_D5) into one results folder.

    python scripts/merge_results.py results/part_A results/part_B results/part_C --out results/merged

Raw per-model CSVs are concatenated (duplicates refused), certificate JSONs copied.
Summary CSVs are NOT merged: regenerate them with make_tables.py on the merged folder.
"""
import argparse
import glob
import os
import shutil

import pandas as pd

RAW = ["ground_truth_certificates.csv", "replacement_fidelity.csv", "graph_metrics_per_example.csv",
       "ground_truth_graph_metrics.csv", "skip_dependence.csv", "ablation_check.csv", "feature_alignment.csv",
       "probes_causal_2x2.csv", "naive_patching.csv", "logit_level_bypass.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cert_out = os.path.join(a.out, "certificates")
    if os.path.isdir(cert_out):            # re-running the merge rebuilds the output from scratch
        shutil.rmtree(cert_out)
    os.makedirs(cert_out, exist_ok=True)
    for name in RAW:
        parts = []
        for d in a.inputs:
            f = os.path.join(d, name)
            if os.path.exists(f) and os.path.getsize(f) > 2:      # skip empty files (e.g. no certified model)
                parts.append(pd.read_csv(f))
        if not parts:
            print(f"missing everywhere: {name}")
            continue
        df = pd.concat(parts, ignore_index=True, sort=False)
        dup = df.duplicated().sum()
        if dup:
            raise SystemExit(f"{name}: {dup} duplicated rows; the split runs overlap. Fix before merging.")
        df.to_csv(os.path.join(a.out, name), index=False)
        print(f"{name}: {len(df)} rows from {len(parts)} runs")
    n = 0
    for d in a.inputs:
        for f in glob.glob(os.path.join(d, "certificates", "*.json")):
            dst = os.path.join(cert_out, os.path.basename(f))
            if os.path.exists(dst):
                raise SystemExit(f"duplicate certificate {f}: two parts contain the same model")
            shutil.copy(f, dst); n += 1
    print(f"certificates: {n}")


if __name__ == "__main__":
    main()
