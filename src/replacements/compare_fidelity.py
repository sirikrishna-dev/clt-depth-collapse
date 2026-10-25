"""Compare PLT/CLT fidelity records using validation relative-L2 metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="results/person2/fidelity_table.csv")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Provisional maximum absolute difference in validation relative-L2; team must approve this threshold")
    args = parser.parse_args()
    path = Path(args.table)
    if not path.exists():
        raise FileNotFoundError(f"No fidelity table at {path}; train at least one PLT and one CLT first.")
    df = pd.read_csv(path)
    needed = {"model_type", "task", "base_checkpoint", "validation_relative_l2", "run_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Fidelity table missing columns: {sorted(missing)}")
    rows = []
    groups = df.groupby(["task", "base_checkpoint"], dropna=False)
    for (task, base), group in groups:
        plts = group[group.model_type == "plt"]
        clts = group[group.model_type == "clt"]
        if plts.empty or clts.empty:
            rows.append({"task": task, "base_checkpoint": base, "status": "INCOMPLETE", "plt_validation_relative_l2": None, "clt_validation_relative_l2": None, "absolute_gap": None, "tolerance": args.tolerance})
            continue
        # If multiple settings exist, compare every PLT/CLT pairing in this base-model group.
        for _, p in plts.iterrows():
            for _, c in clts.iterrows():
                gap = abs(float(p.validation_relative_l2) - float(c.validation_relative_l2))
                rows.append({
                    "task": task, "base_checkpoint": base,
                    "plt_run_id": p.run_id, "clt_run_id": c.run_id,
                    "plt_validation_relative_l2": float(p.validation_relative_l2),
                    "clt_validation_relative_l2": float(c.validation_relative_l2),
                    "absolute_gap": gap, "tolerance": args.tolerance,
                    "status": "PROVISIONAL_MATCH" if gap <= args.tolerance else "NOT_MATCHED",
                })
    out = path.with_name("fidelity_matching_report.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {out}. Threshold is provisional and requires team agreement.")

if __name__ == "__main__":
    main()
