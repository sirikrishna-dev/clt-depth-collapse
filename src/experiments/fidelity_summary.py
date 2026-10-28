
import math
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_file = project_root / "results" / "master_fidelity_table.csv"
    output_folder = project_root / "results"

    if not input_file.exists():
        print(f"ERROR: Fidelity table not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    required_columns = [
        "status",
        "model_type",
        "task",
        "base_seed",
        "validation_relative_l2",
        "test_relative_l2",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        print("ERROR: Missing required columns:", missing)
        return

    if df.empty:
        print("ERROR: The fidelity table contains no rows.")
        return

    df["model_type"] = df["model_type"].astype(str).str.lower().str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.lower().str.strip()

    for column in ["validation_relative_l2", "test_relative_l2"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    issues = []

    if not df["model_type"].isin(["plt", "clt"]).all():
        issues.append("Some rows have an unknown model type.")

    if not df["task"].isin(["2hop", "3hop"]).all():
        issues.append("Some rows have an unknown task.")

    if not df["status"].eq("completed").all():
        issues.append("Some runs are not marked completed.")

    for column in ["validation_relative_l2", "test_relative_l2"]:
        valid = df[column].map(
            lambda value: pd.notna(value)
            and math.isfinite(value)
            and value >= 0
        )
        if not valid.all():
            issues.append(f"Some values in {column} are missing or invalid.")

    if issues:
        print("ERROR: The table needs attention:")
        for issue in issues:
            print("-", issue)
        print("No summary files were generated.")
        return

    # Summarize each model type across original checkpoint seeds.
    summary = (
        df.groupby(["task", "model_type"])
        .agg(
            run_count=("base_seed", "size"),
            validation_mean=("validation_relative_l2", "mean"),
            validation_std=("validation_relative_l2", "std"),
            test_mean=("test_relative_l2", "mean"),
            test_std=("test_relative_l2", "std"),
        )
        .reset_index()
    )

    summary.to_csv(output_folder / "fidelity_summary.csv", index=False)

    # Compare PLT and CLT for each task and original checkpoint seed.
    paired = df.pivot(
        index=["task", "base_seed"],
        columns="model_type",
        values="validation_relative_l2",
    ).reset_index()

    if not {"plt", "clt"}.issubset(paired.columns):
        print("ERROR: Could not pair PLT and CLT results.")
        return

    paired["absolute_validation_gap"] = (
        paired["plt"] - paired["clt"]
    ).abs()

    paired["lower_error_model"] = paired.apply(
        lambda row: (
            "PLT" if row["plt"] < row["clt"]
            else "CLT" if row["clt"] < row["plt"]
            else "Equal"
        ),
        axis=1,
    )

    paired.to_csv(
        output_folder / "fidelity_paired_comparison.csv",
        index=False,
    )

    print("Fidelity summary across original checkpoint seeds:")
    print(summary.to_string(index=False))

    print("\nPLT–CLT comparison for each original checkpoint seed:")
    print(paired.to_string(index=False))

    print("\nCreated:")
    print(output_folder / "fidelity_summary.csv")
    print(output_folder / "fidelity_paired_comparison.csv")

    print(
        "\nNote: These are reconstruction comparisons, "
        "not proof of causal validity."
    )


if __name__ == "__main__":
    main()
