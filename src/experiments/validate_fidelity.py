
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "run_id",
    "status",
    "model_type",
    "task",
    "base_checkpoint",
    "base_seed",
    "replacement_seed",
    "n_features_per_layer",
    "sparsity_coefficient",
    "write_span",
    "validation_relative_l2",
]


def validate_fidelity():
    """Check fidelity records and matched PLT/CLT configurations."""

    root = Path(__file__).resolve().parents[2]
    input_file = root / "results" / "master_fidelity_table.csv"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        print("ERROR: Missing required columns:", missing)
        return

    if df.empty:
        print("ERROR: Fidelity table contains no records.")
        return

    issues = []

    # Check for missing run IDs.
    run_ids = df["run_id"].astype("string").str.strip()

    if run_ids.isna().any() or run_ids.eq("").any():
        issues.append("Some records have a missing run ID.")

    # Check for repeated run IDs paired with the same original checkpoint.
    # A run ID alone is not unique in the current master table.
    identity_columns = ["run_id", "base_checkpoint"]

    duplicate_records = df.duplicated(
        subset=identity_columns,
        keep=False,
    )

    if duplicate_records.any():
        duplicate_rows = df.loc[
            duplicate_records, identity_columns
        ].drop_duplicates()

        issues.append(
            "Repeated run ID and original-checkpoint combinations found:\n"
            + duplicate_rows.to_string(index=False)
        )

    # Check model types.
    model_types = df["model_type"].astype("string").str.lower().str.strip()

    if model_types.isna().any() or not model_types.isin(
        ["plt", "clt"]
    ).all():
        issues.append("Missing or unknown model types are present.")

    # Check task names.
    tasks = df["task"].astype("string").str.strip()

    if tasks.isna().any() or not tasks.isin(["2hop", "3hop"]).all():
        issues.append("Missing or unknown task names are present.")

    # Check completion status.
    statuses = df["status"].astype("string").str.lower().str.strip()

    if statuses.isna().any() or not statuses.eq("completed").all():
        issues.append("Some runs are missing a completed status.")

    # Check validation errors.
    metric = pd.to_numeric(
        df["validation_relative_l2"], errors="coerce"
    )

    import numpy as np

    invalid_metric = (
        metric.isna()
        | (metric < 0)
        | ~np.isfinite(metric)
    )

    if invalid_metric.any():
        issues.append(
            "Missing, non-numeric, negative, or non-finite "
            "validation errors exist."
        )

    # Check for both PLT and CLT for each task and checkpoint.
    for key, group in df.groupby(
        ["task", "base_checkpoint"], dropna=False
    ):
        group_types = set(
            group["model_type"].astype(str).str.lower().str.strip()
        )

        if not {"plt", "clt"}.issubset(group_types):
            issues.append(
                f"Incomplete PLT/CLT pair for task/checkpoint: {key}"
            )

        # Compare settings expected to match between PLT and CLT.
        for column in [
            "base_seed",
            "replacement_seed",
            "n_features_per_layer",
            "sparsity_coefficient",
        ]:
            plt_values = group.loc[
                group["model_type"].astype(str).str.lower().str.strip()
                == "plt",
                column,
            ].dropna().unique()

            clt_values = group.loc[
                group["model_type"].astype(str).str.lower().str.strip()
                == "clt",
                column,
            ].dropna().unique()

            if len(plt_values) and len(clt_values):
                if set(plt_values) != set(clt_values):
                    issues.append(
                        f"Configuration mismatch in {column} "
                        f"for task/checkpoint {key}."
                    )

    print(f"Fidelity records checked: {len(df)}")

    if issues:
        print(f"\nFound {len(issues)} issue(s):")
        for issue in issues:
            print("-", issue)
    else:
        print("No issues found by these structural checks.")

    print(
        "\nThese checks do not establish that the models "
        "or attribution graphs are scientifically valid."
    )


if __name__ == "__main__":
    validate_fidelity()
