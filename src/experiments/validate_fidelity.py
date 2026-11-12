
from pathlib import Path

import numpy as np
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


PAIRING_COLUMNS = [
    "task",
    "base_checkpoint",
    "base_seed",
    "replacement_seed",
    "n_features_per_layer",
    "sparsity_coefficient",
]


def validate_fidelity():
    """Validate fidelity records and one-to-one PLT/CLT pairs."""

    root = Path(__file__).resolve().parents[2]
    input_file = root / "results" / "master_fidelity_table.csv"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return False

    df = pd.read_csv(input_file)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns:", missing)
        return False

    if df.empty:
        print("ERROR: Fidelity table contains no records.")
        return False

    issues = []

    # Normalize text fields.
    for column in ["run_id", "status", "model_type", "task"]:
        df[column] = df[column].astype("string").str.strip()

    df["status"] = df["status"].str.lower()
    df["model_type"] = df["model_type"].str.lower()

    # Check required identifiers.
    for column in ["run_id", "base_checkpoint"]:
        values = df[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            issues.append(f"Missing values in {column}.")

    if df["run_id"].duplicated().any():
        issues.append("Duplicate run IDs found.")

    if not df["model_type"].isin(["plt", "clt"]).all():
        issues.append("Missing or unknown model types.")

    if not df["task"].isin(["2hop", "3hop"]).all():
        issues.append("Missing or unknown task names.")

    if not df["status"].eq("completed").all():
        issues.append("Some runs are not marked completed.")

    # Validate numeric configuration and metric fields.
    numeric_columns = [
        "base_seed",
        "replacement_seed",
        "n_features_per_layer",
        "sparsity_coefficient",
        "write_span",
        "validation_relative_l2",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        values = df[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            issues.append(f"Missing or non-finite values in {column}.")

    for column in [
        "base_seed",
        "replacement_seed",
        "n_features_per_layer",
        "write_span",
        "validation_relative_l2",
    ]:
        if (df[column] < 0).any():
            issues.append(f"Negative values found in {column}.")

    if (df["n_features_per_layer"] <= 0).any():
        issues.append("Feature count must be positive.")

    if (df["write_span"] <= 0).any():
        issues.append("Write span must be positive.")

    if (df["sparsity_coefficient"] < 0).any():
        issues.append("Sparsity coefficient cannot be negative.")

    # Stop before grouping if configuration values are invalid.
    if issues:
        print(f"Fidelity records checked: {len(df)}")
        print(f"\nFound {len(issues)} issue(s):")
        for issue in issues:
            print("-", issue)
        return False

    # Every shared configuration must contain exactly one PLT and one CLT.
    grouped = df.groupby(PAIRING_COLUMNS, dropna=False)

    for key, group in grouped:
        counts = group["model_type"].value_counts()

        if counts.get("plt", 0) != 1 or counts.get("clt", 0) != 1:
            issues.append(
                f"Expected exactly one PLT and one CLT for {key}; "
                f"found PLT={counts.get('plt', 0)}, "
                f"CLT={counts.get('clt', 0)}."
            )

    # Check that every record belongs to a complete pair.
    expected_pairs = (
        df.groupby(["task", "base_seed"])["model_type"]
        .agg(lambda x: set(x))
    )

    for key, model_types in expected_pairs.items():
        if model_types != {"plt", "clt"}:
            issues.append(f"Incomplete PLT/CLT pair for {key}.")

    print(f"Fidelity records checked: {len(df)}")
    print(f"Configuration groups checked: {len(grouped)}")

    if issues:
        print(f"\nFound {len(issues)} issue(s):")
        for issue in issues:
            print("-", issue)
        return False

    print("No structural or pairing issues found.")
    print(
        "Note: Passing these checks does not establish scientific "
        "validity or fidelity equivalence."
    )
    return True


if __name__ == "__main__":
    validate_fidelity()
