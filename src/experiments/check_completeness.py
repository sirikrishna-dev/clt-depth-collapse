
from pathlib import Path

import pandas as pd


EXPECTED_TASKS = ["2hop", "3hop"]
EXPECTED_BASE_SEEDS = [0, 1, 2]
EXPECTED_MODELS = ["plt", "clt"]


def check_completeness():
    """Check expected PLT/CLT records against the fidelity table."""

    project_root = Path(__file__).resolve().parents[2]
    input_file = project_root / "results" / "master_fidelity_table.csv"
    output_file = project_root / "results" / "completeness_report.csv"

    if not input_file.exists():
        print(f"ERROR: Fidelity table not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    required_columns = [
        "task",
        "base_seed",
        "model_type",
        "base_checkpoint",
        "status",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("ERROR: Missing columns:", missing)
        return

    df["task"] = df["task"].astype(str).str.strip()
    df["model_type"] = (
        df["model_type"].astype(str).str.lower().str.strip()
    )
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df["base_seed"] = pd.to_numeric(df["base_seed"], errors="coerce")

    expected_records = []

    for task in EXPECTED_TASKS:
        for seed in EXPECTED_BASE_SEEDS:
            for model in EXPECTED_MODELS:
                matching = df[
                    (df["task"] == task)
                    & (df["base_seed"] == seed)
                    & (df["model_type"] == model)
                ]

                if matching.empty:
                    status = "MISSING"
                    checkpoint = ""
                elif len(matching) > 1:
                    status = "MULTIPLE_RECORDS_CHECK"
                    checkpoint = "; ".join(
                        matching["base_checkpoint"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                elif matching.iloc[0]["status"] != "completed":
                    status = "NOT_COMPLETED"
                    checkpoint = str(matching.iloc[0]["base_checkpoint"])
                else:
                    status = "PRESENT"
                    checkpoint = str(matching.iloc[0]["base_checkpoint"])

                expected_records.append(
                    {
                        "task": task,
                        "base_seed": seed,
                        "model_type": model,
                        "base_checkpoint": checkpoint,
                        "completeness_status": status,
                    }
                )

    report = pd.DataFrame(expected_records)
    report.to_csv(output_file, index=False)

    print("Expected experiment records:", len(report))
    print("\nCompleteness status:")
    print(report.to_string(index=False))

    print("\nSummary:")
    print(report["completeness_status"].value_counts().to_string())

    print(f"\nReport saved to: {output_file}")
    print(
        "Note: This checks the currently configured experiment scope only. "
        "It does not prove scientific validity."
    )


if __name__ == "__main__":
    check_completeness()
