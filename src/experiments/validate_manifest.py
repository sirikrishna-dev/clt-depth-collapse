
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "experiment_id",
    "task",
    "model_type",
    "seed",
    "sparsity",
    "status",
    "notes",
]


def validate_manifest():
    """Check the experiment manifest for missing or invalid information."""

    project_root = Path(__file__).resolve().parents[2]
    manifest_file = project_root / "results" / "experiment_manifest.csv"

    if not manifest_file.exists():
        print("ERROR: Experiment manifest was not found.")
        return

    with manifest_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            print("ERROR: Manifest is empty.")
            return

        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing_columns:
            print("ERROR: Missing columns:", missing_columns)
            return

        rows = list(reader)

    if not rows:
        print("ERROR: Manifest has no experiment entries.")
        return

    errors = []
    seen_ids = set()

    for line_number, row in enumerate(rows, start=2):
        experiment_id = row["experiment_id"].strip()

        if not experiment_id:
            errors.append(f"Line {line_number}: Missing experiment ID.")
        elif experiment_id in seen_ids:
            errors.append(
                f"Line {line_number}: Duplicate ID '{experiment_id}'."
            )
        else:
            seen_ids.add(experiment_id)

        if row["task"].strip() not in {"2hop", "3hop"}:
            errors.append(
                f"Line {line_number}: Invalid task '{row['task']}'."
            )

        if row["model_type"].strip().upper() not in {"PLT", "CLT"}:
            errors.append(
                f"Line {line_number}: Invalid model type "
                f"'{row['model_type']}'."
            )

        try:
            seed = int(row["seed"])
            if seed < 0:
                errors.append(
                    f"Line {line_number}: Seed cannot be negative."
                )
        except (ValueError, TypeError):
            errors.append(
                f"Line {line_number}: Seed must be a whole number."
            )

        if not row["status"].strip():
            errors.append(f"Line {line_number}: Missing status.")

    print(f"Manifest entries checked: {len(rows)}")

    if errors:
        print(f"Validation found {len(errors)} issue(s):")
        for error in errors:
            print("-", error)
    else:
        print("No structural validation errors found.")

    print(
        "Note: Structural validity does not mean an experiment "
        "has been scientifically validated."
    )


if __name__ == "__main__":
    validate_manifest()
