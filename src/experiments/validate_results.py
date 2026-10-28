
import csv
import math
from pathlib import Path


def validate_results():
    """Check experiment results for missing or invalid values."""

    project_root = Path(__file__).resolve().parents[2]
    results_folder = project_root / "results"
    results_file = results_folder / "experiment_results.csv"

    if not results_file.exists():
        print("No experiment results file found yet.")
        print(f"Expected location: {results_file}")
        print("This is normal if experiments have not been run.")
        return

    with results_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "experiment_id",
            "status",
            "validation_relative_l2",
        }

        if reader.fieldnames is None:
            print("ERROR: Results file is empty.")
            return

        missing = required_columns - set(reader.fieldnames)

        if missing:
            print("ERROR: Missing columns:", sorted(missing))
            return

        rows = list(reader)

    if not rows:
        print("Results file contains no experiment records.")
        return

    issues = []

    for line_number, row in enumerate(rows, start=2):
        experiment_id = row["experiment_id"].strip()

        if not experiment_id:
            issues.append(f"Line {line_number}: Missing experiment ID.")

        status = row["status"].strip().lower()

        if status not in {"completed", "failed", "incomplete"}:
            issues.append(
                f"Line {line_number}: Unknown result status '{status}'."
            )

        metric = row["validation_relative_l2"].strip()

        if status == "completed":
            try:
                value = float(metric)

                if not math.isfinite(value) or value < 0:
                    issues.append(
                        f"Line {line_number}: Invalid relative-L2 value."
                    )

            except ValueError:
                issues.append(
                    f"Line {line_number}: Missing or invalid relative-L2 value."
                )

    print(f"Result records checked: {len(rows)}")

    if issues:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues:
            print("-", issue)
    else:
        print("No result-record validation errors found.")

    print("Note: Passing these checks does not prove scientific validity.")


if __name__ == "__main__":
    validate_results()
