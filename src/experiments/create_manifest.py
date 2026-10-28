
import csv
from pathlib import Path


def create_manifest():
    """Create a table of planned experiments."""

    project_root = Path(__file__).resolve().parents[2]
    results_folder = project_root / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    manifest_file = results_folder / "experiment_manifest.csv"

    tasks = ["2hop", "3hop"]
    model_types = ["PLT", "CLT"]
    seeds = [0, 1, 2]

    columns = [
        "experiment_id",
        "task",
        "model_type",
        "seed",
        "sparsity",
        "status",
        "notes",
    ]

    experiment_number = 1

    with manifest_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for task in tasks:
            for model_type in model_types:
                for seed in seeds:
                    writer.writerow(
                        {
                            "experiment_id": f"EXP-{experiment_number:03d}",
                            "task": task,
                            "model_type": model_type,
                            "seed": seed,
                            "sparsity": "",
                            "status": "planned_not_validated",
                            "notes": (
                                "Confirm model availability, sparsity settings, "
                                "and validation criteria before running."
                            ),
                        }
                    )
                    experiment_number += 1

    print(f"Manifest created: {manifest_file}")
    print(f"Number of planned experiment entries: {experiment_number - 1}")


if __name__ == "__main__":
    create_manifest()
