import csv
from pathlib import Path


def test_independent_model_evaluation_has_all_six_models():
    root = Path(__file__).resolve().parents[1]
    results_file = root / "results" / "independent_model_evaluation.csv"

    assert results_file.exists(), "Run src/evaluate_model_zoo.py first"

    with results_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6

    expected = {
        f"{task}_L2_seed{seed}"
        for task in ("2hop", "3hop")
        for seed in (0, 1, 2)
    }
    assert {row["model_id"] for row in rows} == expected

    for row in rows:
        assert int(row["test_examples"]) == 1000
        assert float(row["test_bit_accuracy"]) == 1.0
        assert float(row["test_exact_vector_accuracy"]) == 1.0
        assert Path(root / row["checkpoint"]).exists()
