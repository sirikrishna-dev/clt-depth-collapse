
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
RESULTS = ROOT / "results" / "3hop_causal_chain_validation.csv"

with RESULTS.open(newline="", encoding="utf-8") as f:
    validation = {int(r["seed"]): r for r in csv.DictReader(f)}

assert set(validation) == set(range(5)), "Missing validation seeds"

records = []

for path in sorted(META.glob("*.json")):
    if path.name == "model_zoo_schema.json":
        continue

    data = json.loads(path.read_text(encoding="utf-8"))

    if "model_id" not in data:
        continue

    model_id = data["model_id"]

    # Correct misleading source labels for normally trained models.
    if model_id.startswith(("2hop_L2_seed", "3hop_L2_seed")):
        data["source"] = "standard_trained_transformer"
        data["validated"] = False
        data["validation_status"] = "standard_training_baseline"

    # Record the expanded 3-hop causal-chain validation.
    if model_id.startswith("3hop_L3_siit_k1_seed"):
        seed = data["seed"]
        result = validation[seed]

        assert int(result["changed_bits"]) > 0

        metrics = {
            key: float(result[key])
            for key in ("v_propagation", "y_counterfactual", "v_mediation")
        }

        passed = all(value >= 0.95 for value in metrics.values())

        data["causal_chain_validation"] = {
            "status": "passed" if passed else "failed",
            "results_file": "results/3hop_causal_chain_validation.csv",
            "changed_bits": int(result["changed_bits"]),
            "pairing_seed": int(result["pairing_seed"]),
            "metrics": metrics,
            "interpretation": (
                "Intervention-based evidence that v mediates "
                "the tested causal effect of u on the final output."
            ),
        }

        data["validated"] = passed
        data["validation_status"] = (
            "passed_causal_chain_protocol"
            if passed else "failed_causal_chain_protocol"
        )
        data["exclusive_depth_verified"] = False

        # Retain original training_interchange_check unchanged.
        # Retain original validation_todo as historical information.

    path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    records.append(data)

zoo = META / "model_zoo.jsonl"

with zoo.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

print("Metadata records:", len(records))
print("Unique model IDs:", len({r["model_id"] for r in records}))
print("Validated models:", sum(r.get("validated", False) for r in records))
print("Model zoo rebuilt:", zoo)
