import json
from pathlib import Path

import torch

from train_transformer import BitVectorTransformer, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "boolean_combined.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUT = ROOT / "results" / "independent_model_evaluation.csv"


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    rows = []

    for task in ("2hop", "3hop"):
        X, Y = load_split(DATA, task, "test")

        for seed in (0, 1, 2):
            model_id = f"{task}_L2_seed{seed}"
            checkpoint_path = CHECKPOINT_DIR / f"{model_id}.pt"

            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)

            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            model = BitVectorTransformer(
                n_inputs=X.shape[1], layers=checkpoint["layers"]
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            with torch.no_grad():
                logits = model(X.to(device)).cpu()
                predictions = (logits >= 0).float()
                bit_accuracy = (predictions == Y).float().mean().item()
                exact_accuracy = (
                    (predictions == Y).all(dim=1).float().mean().item()
                )

            row = {
                "model_id": model_id,
                "task": task,
                "seed": seed,
                "test_examples": len(Y),
                "test_bit_accuracy": bit_accuracy,
                "test_exact_vector_accuracy": exact_accuracy,
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            }
            rows.append(row)
            print(json.dumps(row))

    import pandas as pd

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\nSaved:", OUT)
    print("Evaluated models:", len(rows))


if __name__ == "__main__":
    main()
