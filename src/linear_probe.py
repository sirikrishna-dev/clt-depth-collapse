
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_transformer import BitVectorTransformer, load_split

DATA_PATH = Path("data/boolean_combined.csv")
OUT_PATH = Path("results/linear_probe_all_seeds.csv")
SEEDS = [0, 1, 2]
TASK = "2hop"
LAYERS = 2
PROBE_SEED = 42


def collect_activations(model, X, device):
    activations = {}
    handles = []

    for idx, layer in enumerate(model.encoder.layers):
        def capture(module, inputs, output, layer_idx=idx):
            activations[layer_idx] = output.detach().cpu().numpy()
        handles.append(layer.register_forward_hook(capture))

    with torch.no_grad():
        model(X.to(device))

    for handle in handles:
        handle.remove()

    return activations


def get_u(X):
    x = X.detach().cpu().numpy()
    return np.logical_xor(x[:, 0, :] > 0.5,
                          x[:, 1, :] > 0.5).astype(int)


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    X_train, _ = load_split(DATA_PATH, TASK, "train")
    X_test, _ = load_split(DATA_PATH, TASK, "test")
    y_train = get_u(X_train)
    y_test = get_u(X_test)

    all_results = []

    for seed in SEEDS:
        checkpoint_path = Path(f"checkpoints/2hop_L2_seed{seed}.pt")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        model = BitVectorTransformer(n_inputs=3, layers=LAYERS).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        train_acts = collect_activations(model, X_train, device)
        test_acts = collect_activations(model, X_test, device)

        for layer_idx in sorted(train_acts):
            per_bit = []

            for bit in range(16):
                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1000,
                        random_state=PROBE_SEED,
                        class_weight="balanced",
                    ),
                )
                probe.fit(train_acts[layer_idx][:, bit, :], y_train[:, bit])
                pred = probe.predict(test_acts[layer_idx][:, bit, :])

                per_bit.append({
                    "bit": bit,
                    "accuracy": accuracy_score(y_test[:, bit], pred),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_test[:, bit], pred
                    ),
                })

            all_results.append({
                "model_seed": seed,
                "layer": layer_idx + 1,
                "mean_test_accuracy": np.mean(
                    [r["accuracy"] for r in per_bit]
                ),
                "std_test_accuracy_across_bits": np.std(
                    [r["accuracy"] for r in per_bit]
                ),
                "mean_balanced_accuracy": np.mean(
                    [r["balanced_accuracy"] for r in per_bit]
                ),
            })

        print(f"Finished seed {seed}")

    df = pd.DataFrame(all_results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\nResults for each model:")
    print(df.to_string(index=False))

    print("\nMean and standard deviation across model seeds:")
    print(
        df.groupby("layer")[
            ["mean_test_accuracy", "mean_balanced_accuracy"]
        ].agg(["mean", "std"]).to_string()
    )

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
