
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_transformer import BitVectorTransformer, load_split

SEED = 0
LAYER = 1
DATA = Path("data/boolean_combined.csv")
CHECKPOINT = Path("checkpoints/2hop_L2_seed0.pt")
OUT = Path("results/directional_intervention_seed0_layer1.csv")
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def get_u(X):
    x = X.detach().cpu().numpy()
    return np.logical_xor(x[:, 0, :] > 0.5,
                          x[:, 1, :] > 0.5).astype(int)


def capture(model, X, device, layer):
    saved = {}

    def hook(module, inputs, output):
        saved["activation"] = output.detach().clone()

    handle = model.encoder.layers[layer - 1].register_forward_hook(hook)
    with torch.no_grad():
        logits = model(X.to(device))
    handle.remove()

    return saved["activation"], logits.detach()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = BitVectorTransformer(n_inputs=3, layers=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X_train, _ = load_split(DATA, "2hop", "train")
    X_test, Y_test = load_split(DATA, "2hop", "test")
    u_train, u_test = get_u(X_train), get_u(X_test)

    train_acts, _ = capture(model, X_train, device, LAYER)
    clean_acts, clean_logits = capture(model, X_test, device, LAYER)

    rows = []

    for bit in range(16):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED,
                               class_weight="balanced")
        )
        probe.fit(train_acts[:, bit, :].cpu().numpy(), u_train[:, bit])

        scaler = probe.named_steps["standardscaler"]
        clf = probe.named_steps["logisticregression"]
        direction = clf.coef_[0] / scaler.scale_
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        direction_t = torch.tensor(
            direction, device=device, dtype=clean_acts.dtype
        )

        # Evaluate both true-u classes to avoid an all-zero-only subset.
        rng = np.random.default_rng(SEED + bit)
        selected = []
        for u_value in [0, 1]:
            candidates = np.where(u_test[:, bit] == u_value)[0]
            rng.shuffle(candidates)
            selected.extend(candidates[:50].tolist())

        for idx in selected:
            original = clean_acts[idx:idx + 1].clone()
            true_y = int(Y_test[idx, bit].item())

            # Compare probe direction with a random direction of equal norm.
            random_direction = torch.randn_like(direction_t)
            random_direction = random_direction / (
                torch.linalg.vector_norm(random_direction) + 1e-12
            )

            for control, vector in [
                ("probe_direction", direction_t),
                ("random_direction", random_direction),
            ]:
                for alpha in ALPHAS:
                    def intervene(module, inputs, output,
                                  base=original, vec=vector, pos=bit,
                                  scale=alpha):
                        changed = output.clone()
                        changed[:, pos, :] = (
                            base[:, pos, :] + scale * vec
                        )
                        return changed

                    handle = model.encoder.layers[
                        LAYER - 1
                    ].register_forward_hook(intervene)

                    with torch.no_grad():
                        logits = model(X_test[idx:idx + 1].to(device))

                    handle.remove()

                    logit = float(logits[0, bit].item())
                    pred = int(logit >= 0)

                    rows.append({
                        "seed": SEED,
                        "layer": LAYER,
                        "bit": bit,
                        "example_index": int(idx),
                        "true_u": int(u_test[idx, bit]),
                        "true_y": true_y,
                        "control": control,
                        "alpha": alpha,
                        "clean_logit": float(clean_logits[idx, bit].item()),
                        "intervened_logit": logit,
                        "logit_change": logit - float(
                            clean_logits[idx, bit].item()
                        ),
                        "prediction": pred,
                        "bit_correct": int(pred == true_y),
                    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("Rows:", len(df))
    print("Saved:", OUT)
    print("\nSummary:")
    print(df.groupby(["control", "alpha"]).agg(
        mean_logit=("intervened_logit", "mean"),
        mean_logit_change=("logit_change", "mean"),
        bit_accuracy=("bit_correct", "mean"),
    ).to_string())


if __name__ == "__main__":
    main()
