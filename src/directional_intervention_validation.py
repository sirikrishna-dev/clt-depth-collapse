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
OUT = Path("results/directional_intervention_validation_seed0_layer1.csv")
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SAMPLES_PER_CLASS = 50


def get_u(X):
    x = X.detach().cpu().numpy()
    return np.logical_xor(
        x[:, 0, :] > 0.5, x[:, 1, :] > 0.5
    ).astype(int)


def capture(model, X, device, layer):
    saved = {}

    def hook(module, inputs, output):
        saved["activation"] = output.detach().clone()

    handle = model.encoder.layers[layer - 1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(X.to(device))
    finally:
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

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    model = BitVectorTransformer(n_inputs=3, layers=2).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_train, _ = load_split(DATA, "2hop", "train")
    X_test, Y_test = load_split(DATA, "2hop", "test")
    u_train, u_test = get_u(X_train), get_u(X_test)

    train_acts, _ = capture(model, X_train, device, LAYER)
    clean_acts, clean_logits = capture(model, X_test, device, LAYER)

    probes = []
    directions = []

    for bit in range(16):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=SEED,
                class_weight="balanced",
            ),
        )
        probe.fit(
            train_acts[:, bit, :].cpu().numpy(),
            u_train[:, bit],
        )

        scaler = probe.named_steps["standardscaler"]
        classifier = probe.named_steps["logisticregression"]
        direction = classifier.coef_[0] / scaler.scale_
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        probes.append(probe)
        directions.append(
            torch.tensor(direction, device=device, dtype=clean_acts.dtype)
        )

    rows = []

    for bit in range(16):
        probe = probes[bit]
        probe_direction = directions[bit]

        rng = np.random.default_rng(SEED + bit)
        selected = []

        for u_value in (0, 1):
            candidates = np.where(u_test[:, bit] == u_value)[0].copy()
            rng.shuffle(candidates)
            selected.extend(candidates[:SAMPLES_PER_CLASS].tolist())

        for idx in selected:
            original = clean_acts[idx:idx + 1].clone()
            original_vector = original[0, bit, :].cpu().numpy()
            score_before = float(
                probe.decision_function(original_vector.reshape(1, -1))[0]
            )
            clean_logit = float(clean_logits[idx, bit].item())
            true_u = int(u_test[idx, bit])
            true_y = int(Y_test[idx, bit].item())

            random_direction = torch.randn(
                probe_direction.shape,
                generator=torch.Generator(device="cpu").manual_seed(
                    SEED * 100000 + bit * 1000 + int(idx)
                ),
                dtype=torch.float32,
            ).to(device=device, dtype=clean_acts.dtype)
            random_direction = random_direction / (
                torch.linalg.vector_norm(random_direction) + 1e-12
            )

            for control, vector in (
                ("probe_direction", probe_direction),
                ("random_direction", random_direction),
            ):
                for alpha in ALPHAS:
                    captured = {}

                    def intervene(module, inputs, output,
                                  base=original, vec=vector,
                                  pos=bit, scale=alpha):
                        changed = output.clone()
                        changed[:, pos, :] = (
                            base[:, pos, :] + scale * vec
                        )
                        captured["activation"] = changed[:, pos, :].detach()
                        return changed

                    handle = model.encoder.layers[
                        LAYER - 1
                    ].register_forward_hook(intervene)

                    try:
                        with torch.no_grad():
                            logits = model(X_test[idx:idx + 1].to(device))
                    finally:
                        handle.remove()

                    changed_vector = (
                        captured["activation"][0].cpu().numpy()
                    )
                    score_after = float(
                        probe.decision_function(
                            changed_vector.reshape(1, -1)
                        )[0]
                    )
                    logit_after = float(logits[0, bit].item())
                    prediction = int(logit_after >= 0)

                    rows.append({
                        "seed": SEED,
                        "layer": LAYER,
                        "bit": bit,
                        "example_index": int(idx),
                        "true_u": true_u,
                        "true_y": true_y,
                        "control": control,
                        "alpha": alpha,
                        "probe_score_before": score_before,
                        "probe_score_after": score_after,
                        "probe_score_change": score_after - score_before,
                        "abs_probe_score_change": abs(
                            score_after - score_before
                        ),
                        "clean_logit": clean_logit,
                        "intervened_logit": logit_after,
                        "logit_change": logit_after - clean_logit,
                        "abs_logit_change": abs(logit_after - clean_logit),
                        "prediction": prediction,
                        "prediction_flip": int(
                            prediction != int(clean_logit >= 0)
                        ),
                        "bit_correct": int(prediction == true_y),
                    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    summary = df.groupby(["control", "alpha"]).agg(
        mean_abs_probe_score_change=("abs_probe_score_change", "mean"),
        mean_abs_logit_change=("abs_logit_change", "mean"),
        prediction_flip_rate=("prediction_flip", "mean"),
        bit_accuracy=("bit_correct", "mean"),
        examples=("example_index", "count"),
    )

    print("Rows:", len(df))
    print("Saved:", OUT)
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
