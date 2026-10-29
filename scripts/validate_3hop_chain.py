
import sys
import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from train_siit import all_hidden, xor
from das_interchange import load_model, load_split


def validate_seed(seed):
    checkpoint = ROOT / "checkpoints" / f"3hop_L3_siit_k1_seed{seed}.pt"
    dataset = ROOT / "data" / "boolean_combined.csv"

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, task = load_model(str(checkpoint))
    model.eval()

    X, _ = load_split(str(dataset), task, "test")

    generator = torch.Generator().manual_seed(123)
    R = X
    D = X[torch.randperm(len(X), generator=generator)]

    with torch.no_grad():
        hsR = all_hidden(model, R)
        hsD = all_hidden(model, D)

        h = hsR[0].clone()
        h[..., 0] = hsD[0][..., 0]
        h = model.encoder.layers[1](h)

        w = ck["siit_heads"]["v_L1.weight"]
        b = ck["siit_heads"]["v_L1.bias"]

        v_pred = (
            (h[..., 0:1] @ w.T + b).squeeze(-1) > 0
        ).float()

        uR = xor(R[:, 0], R[:, 1])
        uD = xor(D[:, 0], D[:, 1])

        v_clean = uR * R[:, 2]
        v_cf = uD * R[:, 2]

        y_clean = xor(v_clean, R[:, 3])
        y_cf = xor(v_cf, R[:, 3])

        changed = y_cf != y_clean
        n_changed = int(changed.sum())

        if n_changed == 0:
            raise ValueError("No changed-output bits found")

        y_u = (
            model.head(model.encoder.layers[2](h))
            .squeeze(-1) > 0
        ).float()

        h_restore = h.clone()
        h_restore[..., 0] = hsR[1][..., 0]

        y_restore = (
            model.head(model.encoder.layers[2](h_restore))
            .squeeze(-1) > 0
        ).float()

        return {
            "seed": seed,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "task": task,
            "total_bits": int(y_cf.numel()),
            "changed_bits": n_changed,
            "pairing_seed": 123,
            "intervention": "swap_u_L0_restore_v_L1",
            "v_propagation": (
                (v_pred == v_cf)[changed].float().mean().item()
            ),
            "y_counterfactual": (
                (y_u == y_cf)[changed].float().mean().item()
            ),
            "v_mediation": (
                (y_restore == y_clean)[changed].float().mean().item()
            ),
        }


def main():
    results = []

    for seed in range(5):
        result = validate_seed(seed)
        results.append(result)

        print(
            f"seed={seed} "
            f"changed={result['changed_bits']} "
            f"v_propagation={result['v_propagation']:.4f} "
            f"y_counterfactual={result['y_counterfactual']:.4f} "
            f"v_mediation={result['v_mediation']:.4f}",
            flush=True,
        )

    output = ROOT / "results" / "3hop_causal_chain_validation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
