from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from train_transformer import BitVectorTransformer, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "boolean_combined.csv"
OUT = ROOT / "results"
PAIRS_PER_BIT_GROUP = 100
RNG_SEED = 2026


def make_pairs(intermediate, downstream, treatment, rng, task):
    """Match pairs on downstream bit; vary or hold intermediate bit fixed."""
    pairs = []
    downstream_values = [1] if task == "2hop" else [0, 1]

    for downstream_value in downstream_values:
        eligible = np.flatnonzero(downstream == downstream_value)
        if treatment:
            left = eligible[intermediate[eligible] == 0].copy()
            right = eligible[intermediate[eligible] == 1].copy()
            rng.shuffle(left)
            rng.shuffle(right)
            pairs.extend(zip(left, right))
        else:
            for value in (0, 1):
                indices = eligible[intermediate[eligible] == value].copy()
                rng.shuffle(indices)
                pairs.extend(zip(indices[::2], indices[1::2]))

    pairs = [(int(a), int(b)) for a, b in pairs if a != b]
    rng.shuffle(pairs)
    return pairs[:PAIRS_PER_BIT_GROUP]


def run_checkpoint(task, seed, rng):
    checkpoint_path = ROOT / "checkpoints" / f"{task}_L2_seed{seed}.pt"
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    X, Y = load_split(DATA, task, "test")
    X, Y = X.to(device), Y.to(device)

    state = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = BitVectorTransformer(
        n_inputs=3 if task == "2hop" else 4, layers=2
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    activations = {}
    handles = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        handles.append(layer.register_forward_hook(
            lambda module, inputs, output, idx=layer_idx:
                activations.__setitem__(idx, output.detach().clone())
        ))

    with torch.no_grad():
        clean_logits = model(X)
    for handle in handles:
        handle.remove()

    clean_pred = (clean_logits >= 0).cpu().numpy().astype(int)
    targets = Y.cpu().numpy().astype(int)
    x = X.cpu().numpy()
    a, b, c = x[:, 0, :], x[:, 1, :], x[:, 2, :]
    u = np.logical_xor(a > 0.5, b > 0.5).astype(int)

    if task == "2hop":
        intermediate = u
        downstream = (c > 0.5).astype(int)
    else:
        d = x[:, 3, :]
        intermediate = np.logical_and(u > 0, c > 0.5).astype(int)
        downstream = (d > 0.5).astype(int)

    records = []
    for bit in range(16):
        for group, treatment in (
            ("intermediate_changes", True),
            ("same_intermediate_control", False),
        ):
            pairs = make_pairs(
                intermediate[:, bit], downstream[:, bit],
                treatment, rng, task
            )
            if not pairs:
                continue

            recipients = np.array([p[0] for p in pairs], dtype=int)
            donors = np.array([p[1] for p in pairs], dtype=int)

            for layer_idx, layer in enumerate(model.encoder.layers):
                donor_acts = activations[layer_idx][
                    torch.as_tensor(donors, device=device)
                ]

                def patch(module, inputs, output, donor_acts=donor_acts, pos=bit):
                    changed = output.clone()
                    changed[:, pos, :] = donor_acts[:, pos, :]
                    return changed

                handle = layer.register_forward_hook(patch)
                try:
                    with torch.no_grad():
                        patched_logits = model(
                            X[torch.as_tensor(recipients, device=device)]
                        )
                finally:
                    handle.remove()

                patched = (
                    (patched_logits[:, bit] >= 0)
                    .cpu().numpy().astype(int)
                )

                for i, (recipient, donor) in enumerate(pairs):
                    records.append({
                        "task": task,
                        "seed": seed,
                        "checkpoint": checkpoint_path.name,
                        "layer": layer_idx + 1,
                        "bit": bit,
                        "group": group,
                        "recipient_index": recipient,
                        "donor_index": donor,
                        "recipient_intermediate": int(intermediate[recipient, bit]),
                        "donor_intermediate": int(intermediate[donor, bit]),
                        "recipient_downstream_bit": int(downstream[recipient, bit]),
                        "recipient_target_bit": int(targets[recipient, bit]),
                        "donor_target_bit": int(targets[donor, bit]),
                        "clean_bit_correct": int(
                            clean_pred[recipient, bit] == targets[recipient, bit]
                        ),
                        "patched_bit": int(patched[i]),
                        "patched_bit_correct": int(
                            patched[i] == targets[recipient, bit]
                        ),
                        "patched_matches_donor_target": int(
                            patched[i] == targets[donor, bit]
                        ),
                        "prediction_flipped": int(
                            patched[i] != clean_pred[recipient, bit]
                        ),
                    })

    return records


def main():
    rng = np.random.default_rng(RNG_SEED)
    all_records = []

    for task in ("2hop", "3hop"):
        for seed in (0, 1, 2):
            print(f"Evaluating {task}, seed {seed}...", flush=True)
            all_records.extend(run_checkpoint(task, seed, rng))

    OUT.mkdir(exist_ok=True)
    raw = pd.DataFrame(all_records)
    raw_path = OUT / "learned_model_patching_all_checkpoints.csv"
    raw.to_csv(raw_path, index=False)

    summary = raw.groupby(
        ["task", "seed", "checkpoint", "layer", "group"], as_index=False
    ).agg(
        n=("prediction_flipped", "size"),
        clean_bit_accuracy=("clean_bit_correct", "mean"),
        patched_bit_accuracy=("patched_bit_correct", "mean"),
        prediction_flip_rate=("prediction_flipped", "mean"),
        donor_target_agreement=("patched_matches_donor_target", "mean"),
    )

    # No pre-existing learned-model pass threshold was defined.
    # Record measured evidence; do not manufacture PASS/FAIL labels.
    summary["filter_status"] = "INCONCLUSIVE_NO_PREDEFINED_THRESHOLD"
    summary_path = OUT / "learned_model_patching_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
