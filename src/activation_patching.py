import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_transformer import BitVectorTransformer, load_split

SEED = 42
PAIRS_PER_GROUP = 100
CHECKPOINT = Path("checkpoints/2hop_L2_seed0.pt")
DATA_PATH = Path("data/boolean_combined.csv")
OUT_PATH = Path("results/targeted_activation_patching_2hop.csv")


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    X, Y = load_split(DATA_PATH, "2hop", "test")
    X, Y = X.to(device), Y.to(device)

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    model = BitVectorTransformer(n_inputs=3, layers=2).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Capture each encoder layer's output on the clean test examples.
    clean_acts = {}
    handles = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        def capture(module, inputs, output, idx=layer_idx):
            clean_acts[idx] = output.detach().clone()
        handles.append(layer.register_forward_hook(capture))

    with torch.no_grad():
        clean_logits = model(X)
        clean_predictions = (clean_logits >= 0).float()
        clean_exact = (clean_predictions == Y).all(dim=1).float()

    for handle in handles:
        handle.remove()

    x_np = X.detach().cpu().numpy()
    y_np = Y.detach().cpu().numpy()
    a, b, c = x_np[:, 0, :], x_np[:, 1, :], x_np[:, 2, :]
    u = np.logical_xor(a > 0.5, b > 0.5)
    records = []

    # Match recipient/donor pairs on c at a single bit position.
    for bit in range(16):
        for group, want_u_change in [
            ("u_changes", True),
            ("u_same_control", False),
        ]:
            candidates = []
            for r in range(len(x_np)):
                for d in range(len(x_np)):
                    if r == d or c[r, bit] != c[d, bit]:
                        continue
                    differs = u[r, bit] != u[d, bit]
                    if differs == want_u_change:
                        candidates.append((r, d))

            if not candidates:
                continue

            rng.shuffle(candidates)
            for recipient, donor in candidates[:PAIRS_PER_GROUP]:
                for layer_idx, layer in enumerate(model.encoder.layers):
                    donor_act = clean_acts[layer_idx][donor:donor + 1]

                    def patch(module, inputs, output,
                              donor_activation=donor_act, pos=bit):
                        changed = output.clone()
                        changed[:, pos, :] = donor_activation[:, pos, :]
                        return changed

                    handle = layer.register_forward_hook(patch)
                    with torch.no_grad():
                        patched_logits = model(X[recipient:recipient + 1])
                    handle.remove()

                    patched_bits = (patched_logits >= 0).float()[0].cpu().numpy()
                    clean_bits = clean_predictions[recipient].cpu().numpy()
                    donor_bits = clean_predictions[donor].cpu().numpy()

                    records.append({
                        "bit_position": bit,
                        "layer": layer_idx + 1,
                        "group": group,
                        "recipient_index": recipient,
                        "donor_index": donor,
                        "recipient_u": int(u[recipient, bit]),
                        "donor_u": int(u[donor, bit]),
                        "c_bit": int(c[recipient, bit]),
                        "recipient_target_bit": int(y_np[recipient, bit]),
                        "patched_bit": int(patched_bits[bit]),
                        "clean_recipient_bit_correct": int(
                            clean_bits[bit] == y_np[recipient, bit]
                        ),
                        "patched_bit_correct": int(
                            patched_bits[bit] == y_np[recipient, bit]
                        ),
                        "patched_bit_matches_donor": int(
                            patched_bits[bit] == donor_bits[bit]
                        ),
                        "clean_exact_accuracy": float(
                            clean_exact[recipient].item()
                        ),
                    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)

    print("Test examples:", len(X))
    print("Rows saved:", len(df))
    print("Saved:", OUT_PATH)
    if len(df):
        print("\nSummary by layer and group:")
        print(df.groupby(["layer", "group"])[
            ["clean_recipient_bit_correct", "patched_bit_correct",
             "patched_bit_matches_donor"]
        ].mean().to_string())
    else:
        print("No eligible pairs found.")


if __name__ == "__main__":
    main()
