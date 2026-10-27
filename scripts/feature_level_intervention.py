
from pathlib import Path
import csv
import torch
import torch.nn.functional as F

from _person2_src.person2_loader.model_adapter import (
    load_original_checkpoint,
    load_task_split,
)
from _person2_src.person2_loader.checkpoint_io import (
    load_replacement_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
TOP_K = 5
SEED = 42

ORIGINAL_CKPT = ROOT / "checkpoints" / "3hop_L2_seed0.pt"
DATA_PATH = ROOT / "data" / "boolean_3hop.csv"
CLT_CKPT = ROOT.parent / "3hop_clt_seed0_features128_span2.pt"
OUT_CSV = ROOT / "feature_level_intervention_results.csv"


def accuracy(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    bit_acc = (preds == targets).float().mean().item()
    exact_acc = (preds == targets).all(dim=1).float().mean().item()
    return bit_acc, exact_acc, preds


def evaluate(model, x, y, batch_size=BATCH_SIZE, intervention=None):
    """intervention: dict with source, target, feature, mode; or None."""
    model.eval()
    all_logits = []
    changed_count = 0
    baseline_preds = []

    # Capture the original model's MLP input for each source layer.
    captured_inputs = {}
    handles = []

    for source_layer, layer in enumerate(model.encoder.layers):
        def make_input_hook(idx):
            def hook(module, args):
                captured_inputs[idx] = args[0]
            return hook

        handles.append(
            layer.linear1.register_forward_pre_hook(
                make_input_hook(source_layer)
            )
        )

    if intervention is not None:
        source = intervention["source"]
        target = intervention["target"]
        feature = intervention["feature"]
        mode = intervention["mode"]

        def write_hook(module, args, output):
            if source not in captured_inputs:
                raise RuntimeError(
                    f"Source layer {source} input was not captured."
                )

            # Feature activation from the original MLP input.
            with torch.no_grad():
                z = F.relu(clt.encoders[source](
                    captured_inputs[source]
                ))
                z_f = torch.zeros_like(z)
                z_f[..., feature] = z[..., feature]

                # This feature's decoder contribution to the target layer.
                contribution = clt.decoders[
                    f"{source}_{target}"
                ](z_f)

                if mode == "remove":
                    # Subtract the estimated feature write.
                    return output - contribution

                if mode == "shuffle":
                    # Control: same contribution, but assigned to
                    # different examples in the batch.
                    if contribution.shape[0] > 1:
                        control = contribution.roll(
                            shifts=1, dims=0
                        )
                    else:
                        control = contribution
                    return output - control

                raise ValueError(f"Unknown mode: {mode}")

        handles.append(
            model.encoder.layers[target].linear2.register_forward_hook(
                write_hook
            )
        )

    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = x[start:start + batch_size].to(DEVICE)
                yb = y[start:start + batch_size].to(DEVICE)

                captured_inputs.clear()
                logits = model(xb)
                all_logits.append(logits.cpu())

    finally:
        for handle in handles:
            handle.remove()

    logits = torch.cat(all_logits, dim=0)
    bit_acc, exact_acc, preds = accuracy(logits, y)
    return {
        "bit_accuracy": bit_acc,
        "exact_accuracy": exact_acc,
        "predictions": preds,
        "logits": logits,
    }


def main():
    global clt

    torch.manual_seed(SEED)

    if not ORIGINAL_CKPT.is_file():
        raise FileNotFoundError(ORIGINAL_CKPT)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    if not CLT_CKPT.is_file():
        raise FileNotFoundError(CLT_CKPT)

    original, original_meta = load_original_checkpoint(
        ORIGINAL_CKPT, DEVICE
    )
    clt, clt_meta = load_replacement_checkpoint(CLT_CKPT, DEVICE)

    if clt_meta.get("model_type") != "clt":
        raise ValueError("Expected a CLT checkpoint.")
    if clt.n_layers != len(original.encoder.layers):
        raise ValueError("Original and CLT layer counts differ.")

    # Select candidate features on TRAIN, not on the test set.
    x_train, _ = load_task_split(DATA_PATH, "3hop", "train")
    train_inputs = {}
    train_handles = []

    for idx, layer in enumerate(original.encoder.layers):
        def make_hook(layer_idx):
            def hook(module, args):
                train_inputs[layer_idx] = args[0].detach()
            return hook

        train_handles.append(
            layer.linear1.register_forward_pre_hook(make_hook(idx))
        )

    try:
        activation_sums = [
            torch.zeros(clt.n_features, device=DEVICE)
            for _ in range(clt.n_layers)
        ]
        activation_counts = 0

        with torch.no_grad():
            for start in range(0, len(x_train), BATCH_SIZE):
                xb = x_train[start:start + BATCH_SIZE].to(DEVICE)
                train_inputs.clear()
                original(xb)

                for layer_idx in range(clt.n_layers):
                    z = F.relu(
                        clt.encoders[layer_idx](
                            train_inputs[layer_idx]
                        )
                    )
                    activation_sums[layer_idx] += z.sum(
                        dim=(0, 1)
                    )
                activation_counts += (
                    xb.shape[0] * train_inputs[0].shape[1]
                )
    finally:
        for handle in train_handles:
            handle.remove()

    mean_acts = [
        sums / max(activation_counts, 1)
        for sums in activation_sums
    ]
    candidates = {
        layer_idx: torch.topk(
            mean_acts[layer_idx],
            k=min(TOP_K, clt.n_features)
        ).indices.tolist()
        for layer_idx in range(clt.n_layers)
    }

    x_test, y_test = load_task_split(DATA_PATH, "3hop", "test")
    baseline = evaluate(original, x_test, y_test)
    baseline_preds = baseline["predictions"]

    print("\nBASELINE")
    print(f"Bit accuracy: {baseline['bit_accuracy']:.4f}")
    print(f"Exact-vector accuracy: {baseline['exact_accuracy']:.4f}")
    baseline_logits = baseline["logits"]
    baseline_margins = baseline_logits.abs()

    print(
        f"Mean absolute baseline logit: "
        f"{baseline_margins.mean().item():.6f}"
    )
    print(
        f"Minimum absolute baseline logit: "
        f"{baseline_margins.min().item():.6f}"
    )
    rows = []
    edges = [(0, 0), (0, 1), (1, 1)]

    for source, target in edges:
        if f"{source}_{target}" not in clt.decoders:
            continue

        for feature in candidates[source]:
            for mode in ("remove", "shuffle"):
                # Fresh model ensures no state is carried between tests.
                model, _ = load_original_checkpoint(
                    ORIGINAL_CKPT, DEVICE
                )
                result = evaluate(
                    model, x_test, y_test,
                    intervention={
                        "source": source,
                        "target": target,
                        "feature": feature,
                        "mode": mode,
                    },
                )

                logit_change = (
                    result["logits"] - baseline["logits"]
                ).abs()
                baseline_abs = baseline["logits"].abs()
                intervened_abs = result["logits"].abs()

                minimum_intervened_margin = intervened_abs.min().item()
                minimum_baseline_margin = baseline_abs.min().item()

                margin_reductions = baseline_abs - intervened_abs
                max_margin_reduction = margin_reductions.max().item()

                mean_logit_change = logit_change.mean().item()
                max_logit_change = logit_change.max().item()

                changed = (
                    result["predictions"] != baseline_preds
                ).any(dim=1).float().mean().item()

                
                row = {
                    "source_layer": source,
                    "target_layer": target,
                    "feature": feature,
                    "condition": mode,
                    "bit_accuracy": result["bit_accuracy"],
                    "exact_accuracy": result["exact_accuracy"],
                    "fraction_examples_changed": changed,
                    "mean_absolute_logit_change": mean_logit_change,
                    "max_absolute_logit_change": max_logit_change,
                    "minimum_baseline_margin": minimum_baseline_margin,
                    "minimum_intervened_margin": minimum_intervened_margin,
                    "max_margin_reduction": max_margin_reduction,
                
                }

                rows.append(row)
                                
                print(
                    f"edge={source}->{target} feature={feature:3d} "
                    f"condition={mode:7s} "
                    f"bit_acc={row['bit_accuracy']:.4f} "
                    f"exact_acc={row['exact_accuracy']:.4f} "
                    f"changed={changed:.4f} "
                    f"mean_logit_change={mean_logit_change:.8f} "
                    f"max_logit_change={max_logit_change:.8f}"
                )


    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {OUT_CSV}")
    print(
        "\nCaution: the CLT is a reconstruction model. These tests "
        "measure sensitivity to subtracting its estimated feature "
        "contribution from the original model's MLP write; they do "
        "not by themselves establish the original model's true "
        "computational graph or depth."
    )


if __name__ == "__main__":
    main()
