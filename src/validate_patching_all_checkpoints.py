"""Exploratory learned-model activation patching with layer-hook diagnostics.

Important: this patches the full hidden vector at one bit-position token. It does
not selectively patch a represented u/v feature, so results are screening evidence,
not proof of a selective causal mechanism.
"""
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


def build_pairs(values, keys, treatment, rng, max_pairs=PAIRS_PER_BIT_GROUP, indices=None):
    """Pair rows within matched keys, varying or holding the intermediate fixed.

    values: binary intermediate array; keys: tuples of downstream-relevant bits.
    """
    if indices is None:
        indices = np.arange(len(values), dtype=int)
    buckets = {}
    for i, key in enumerate(keys):
        buckets.setdefault(tuple(key), {0: [], 1: []})[int(values[i])].append(i)

    pairs = []
    for key in sorted(buckets, key=str):
        zero = np.asarray(buckets[key][0], dtype=int)
        one = np.asarray(buckets[key][1], dtype=int)
        rng.shuffle(zero)
        rng.shuffle(one)
        if treatment:
            pairs.extend((int(indices[a]), int(indices[b])) for a, b in zip(zero, one))
        else:
            for group in (zero, one):
                pairs.extend(
                    (int(indices[a]), int(indices[b]))
                    for a, b in zip(group[::2], group[1::2])
                )
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def run_checkpoint(task, seed, rng):
    checkpoint_path = ROOT / "checkpoints" / f"{task}_L2_seed{seed}.pt"
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    X, Y = load_split(DATA, task, "test")
    X, Y = X.to(device), Y.to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BitVectorTransformer(
        n_inputs=3 if task == "2hop" else 4, layers=2
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    clean_acts, hook_calls = {}, {0: 0, 1: 0}
    handles = []
    for idx, layer in enumerate(model.encoder.layers):
        def save_hook(module, inputs, output, idx=idx):
            hook_calls[idx] += 1
            clean_acts[idx] = output.detach().clone()
        handles.append(layer.register_forward_hook(save_hook))

    try:
        with torch.no_grad():
            clean_logits = model(X)
    finally:
        for handle in handles:
            handle.remove()

    clean_pred = (clean_logits >= 0).cpu().numpy().astype(int)
    targets = Y.cpu().numpy().astype(int)
    x = X.cpu().numpy()
    a, b, c = x[:, 0, :], x[:, 1, :], x[:, 2, :]
    u = np.logical_xor(a > 0.5, b > 0.5).astype(int)
    d = x[:, 3, :] if task == "3hop" else None
    v = np.logical_and(u > 0, c > 0.5).astype(int) if task == "3hop" else None

    layer1 = clean_acts[0].float()
    layer2 = clean_acts[1].float()
    layer_delta = (layer1 - layer2).abs()
    diagnostics = {
        "task": task,
        "seed": seed,
        "checkpoint": checkpoint_path.name,
        "clean_forward_hook_calls_layer1": hook_calls[0],
        "clean_forward_hook_calls_layer2": hook_calls[1],
        "clean_activation_shape_layer1": str(tuple(layer1.shape)),
        "clean_activation_shape_layer2": str(tuple(layer2.shape)),
        "layers_exactly_equal": bool(torch.equal(layer1, layer2)),
        "layers_mean_absolute_activation_difference": float(layer_delta.mean().item()),
        "layers_fraction_elements_different": float(
            (layer_delta > 1e-7).float().mean().item()
        ),
        "clean_bit_accuracy": float((clean_pred == targets).mean()),
        "clean_exact_vector_accuracy": float(np.all(clean_pred == targets, axis=1).mean()),
    }

    records = []
    for bit in range(16):
        if task == "2hop":
            # y = u AND c. Match c=1 so changing u changes y; control also fixes c=1.
            valid = np.flatnonzero(x[:, 2, bit] > 0.5)
            vals = u[valid, bit]
            keys = [(1,) for _ in valid]
            pair_specs = [
                ("u_treatment", True, valid, vals, keys),
                ("u_same_value_control", False, valid, vals, keys),
            ]
            intermediate_values = {"u": u[:, bit]}
        else:
            # y = v XOR d. u test fixes c=1 and d; v test fixes d.
            u_valid = np.flatnonzero(x[:, 2, bit] > 0.5)
            u_pairs = build_pairs(
                u[u_valid, bit],
                [(int(d[i, bit]),) for i in u_valid],
                True, rng, indices=u_valid,
            )
            u_controls = build_pairs(
                u[u_valid, bit],
                [(int(d[i, bit]),) for i in u_valid],
                False, rng, indices=u_valid,
            )
            # v can be tested while matching d, with v differing / held equal.
            v_keys = [(int(d[i, bit]),) for i in range(len(X))]
            v_pairs = build_pairs(v[:, bit], v_keys, True, rng)
            v_controls = build_pairs(v[:, bit], v_keys, False, rng)
            pair_specs = [
                ("u_treatment_c1_d_matched", True, None, None, None),
                ("u_same_value_control_c1_d_matched", False, None, None, None),
                ("v_treatment_d_matched", True, None, None, None),
                ("v_same_value_control_d_matched", False, None, None, None),
            ]
            intermediate_values = {"u": u[:, bit], "v": v[:, bit]}

        for group, treatment, valid, vals, keys in pair_specs:
            if task == "2hop":
                pairs = build_pairs(vals, keys, treatment, rng, indices=valid)
                intermediate_name = "u"
                value_array = intermediate_values["u"]
                match_condition = "c=1"
            elif group.startswith("u_"):
                pairs = u_pairs if treatment else u_controls
                intermediate_name = "u"
                value_array = intermediate_values["u"]
                match_condition = "c=1 and d matched"
            else:
                pairs = v_pairs if treatment else v_controls
                intermediate_name = "v"
                value_array = intermediate_values["v"]
                match_condition = "d matched"

            if not pairs:
                continue

            recipients = np.asarray([p[0] for p in pairs], dtype=int)
            donors = np.asarray([p[1] for p in pairs], dtype=int)

            # Guard the matching logic: treatment differs in target intermediate;
            # controls hold it fixed. For u, c is fixed at 1 and d is matched in 3-hop.
            if treatment and not np.all(value_array[recipients] != value_array[donors]):
                raise RuntimeError(f"Treatment pair mismatch: {task}/{group}")
            if not treatment and not np.all(value_array[recipients] == value_array[donors]):
                raise RuntimeError(f"Control pair mismatch: {task}/{group}")

            recipient_tensor = torch.as_tensor(recipients, device=device)
            for layer_idx, layer in enumerate(model.encoder.layers):
                donor_tensor = torch.as_tensor(donors, device=device)
                donor_acts = clean_acts[layer_idx][donor_tensor]
                patch_diag = {
                    "calls": 0,
                    "patched_token_delta_sum": 0.0,
                    "patched_token_delta_n": 0,
                }

                def patch_hook(
                    module, inputs, output, donor_acts=donor_acts,
                    bit=bit, patch_diag=patch_diag
                ):
                    patch_diag["calls"] += 1
                    changed = output.clone()
                    before = output[:, bit, :].detach().float()
                    donor = donor_acts[:, bit, :].to(
                        device=output.device, dtype=output.dtype
                    )
                    patch_diag["patched_token_delta_sum"] += float(
                        (before - donor.float()).abs().sum().item()
                    )
                    patch_diag["patched_token_delta_n"] += int(before.numel())
                    changed[:, bit, :] = donor
                    return changed

                handle = layer.register_forward_hook(patch_hook)
                try:
                    with torch.no_grad():
                        logits = model(X[recipient_tensor])
                finally:
                    handle.remove()

                # Preserve both continuous logit values and thresholded predictions.
                clean_bit_logits = clean_logits[recipient_tensor, bit]
                patched_bit_logits = logits[:, bit]
                clean_bit_logits_np = clean_bit_logits.detach().cpu().numpy()
                patched_bit_logits_np = patched_bit_logits.detach().cpu().numpy()
                patched = (patched_bit_logits >= 0).cpu().numpy().astype(int)

                for i, (recipient, donor) in enumerate(pairs):
                    clean_logit_value = float(clean_bit_logits_np[i])
                    patched_logit_value = float(patched_bit_logits_np[i])
                    records.append({
                        "task": task,
                        "seed": seed,
                        "checkpoint": checkpoint_path.name,
                        "intermediate_tested": intermediate_name,
                        "match_condition": match_condition,
                        "layer": layer_idx + 1,
                        "bit": bit,
                        "group": group,
                        "recipient_index": recipient,
                        "donor_index": donor,
                        "recipient_intermediate": int(value_array[recipient]),
                        "donor_intermediate": int(value_array[donor]),
                        "recipient_target_bit": int(targets[recipient, bit]),
                        "donor_target_bit": int(targets[donor, bit]),
                        "clean_bit_correct": int(
                            clean_pred[recipient, bit] == targets[recipient, bit]
                        ),
                        "clean_bit_logit": clean_logit_value,
                        "patched_bit_logit": patched_logit_value,
                        "logit_abs_delta": abs(patched_logit_value - clean_logit_value),
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
                        "hook_calls_for_batch": patch_diag["calls"],
                        "patched_token_mean_abs_delta": (
                            patch_diag["patched_token_delta_sum"]
                            / max(1, patch_diag["patched_token_delta_n"])
                        ),
                    })
    return records, diagnostics


def main():
    rng = np.random.default_rng(RNG_SEED)
    all_records, diagnostics = [], []
    for task in ("2hop", "3hop"):
        for seed in (0, 1, 2):
            print(f"Evaluating {task}, seed {seed}...", flush=True)
            records, diag = run_checkpoint(task, seed, rng)
            all_records.extend(records)
            diagnostics.append(diag)

    OUT.mkdir(exist_ok=True)
    raw = pd.DataFrame(all_records)
    raw.to_csv(OUT / "learned_model_patching_all_checkpoints.csv", index=False)

    summary = raw.groupby(
        [
            "task", "seed", "checkpoint", "intermediate_tested",
            "match_condition", "layer", "group",
        ],
        as_index=False,
    ).agg(
        n=("prediction_flipped", "size"),
        clean_bit_accuracy=("clean_bit_correct", "mean"),
        patched_bit_accuracy=("patched_bit_correct", "mean"),
        prediction_flip_rate=("prediction_flipped", "mean"),
        donor_target_agreement=("patched_matches_donor_target", "mean"),
        hook_calls_per_row=("hook_calls_for_batch", "mean"),
        patched_token_mean_abs_delta=("patched_token_mean_abs_delta", "mean"),
        mean_logit_abs_delta=("logit_abs_delta", "mean"),
        median_logit_abs_delta=("logit_abs_delta", "median"),
    )
    summary["screen_status"] = "EXPLORATORY_RESULT_NOT_CAUSAL_VALIDATION"
    summary.to_csv(OUT / "learned_model_patching_summary.csv", index=False)

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(OUT / "layer_hook_diagnostics.csv", index=False)

    status_rows = []
    for checkpoint, part in summary.groupby("checkpoint"):
        # Exploratory criterion only, computed separately for every tested intermediate/layer/group.
        tr = part[part.group.str.contains("treatment")]
        co = part[part.group.str.contains("control")]
        layers_ok = set(tr.layer) == {1, 2} and set(co.layer) == {1, 2}
        clean_ok = bool(part.clean_bit_accuracy.min() >= 0.99)
        treatment_ok = bool(len(tr) and (tr.prediction_flip_rate >= 0.90).all())
        control_ok = bool(len(co) and (co.prediction_flip_rate <= 0.10).all())
        status_rows.append({
            "checkpoint": checkpoint,
            "clean_bit_accuracy_min": float(part.clean_bit_accuracy.min()),
            "treatment_flip_rate_min": float(tr.prediction_flip_rate.min()) if len(tr) else float("nan"),
            "control_flip_rate_max": float(co.prediction_flip_rate.max()) if len(co) else float("nan"),
            "screen_status": (
                "PASS_EXPLORATORY"
                if layers_ok and clean_ok and treatment_ok and control_ok
                else "FAIL_OR_INCOMPLETE_EXPLORATORY"
            ),
            "criterion": (
                "post-hoc: clean accuracy >= 0.99; every treatment group/layer "
                ">= 0.90 flips; every control group/layer <= 0.10 flips"
            ),
            "causal_validation_status": "INCOMPLETE",
            "caveat": (
                "Full token-vector patching; not selective u/v feature patching; "
                "post-hoc criterion; layer equivalence and hook diagnostics must be reviewed"
            ),
        })
    status = pd.DataFrame(status_rows)
    status.to_csv(OUT / "patching_filter_checkpoint_status.csv", index=False)

    print("\nLayer hook diagnostics:\n", diag_df.to_string(index=False))
    print("\nCheckpoint exploratory screen:\n", status.to_string(index=False))
    print(f"\nSaved raw: {OUT / 'learned_model_patching_all_checkpoints.csv'}")
    print(f"Saved summary: {OUT / 'learned_model_patching_summary.csv'}")
    print(f"Saved diagnostics: {OUT / 'layer_hook_diagnostics.csv'}")

    # Keep metadata and model_zoo registry consistent without claiming causal validation.
    registry = []
    for path in sorted((ROOT / "metadata").glob("*_L2_seed*.json")):
        meta = json.loads(path.read_text())
        ckpt = meta["checkpoint"].split("/")[-1]
        sr = status[status.checkpoint == ckpt]
        meta["patching_status"] = (
            "exploratory_screen_pass"
            if len(sr) and sr.iloc[0].screen_status == "PASS_EXPLORATORY"
            else "exploratory_screen_incomplete_or_failed"
        )
        meta["causal_validation_status"] = "incomplete"
        meta["validated"] = False
        meta["patching_evidence"] = {
            "raw_results": "results/learned_model_patching_all_checkpoints.csv",
            "summary": "results/learned_model_patching_summary.csv",
            "hook_diagnostics": "results/layer_hook_diagnostics.csv",
            "status_report": "results/patching_filter_checkpoint_status.csv",
            "intervention": "full hidden vector at one bit-position token",
            "limitation": "not selective intervention on u or v; exploratory screen only",
        }
        path.write_text(json.dumps(meta, indent=2) + "\n")
        registry.append(meta)
    (ROOT / "metadata" / "model_zoo.jsonl").write_text(
        "".join(json.dumps(meta) + "\n" for meta in registry)
    )


if __name__ == "__main__":
    main()
