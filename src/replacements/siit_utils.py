
"""Torch-free helpers for the SIIT PLT/CLT batch experiment."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASK_ALIASES = {
    "2hop": {"2hop", "boolean_2hop_xor_then_and"},
    "3hop": {"3hop", "boolean_3hop_xor_and_then_xor"},
}

N_BITS = 16

_MODEL_ID = re.compile(
    r"^(?P<task>[23]hop)_L(?P<layers>\d+)_siit_k(?P<k>\d+)_seed(?P<seed>\d+)$"
)


def parse_model_id(model_id: str) -> dict[str, Any]:
    match = _MODEL_ID.match(model_id)

    if not match:
        raise ValueError(f"Not a SIIT model id: {model_id!r}")

    return {
        "model_id": model_id,
        "task": match["task"],
        "layers": int(match["layers"]),
        "k": int(match["k"]),
        "seed": int(match["seed"]),
    }


def discover_siit_checkpoints(
    checkpoint_dir: str | Path,
) -> list[dict[str, Any]]:

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    found = []

    for path in checkpoint_dir.glob("*_siit_k*_seed*.pt"):
        try:
            info = parse_model_id(path.stem)
        except ValueError:
            continue

        info["path"] = path
        found.append(info)

    return sorted(
        found,
        key=lambda d: (d["task"], d["seed"])
    )


def default_siit_sites(
    task: str,
    layers: int,
    k: int = 1,
) -> list[dict[str, Any]]:

    if task == "2hop":
        return [
            dict(var="u", layer=0, start=0, end=k)
        ]

    if layers >= 3:
        return [
            dict(var="u", layer=0, start=0, end=k),
            dict(var="v", layer=1, start=0, end=k),
        ]

    return [
        dict(var="v", layer=0, start=0, end=k)
    ]


def load_intermediate_labels(
    data_path: str | Path,
    task: str,
    split: str,
    var: str,
) -> np.ndarray:

    if var not in {"u", "v"}:
        raise ValueError("var must be 'u' or 'v'")

    if task not in TASK_ALIASES:
        raise ValueError(f"Unknown task: {task}")

    frame = pd.read_csv(data_path)

    frame = frame.loc[
        frame["split"].astype(str) == split
    ]

    if "task" in frame.columns:
        rows = frame.loc[
            frame["task"].astype(str).isin(TASK_ALIASES[task])
        ]

        if len(rows):
            frame = rows

    cols = [
        f"{var}_{bit:02d}"
        for bit in range(N_BITS)
    ]

    missing = [
        c for c in cols
        if c not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {var} columns: {missing[:4]}"
        )

    values = frame[cols].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite {var} labels in {split} split"
        )

    return values.astype(np.int64)


def _fit_predict(train_x, train_y, other_xs):

    train_x = np.asarray(
        train_x,
        dtype=np.float32
    )

    train_y = np.asarray(train_y)

    other_xs = [
        np.asarray(x, dtype=np.float32)
        for x in other_xs
    ]

    classes = np.unique(train_y)

    if len(classes) < 2:
        return [
            np.full(
                len(x),
                classes[0],
                dtype=np.int64
            )
            for x in other_xs
        ]

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe.fit(train_x, train_y)

    return [
        probe.predict(x)
        for x in other_xs
    ]


def probe_representation(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    shuffle_seed: int = 42,
    n_shuffles: int = 20,
) -> dict[str, Any]:

    if (
        not isinstance(n_shuffles, int)
        or isinstance(n_shuffles, bool)
        or n_shuffles < 1
    ):
        raise ValueError(
            "n_shuffles must be a positive integer"
        )

    if (
        not isinstance(shuffle_seed, int)
        or isinstance(shuffle_seed, bool)
    ):
        raise ValueError(
            "shuffle_seed must be an integer"
        )

    per_bit = []

    shuffled_per_bit = [
        [] for _ in range(N_BITS)
    ]

    rng = np.random.default_rng(shuffle_seed)

    for bit in range(N_BITS):

        tr_x = np.asarray(
            features["train"][:, bit, :],
            dtype=np.float32
        )

        va_x = np.asarray(
            features["validation"][:, bit, :],
            dtype=np.float32
        )

        te_x = np.asarray(
            features["test"][:, bit, :],
            dtype=np.float32
        )

        tr_y = np.asarray(
            labels["train"][:, bit]
        )

        va_y = np.asarray(
            labels["validation"][:, bit]
        )

        te_y = np.asarray(
            labels["test"][:, bit]
        )

        # Original probe trained using real training labels.
        va_pred, te_pred = _fit_predict(
            tr_x,
            tr_y,
            [va_x, te_x]
        )

        majority = int(
            round(float(tr_y.mean()))
        )

        per_bit.append({
            "bit": bit,

            "validation_accuracy": float(
                accuracy_score(va_y, va_pred)
            ),

            "test_accuracy": float(
                accuracy_score(te_y, te_pred)
            ),

            "test_f1": float(
                f1_score(
                    te_y,
                    te_pred,
                    zero_division=0
                )
            ),

            "test_majority_baseline_accuracy": float(
                (te_y == majority).mean()
            ),
        })

        # Shuffled-label control.
        for _ in range(n_shuffles):

            shuffled_y = tr_y.copy()

            rng.shuffle(shuffled_y)

            shuffled_va_pred, shuffled_te_pred = _fit_predict(
                tr_x,
                shuffled_y,
                [va_x, te_x]
            )

            shuffled_per_bit[bit].append({
                "validation_accuracy": float(
                    accuracy_score(
                        va_y,
                        shuffled_va_pred
                    )
                ),

                "test_accuracy": float(
                    accuracy_score(
                        te_y,
                        shuffled_te_pred
                    )
                ),
            })

    mean = lambda key: float(
        np.mean([
            result[key]
            for result in per_bit
        ])
    )

    # Mean test accuracy across all bits for each shuffle.
    null_scores = [
        float(np.mean([
            shuffled_per_bit[bit][iteration]["test_accuracy"]
            for bit in range(N_BITS)
        ]))
        for iteration in range(n_shuffles)
    ]

    null_mean = float(
        np.mean(null_scores)
    )

    null_std = (
        float(np.std(null_scores, ddof=1))
        if n_shuffles > 1
        else 0.0
    )

    original_mean = mean("test_accuracy")

    empirical_p = float(
        (
            1 + np.sum(
                np.asarray(null_scores) >= original_mean
            )
        )
        / (n_shuffles + 1)
    )

    return {
        "mean_validation_accuracy": mean(
            "validation_accuracy"
        ),

        "mean_test_accuracy": original_mean,

        "mean_test_f1": mean(
            "test_f1"
        ),

        "mean_test_majority_baseline_accuracy": mean(
            "test_majority_baseline_accuracy"
        ),

        "per_bit": per_bit,

        "shuffled_label_control": {
            "seed": shuffle_seed,

            "n_shuffles": n_shuffles,

            "mean_test_accuracy": null_mean,

            "std_test_accuracy": null_std,

            "min_test_accuracy": float(
                np.min(null_scores)
            ),

            "max_test_accuracy": float(
                np.max(null_scores)
            ),

            "percentile_2_5": float(
                np.percentile(null_scores, 2.5)
            ),

            "percentile_97_5": float(
                np.percentile(null_scores, 97.5)
            ),

            "empirical_p_value": empirical_p,

            "per_shuffle_mean_test_accuracy": [
                float(x)
                for x in null_scores
            ],

            "per_bit": [
                {
                    "bit": bit,

                    "mean_test_accuracy": float(
                        np.mean([
                            x["test_accuracy"]
                            for x in shuffled_per_bit[bit]
                        ])
                    ),

                    "std_test_accuracy": float(
                        np.std([
                            x["test_accuracy"]
                            for x in shuffled_per_bit[bit]
                        ])
                    ),
                }
                for bit in range(N_BITS)
            ],
        },
    }



def flatten_recovery(
    model_id: str,
    task: str,
    seed: int,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    rows = []

    for r in results:
        control = r.get("shuffled_label_control", {})

        rows.append({
            "model_id": model_id,
            "task": task,
            "siit_seed": seed,
            "variable": r["variable"],
            "representation": r["representation"],
            "kind": r["kind"],
            "layer": r["layer"],
            "designated_layer": r["designated_layer"],
            "n_dims_per_token": r["n_dims_per_token"],

            "mean_validation_accuracy": r["mean_validation_accuracy"],
            "mean_test_accuracy": r["mean_test_accuracy"],
            "mean_test_f1": r["mean_test_f1"],
            "majority_baseline_test_accuracy": (
                r["mean_test_majority_baseline_accuracy"]
            ),
            "test_accuracy_above_majority": (
                r["mean_test_accuracy"]
                - r["mean_test_majority_baseline_accuracy"]
            ),

            # Shuffled-label control
            "shuffled_test_accuracy": control.get("mean_test_accuracy"),
            "shuffled_test_accuracy_std": control.get("std_test_accuracy"),
            "shuffled_test_accuracy_min": control.get("min_test_accuracy"),
            "shuffled_test_accuracy_max": control.get("max_test_accuracy"),
            "shuffled_empirical_p_value": control.get("empirical_p_value"),
            "n_shuffles": control.get("n_shuffles"),
        })

    return rows



def build_matching_report(
    fidelity: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:

    rows = []

    for model_id, group in fidelity.groupby(
        "model_id",
        sort=True
    ):

        plt_rows = group[
            group["model_type"] == "plt"
        ]

        clt_rows = group[
            group["model_type"] == "clt"
        ]

        base = {
            "model_id": model_id,
            "task": group["task"].iloc[0],
            "tolerance": tolerance,
        }

        if plt_rows.empty or clt_rows.empty:
            rows.append({
                **base,
                "status": "INCOMPLETE"
            })
            continue

        p = plt_rows.iloc[0]
        c = clt_rows.iloc[0]

        p_val = float(
            p["validation_relative_l2"]
        )

        c_val = float(
            c["validation_relative_l2"]
        )

        gap = abs(p_val - c_val)

        rows.append({
            **base,

            "plt_run_id": p["run_id"],
            "clt_run_id": c["run_id"],

            "clt_write_span": c["write_span"],

            "plt_validation_relative_l2": p_val,
            "clt_validation_relative_l2": c_val,

            "absolute_gap": gap,

            "relative_gap_percent": (
                100.0 * gap / max(p_val, 1e-12)
            ),

            "status": (
                "PROVISIONAL_MATCH"
                if gap <= tolerance
                else "NOT_MATCHED"
            ),
        })

    return pd.DataFrame(rows)
