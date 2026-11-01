
"""Intervention-based feature attribution analysis.

Measures changes in target feature activations after intervening
on a source feature.

This is an experimental sensitivity measure, not automatic proof
of causality. The intervention mechanism must be validated.
"""

import torch


def measure_feature_effect(
    baseline_targets: torch.Tensor,
    intervened_targets: torch.Tensor,
):
    """Measure changes in target feature activations.

    Both tensors must have shape:
        [batch, sequence, features]

    Returns aggregate effect measurements.
    """
    if not isinstance(baseline_targets, torch.Tensor):
        raise TypeError("baseline_targets must be a torch.Tensor.")

    if not isinstance(intervened_targets, torch.Tensor):
        raise TypeError("intervened_targets must be a torch.Tensor.")

    if baseline_targets.shape != intervened_targets.shape:
        raise ValueError("Baseline and intervention shapes must match.")

    if baseline_targets.ndim != 3:
        raise ValueError(
            "Feature tensors must have shape [batch, sequence, features]."
        )

    if baseline_targets.numel() == 0:
        raise ValueError("Feature tensors cannot be empty.")

    if not torch.isfinite(baseline_targets).all():
        raise ValueError("Baseline features must be finite.")

    if not torch.isfinite(intervened_targets).all():
        raise ValueError("Intervened features must be finite.")

    difference = baseline_targets - intervened_targets
    absolute_difference = difference.abs()

    per_feature_effect = absolute_difference.mean(dim=(0, 1))

    return {
        "mean_absolute_effect": float(
            absolute_difference.mean().item()
        ),
        "max_absolute_effect": float(
            absolute_difference.max().item()
        ),
        "per_feature_effect": per_feature_effect.detach().cpu().tolist(),
        "fraction_features_affected": float(
            (per_feature_effect > 0).float().mean().item()
        ),
    }


def rank_target_features(effect_result, top_k=10):
    """Rank target features by measured intervention effect."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    effects = effect_result["per_feature_effect"]

    if not effects:
        raise ValueError("No target feature effects were provided.")

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not torch.isfinite(torch.tensor(value)).item()
        for value in effects
    ):
        raise ValueError("Feature effects must be finite numbers.")

    ranked = sorted(
        enumerate(effects),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "target_feature": feature_id,
            "mean_absolute_effect": float(effect),
        }
        for feature_id, effect in ranked[:top_k]
    ]
