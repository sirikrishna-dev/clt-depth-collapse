
import pytest
import torch

from src.metrics.causal_attribution import (
    measure_feature_effect,
    rank_target_features,
)


def test_measure_feature_effect():
    baseline = torch.tensor([[[0.8, 0.5, 0.9]]])
    intervened = torch.tensor([[[0.8, 0.1, 0.9]]])

    result = measure_feature_effect(baseline, intervened)

    assert result["mean_absolute_effect"] == pytest.approx(0.4 / 3)
    assert result["max_absolute_effect"] == pytest.approx(0.4)
    assert result["per_feature_effect"] == pytest.approx([0.0, 0.4, 0.0])
    assert result["fraction_features_affected"] == pytest.approx(1 / 3)


def test_rank_target_features():
    result = {
        "per_feature_effect": [0.1, 0.8, 0.3]
    }

    ranked = rank_target_features(result, top_k=2)

    assert ranked == [
        {"target_feature": 1, "mean_absolute_effect": 0.8},
        {"target_feature": 2, "mean_absolute_effect": 0.3},
    ]


def test_mismatched_shapes_are_rejected():
    baseline = torch.zeros(1, 1, 3)
    intervened = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError):
        measure_feature_effect(baseline, intervened)


def test_non_finite_values_are_rejected():
    baseline = torch.tensor([[[float("nan")]]])
    intervened = torch.zeros(1, 1, 1)

    with pytest.raises(ValueError):
        measure_feature_effect(baseline, intervened)


def test_invalid_top_k_is_rejected():
    result = {"per_feature_effect": [0.1, 0.2]}

    with pytest.raises(ValueError):
        rank_target_features(result, top_k=0)


def test_empty_effects_are_rejected():
    result = {"per_feature_effect": []}

    with pytest.raises(ValueError):
        rank_target_features(result)
