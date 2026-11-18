
import pytest
import torch

from src.metrics.causal_attribution import (
    measure_feature_effect,
    rank_target_features,
)


def test_measure_feature_effect():
    baseline = torch.ones(4, 3, 5)
    intervened = torch.zeros(4, 3, 5)

    result = measure_feature_effect(baseline, intervened)

    assert result["mean_absolute_effect"] == 1.0
    assert result["max_absolute_effect"] == 1.0
    assert result["fraction_features_affected"] == 1.0


def test_identical_features_have_zero_effect():
    baseline = torch.randn(4, 3, 5)

    result = measure_feature_effect(baseline, baseline.clone())

    assert result["mean_absolute_effect"] == 0.0
    assert result["max_absolute_effect"] == 0.0


def test_feature_ranking():
    result = {
        "per_feature_effect": [0.1, 0.8, 0.3, 0.5]
    }

    ranked = rank_target_features(result, top_k=2)

    assert ranked[0]["target_feature"] == 1
    assert ranked[1]["target_feature"] == 3


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        measure_feature_effect(
            torch.ones(4, 3, 5),
            torch.ones(4, 3, 4),
        )


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError):
        rank_target_features(
            {"per_feature_effect": [0.1, 0.2]},
            top_k=0,
        )
