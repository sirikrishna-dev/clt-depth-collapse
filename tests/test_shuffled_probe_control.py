import numpy as np
import pytest

from src.replacements.siit_utils import probe_representation


def make_random_data():
    rng = np.random.default_rng(42)

    features = {
        "train": rng.normal(size=(500, 16, 8)),
        "validation": rng.normal(size=(200, 16, 8)),
        "test": rng.normal(size=(200, 16, 8)),
    }

    labels = {
        split: rng.integers(0, 2, size=(len(x), 16))
        for split, x in features.items()
    }

    return features, labels


def test_shuffled_control_is_present():
    features, labels = make_random_data()

    result = probe_representation(features, labels)

    control = result["shuffled_label_control"]

    assert control["n_shuffles"] == 20
    assert 0 <= control["mean_test_accuracy"] <= 1
    assert 0 <= control["empirical_p_value"] <= 1


def test_shuffled_control_is_reproducible():
    features, labels = make_random_data()

    first = probe_representation(features, labels, shuffle_seed=42)
    second = probe_representation(features, labels, shuffle_seed=42)

    assert (
        first["shuffled_label_control"]["per_shuffle_mean_test_accuracy"]
        == second["shuffled_label_control"]["per_shuffle_mean_test_accuracy"]
    )


def test_invalid_shuffle_count_is_rejected():
    features, labels = make_random_data()

    with pytest.raises(ValueError):
        probe_representation(features, labels, n_shuffles=0)


def test_random_features_do_not_predict_independent_labels():
    features, labels = make_random_data()

    result = probe_representation(features, labels)

    assert result["mean_test_accuracy"] < 0.60