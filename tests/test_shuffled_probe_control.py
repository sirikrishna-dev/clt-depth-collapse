import numpy as np

from src.replacements.siit_utils import probe_representation


def test_shuffled_training_labels_do_not_recover_original_labels():
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

    shuffled_labels = {
        "train": labels["train"].copy(),
        "validation": labels["validation"],
        "test": labels["test"],
    }

    for bit in range(16):
        rng.shuffle(shuffled_labels["train"][:, bit])

    result = probe_representation(features, shuffled_labels)

    assert result["mean_test_accuracy"] < 0.60
