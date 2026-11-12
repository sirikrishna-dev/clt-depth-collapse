import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stats_utils import bootstrap_ci, decide, paired_t_ci, sign_test, wilson_ci  # noqa: E402


def test_wilson_known_values():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_ci(100, 100)  # perfect score still has a lower bound below 1
    assert hi == 1.0 and lo == pytest.approx(0.9630, abs=1e-3)


def test_paired_t_known_value():
    lo, hi = paired_t_ci([1, 2, 3, 4, 5])  # mean 3, sd 1.581, t(4)=2.776
    assert lo == pytest.approx(3 - 2.776 * 1.5811 / 5 ** 0.5, abs=1e-3)
    assert hi == pytest.approx(3 + 2.776 * 1.5811 / 5 ** 0.5, abs=1e-3)


def test_bootstrap_deterministic_and_brackets_mean():
    a = bootstrap_ci([-0.3, -0.2, -0.25, -0.1, -0.15])
    b = bootstrap_ci([-0.3, -0.2, -0.25, -0.1, -0.15])
    assert a == b
    assert a[0] < -0.2 < a[1] < 0


def test_sign_test_five_seeds_cannot_reach_005():
    assert sign_test([-1, -1, -1, -1, -1]) == pytest.approx(0.0625)


def test_decide_rule():
    neg = [-0.30, -0.20, -0.25, -0.22, -0.28]
    verdict, _ = decide({"2hop": neg, "3hop": neg}, {"2hop": 0.7, "3hop": 0.6})
    assert verdict == "collapse supported"
    verdict, _ = decide({"2hop": neg, "3hop": [0.1, -0.1, 0.05, -0.05, 0.0]}, {"2hop": 0.7, "3hop": 0.9})
    assert verdict == "no collapse detected"
    verdict, _ = decide({"2hop": neg[:3], "3hop": neg}, {"2hop": 0.7, "3hop": 0.6})
    assert verdict == "inconclusive"
    verdict, _ = decide({"2hop": neg, "3hop": neg}, {"2hop": 1.1, "3hop": 0.6})
    assert verdict == "no collapse detected"
