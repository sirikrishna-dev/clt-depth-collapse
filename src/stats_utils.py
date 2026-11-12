"""Statistics used by the pre-registered analysis.

- wilson_ci:      95% Wilson score interval for a proportion (k successes of n).
- paired_t_ci:    95% paired t-interval for the mean of per-seed differences.
- bootstrap_ci:   95% percentile bootstrap CI for the mean (10,000 resamples, fixed seed).
- sign_test:      exact two-sided sign test (zeros dropped).
- decide:         the frozen decision rule from PREREGISTRATION.md Section 8.

All functions are deterministic. Tested in tests/test_stats_utils.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_t_ci(diffs, level: float = 0.95):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return (m, m)
    t = stats.t.ppf(0.5 + level / 2, df=n - 1)
    return (m - t * se, m + t * se)


def bootstrap_ci(diffs, n_resamples: int = 10_000, level: float = 0.95, seed: int = 0):
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))


def sign_test(diffs):
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = int((d > 0).sum())
    return float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def decide(task_diffs: dict, clt_depth_ratio: dict, min_pairs: int = 4):
    """Frozen decision rule (PREREGISTRATION.md Section 8).

    task_diffs:      {"2hop": [CLT-PLT depth-ratio diff per valid seed], "3hop": [...]}
    clt_depth_ratio: {"2hop": mean CLT depth ratio, "3hop": ...}
    Returns one of "collapse supported", "no collapse detected", "inconclusive",
    plus a per-task detail dict.
    """
    detail = {}
    for task, diffs in task_diffs.items():
        d = list(diffs)
        detail[task] = dict(
            n_pairs=len(d),
            mean_diff=float(np.mean(d)) if d else float("nan"),
            t_ci=paired_t_ci(d),
            boot_ci=bootstrap_ci(d),
            sign_p=sign_test(d),
            clt_depth_ratio=clt_depth_ratio.get(task, float("nan")),
        )
    if any(v["n_pairs"] < min_pairs for v in detail.values()):
        return "inconclusive", detail
    ok = all(
        v["mean_diff"] < 0 and v["t_ci"][1] < 0 and v["boot_ci"][1] < 0 and v["clt_depth_ratio"] < 1
        for v in detail.values()
    )
    return ("collapse supported" if ok else "no collapse detected"), detail
