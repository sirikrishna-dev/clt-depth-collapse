"""Small-sample statistics used throughout (seeds are the unit of replication)."""
from __future__ import annotations

import itertools

import numpy as np


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def paired_signflip_p(a, b):
    """Exact two-sided sign-flip permutation test on paired differences (e.g. CLT vs PLT
    on the same base model / seed).  With 5 pairs the smallest p is 2/32 = 0.0625."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return np.nan
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(0).choice([-1, 1], (20000, n))
    null = np.abs((signs * d).mean(1))
    return float((null >= obs - 1e-12).mean())


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def spearman_trend(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)
