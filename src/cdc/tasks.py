"""Boolean multi-hop tasks with known ground-truth depth.

Task with depth D uses D+1 input bit-vectors x0..xD (16 positions each) and
computes, element-wise,

    h1 = OP_1(x0, x1)
    hk = OP_k(h_{k-1}, x_k)  for k = 2..D
    y  = hD

Operator scheme 'final_xor' (default): OP_k = XOR if (D - k) is even else AND,
so the LAST hop is always XOR and operators alternate backwards:
    D=1: XOR      D=2: AND, XOR      D=3: XOR, AND, XOR (= Round-1 3-hop task)
    D=4: AND, XOR, AND, XOR          D=5: XOR, AND, XOR, AND, XOR
Why: an AND final hop is linearly separable, so the linear unembedding can
compute it without any MLP.  We verified this on the Round-1 2-hop task
(u = a XOR b; y = u AND c): zeroing the last MLP leaves accuracy at 100%, so its
nonlinear depth is 1, not 2.  Scheme 'round1' keeps the old operators for
reproduction only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

N_POS = 16
OPS_SCHEME = "final_xor"   # set to "round1" to reproduce the Round-1 operators


def op_name(k: int, D: int, scheme: str | None = None) -> str:
    """Operator of hop k (1-indexed) in a depth-D task."""
    scheme = scheme or OPS_SCHEME
    if scheme == "round1":
        return "XOR" if k == 1 else ("AND" if k % 2 == 0 else "XOR")
    return "XOR" if (D - k) % 2 == 0 else "AND"


def apply_op(k: int, a: np.ndarray, b: np.ndarray, D: int) -> np.ndarray:
    return (a & b) if op_name(k, D) == "AND" else (a ^ b)


def high_level(x: np.ndarray, D: int, overrides: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Run the high-level causal model.

    x: int array (..., D+1, N_POS) of bits.
    overrides: optional {k: array (..., N_POS)} replacing intermediate h_k.
    Returns H: (..., D, N_POS) with H[..., k-1, :] = h_k (H[..., D-1] = y).
    """
    overrides = overrides or {}
    hs = []
    h = None
    for k in range(1, D + 1):
        if k == 1:
            h = apply_op(1, x[..., 0, :], x[..., 1, :], D)
        else:
            h = apply_op(k, h, x[..., k, :], D)
        if k in overrides:
            h = overrides[k].astype(h.dtype)
        hs.append(h)
    return np.stack(hs, axis=-2)


@dataclass
class BooleanDataset:
    D: int
    x: np.ndarray  # (N, D+1, 16) int64
    H: np.ndarray  # (N, D, 16) intermediates, last = y
    split: np.ndarray  # (N,) 0 train, 1 val, 2 test

    @property
    def y(self) -> np.ndarray:
        return self.H[:, -1, :]

    def subset(self, which: str) -> "BooleanDataset":
        idx = {"train": 0, "val": 1, "test": 2}[which]
        m = self.split == idx
        return BooleanDataset(self.D, self.x[m], self.H[m], self.split[m])

    def tensors(self, device="cpu"):
        return (torch.as_tensor(self.x, device=device),
                torch.as_tensor(self.H, device=device).float())


def make_dataset(D: int, n_train=5000, n_val=1000, n_test=1000, seed=0) -> BooleanDataset:
    rng = np.random.default_rng(10_000 + 97 * D + seed)
    n = n_train + n_val + n_test
    x = rng.integers(0, 2, size=(n, D + 1, N_POS), dtype=np.int64)
    H = high_level(x, D)
    split = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    return BooleanDataset(D, x, H, split)


def counterfactual_pairs(x: np.ndarray, D: int, hop: int, rng: np.random.Generator,
                         confound_free: bool = True):
    """Make (base, source) pairs for an interchange at hop `hop` (1..D-1).

    confound_free=True: the source keeps every side input x_{hop+1..D} equal to
    the base, and only the upstream inputs x_0..x_hop are resampled.  Any
    change of the counterfactual target must then flow through h_hop
    (fixes the Round-1 v/c confound).
    Returns base_x, src_x, y_star (target when h_hop := h_hop(src)), y_base.
    """
    n = x.shape[0]
    perm = rng.permutation(n)
    src = x[perm].copy()
    if confound_free:
        src[:, hop + 1:, :] = x[:, hop + 1:, :]
    Hs = high_level(src, D)
    Hb = high_level(x, D)
    y_star = high_level(x, D, overrides={hop: Hs[:, hop - 1, :]})[:, -1, :]
    return x, src, y_star, Hb[:, -1, :]
