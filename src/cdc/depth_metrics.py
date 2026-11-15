"""Depth-collapse metrics on attribution graphs (advisor point (a)).

Let Â be the absolute adjacency, normalised so each target's incoming weights
sum to 1 (Circuit Tracing convention).  The *hop count* of a path is the number
of feature nodes on it; in a faithful graph of a D-hop task the path from the
root inputs (x0, x1, which feed hop 1) to y passes D feature nodes (one per
hop, the last hop being the final-layer features that compute y).  A CLT
feature at layer 0 that decodes directly into the last MLP gives a 1-hop path.

  I_k     influence on the logits carried by paths with exactly k feature hops
  D_eff   sum_k k I_k / sum_k I_k                      (effective graph depth)
  D_feat  same, restricted to k >= 1 (feature-mediated paths)  <- headline metric
  DCR     D_feat / D (ground-truth graph has D_feat = D for root inputs)            (depth-collapse ratio; 1 = no collapse)
  D80     smallest k with cumulative influence >= 80%
  skip    influence-weighted fraction of feature->(feature|logit) edges that
          span more than one layer
  IMF_k   fraction of root-input influence that passes through features
          aligned with ground-truth intermediate h_k
Metrics are computed for 'root' sources (x0, x1) and for 'all' input sources.
"""
from __future__ import annotations

import numpy as np

from .graphs import Graph


def normalized_adjacency(A: np.ndarray) -> np.ndarray:
    Aa = np.abs(A)
    s = Aa.sum(1, keepdims=True)
    return np.divide(Aa, s, out=np.zeros_like(Aa), where=s > 0)


def node_influence(g: Graph, Ah: np.ndarray | None = None) -> np.ndarray:
    """Total influence of every node on the (mean over positions) logits."""
    Ah = normalized_adjacency(g.A) if Ah is None else Ah
    Y = np.where(g.node_type == "logit")[0]
    z = np.zeros(len(Ah)); z[Y] = 1.0 / len(Y)
    # infl = z Â (I - Â)^-1  (Â is nilpotent: DAG)
    rhs = z @ Ah
    return np.linalg.solve((np.eye(len(Ah)) - Ah).T, rhs)


def hop_profile(g: Graph, sources: np.ndarray, Ah: np.ndarray | None = None, kmax: int | None = None):
    """I_k for k = 0..kmax from the given source nodes to the logits."""
    Ah = normalized_adjacency(g.A) if Ah is None else Ah
    Y = np.where(g.node_type == "logit")[0]
    Fi = np.where(g.node_type == "feature")[0]
    kmax = (g.n_layers + 1) if kmax is None else kmax
    I = np.zeros(kmax + 1)
    yrow = Ah[Y].mean(0)                      # (N,) direct influence row
    I[0] = yrow[sources].sum()
    u = yrow[Fi]                              # influence reaching each feature through 1 hop (itself)
    AFF = Ah[np.ix_(Fi, Fi)]
    AFS = Ah[np.ix_(Fi, sources)]
    for k in range(1, kmax + 1):
        I[k] = (u @ AFS).sum()
        u = u @ AFF
        if u.sum() < 1e-12:
            break
    return I


def signed_hop_profile(g: Graph, sources: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Cancellation-aware decomposition.  In the local replacement model the logit is
    exactly linear in the source node values, so the signed contribution of `sources`
    to each logit splits by the number k of feature nodes on the path:
        C_k[y] = M_yF (M_FF)^{k-1} A_{F,S} 1     (C_0[y] = A_{y,S} 1)
    where M is the edge matrix with feature-source columns divided by the source
    activation (a derivative).  Returns mean over logit positions of |C_k[y]|."""
    Y = np.where(g.node_type == "logit")[0]
    Fi = np.where(g.node_type == "feature")[0]
    kmax = (g.n_layers + 1) if kmax is None else kmax
    M = g.A.copy()
    M[:, Fi] = M[:, Fi] / g.act[Fi][None, :]
    out = np.zeros((kmax + 1, len(Y)))
    out[0] = g.A[np.ix_(Y, sources)].sum(1)
    v = g.A[np.ix_(Fi, sources)].sum(1)          # signed contribution reaching each feature (1 hop)
    MFF = M[np.ix_(Fi, Fi)]
    MYF = M[np.ix_(Y, Fi)]
    for k in range(1, kmax + 1):
        out[k] = MYF @ v
        v = MFF @ v
        if np.abs(v).sum() < 1e-12:
            break
    return np.abs(out).mean(1)


def depth_stats(I: np.ndarray) -> dict:
    tot = I.sum()
    if tot <= 0:
        return {"D_eff": np.nan, "D_feat": np.nan, "D80": np.nan, "direct_frac": np.nan, "input_influence": 0.0}
    p = I / tot
    k = np.arange(len(I))
    cum = np.cumsum(p)
    pf = I[1:] / I[1:].sum() if I[1:].sum() > 0 else np.full(len(I) - 1, np.nan)
    return {"D_eff": float((k * p).sum()),
            # depth among feature-mediated paths (excludes direct residual input->logit, I0)
            "D_feat": float((k[1:] * pf).sum()),
            "D80": int(np.searchsorted(cum, 0.8 - 1e-12)),
            "direct_frac": float(p[0]),
            "input_influence": float(tot), **{f"I{j}": float(p[j]) for j in range(len(I))}}


def skip_mass(g: Graph, Ah: np.ndarray | None = None, infl: np.ndarray | None = None) -> float:
    Ah = normalized_adjacency(g.A) if Ah is None else Ah
    infl = node_influence(g, Ah) if infl is None else infl
    Y = g.node_type == "logit"
    w_target = infl.copy(); w_target[Y] = 1.0 / Y.sum()
    Fi = np.where(g.node_type == "feature")[0]
    tgt = np.where((g.node_type == "feature") | Y)[0]
    W = Ah[np.ix_(tgt, Fi)] * w_target[tgt][:, None]
    span = g.layer[tgt][:, None] - g.layer[Fi][None, :]
    tot = W.sum()
    return float(W[span > 1].sum() / tot) if tot > 0 else np.nan


def mediated_fraction(g: Graph, sources: np.ndarray, through: np.ndarray, Ah=None) -> float:
    """Fraction of source->logit influence that passes through node set `through`."""
    Ah = normalized_adjacency(g.A) if Ah is None else Ah
    infl = node_influence(g, Ah)
    tot = infl[sources].sum()
    if tot <= 0 or len(through) == 0:
        return 0.0
    A2 = Ah.copy(); A2[through, :] = 0; A2[:, through] = 0
    rem = node_influence(g, A2)[sources].sum()
    return float(1 - rem / tot)


def root_sources(g: Graph) -> np.ndarray:
    return np.where((g.node_type == "input") & (g.feat <= 1))[0]


def all_sources(g: Graph) -> np.ndarray:
    return np.where(g.node_type == "input")[0]


def graph_metrics(g: Graph, aligned: dict[int, np.ndarray] | None = None, roots: np.ndarray | None = None) -> dict:
    """roots: input-node positions treated as root sources (default: variables x0, x1)."""
    Ah = normalized_adjacency(g.A)
    infl = node_influence(g, Ah)
    out = {}
    root = root_sources(g) if roots is None else np.where((g.node_type == "input") & np.isin(g.pos, roots))[0]
    for name, src in (("root", root), ("all", all_sources(g))):
        st = depth_stats(hop_profile(g, src, Ah))
        out.update({f"{name}_{k}": v for k, v in st.items()})
        ss = depth_stats(signed_hop_profile(g, src))
        out.update({f"{name}_signed_{k}": v for k, v in ss.items() if k != "input_influence"})
    err = np.where(g.node_type == "error")[0]
    out["error_influence"] = float(infl[err].sum())
    out["skip_mass"] = skip_mass(g, Ah, infl)
    out["n_features_active"] = int((g.node_type == "feature").sum())
    if aligned:
        for hop, nodes in aligned.items():
            out[f"IMF_h{hop}"] = mediated_fraction(g, root, nodes, Ah)
    return out


# --------------------------------------------------------- ground-truth graph
def ground_truth_graph(D: int) -> Graph:
    """Graph of the SIIT high-level causal model at one position.
    Hop k is a 'feature' node at layer k-1 (hop D computes y and feeds the logit)."""
    V = D + 1
    types = ["input"] * V + ["feature"] * D + ["logit"]
    layer = [-1] * V + list(range(D)) + [D]
    feat = list(range(V)) + list(range(D)) + [-1]
    N = len(types)
    A = np.zeros((N, N))
    h = lambda k: V + k - 1  # node of hop k
    A[h(1), 0] = A[h(1), 1] = 1.0
    for k in range(2, D + 1):
        A[h(k), h(k - 1)] = 1.0
        A[h(k), k] = 1.0
    A[N - 1, h(D)] = 1.0
    z = np.zeros(N)
    return Graph(A=A, node_type=np.array(types), layer=np.array(layer), pos=np.zeros(N, int),
                 feat=np.array(feat), act=np.ones(N), target_preact=np.full(N, np.nan),
                 target_const=np.full(N, np.nan), n_layers=D, meta={"kind": "ground_truth"})
