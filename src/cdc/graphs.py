"""Feature-to-feature attribution graphs (Circuit Tracing, Ameisen et al. 2025 style).

For one prompt we build the *local replacement model*:
  * attention patterns and all LayerNorm scales are frozen at their values on
    this prompt, which makes every non-MLP path linear;
  * each MLP is replaced by the transcoder's active features plus an error
    node (true MLP output minus reconstruction), so the local model reproduces
    the original forward pass exactly.

Nodes: input nodes (variable j at position t), active features (layer,
feature, position), error nodes (layer, position), and target logits (one per
position).  Edge A[t, s] = direct effect of source s on target t's
pre-activation, i.e. write_s . (linear map between them)^T . read_t.  For a
CLT feature the write is summed over every decoder layer below the target.

Everything is computed in float64 with explicit (T*d x T*d) matrices; that is
cheap for these small models (T=16, d=64) and makes the conservation identity
(sum of incoming edges + constant = pre-activation) exactly testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Graph:
    A: np.ndarray                 # (N, N) signed direct effects, A[target, source]
    node_type: np.ndarray         # 'input' | 'feature' | 'error' | 'logit'
    layer: np.ndarray             # layer of feature/error; -1 input; L for logit
    pos: np.ndarray
    feat: np.ndarray              # feature index (-1 otherwise); input var index for inputs
    act: np.ndarray               # activation (features), 1 otherwise
    target_preact: np.ndarray     # (N,) pre-activation of features / logits (nan for sources)
    target_const: np.ndarray      # (N,) constant (bias) part of each target
    n_layers: int
    meta: dict


def _attn_linear(blk, att, s1, d):
    """Return J (n x n) with vec(Q) = J vec(P) + c, and c (d,), for frozen attention/LN1."""
    T = att.shape[-1]
    C = torch.eye(d, dtype=torch.float64) - 1.0 / d
    g = blk.ln1.g.double()
    beta = blk.ln1.b.double()
    J = torch.eye(T * d, dtype=torch.float64)
    c = blk.attn.bO.double().clone()
    for h in range(att.shape[0]):
        WV = blk.attn.WV[h].double(); WO = blk.attn.WO[h].double()
        M = C @ torch.diag(g) @ WV @ WO                                    # row-vector map (d x d)
        B = att[h].double() / s1.double()[None, :]                         # (T,T): A_ts / sigma_s
        J = J + torch.kron(B.contiguous(), M.T.contiguous())
        c = c + (beta @ WV + blk.attn.bV[h].double()) @ WO
    return J, c


@torch.no_grad()
def build_graph(model, tc, x: np.ndarray) -> Graph:
    """x: (D+1, T) bits for a single example."""
    cfg = model.cfg
    L, d, T = cfg.n_layers, cfg.d_model, cfg.n_pos
    n = T * d
    xt = torch.as_tensor(x[None])
    logits, c = model.run_cache(xt)
    C = torch.eye(d, dtype=torch.float64) - 1.0 / d

    Js, cs = [], []
    for l, blk in enumerate(model.blocks):
        J, cc = _attn_linear(blk, c["att"][l][0], c["s1"][l][0, :, 0], d)
        Js.append(J); cs.append(cc)

    acts = [tc.encode(c["mlp_in"][l], l)[0].double() for l in range(L)]       # (T,F)
    pre = [tc.preact(c["mlp_in"][l], l)[0].double() for l in range(L)]
    rec = tc.decode_all([a.float()[None] for a in acts])
    errs = [(c["mlp_out"][l][0] - rec[l][0]).double() for l in range(L)]

    # ---------------- nodes
    types, layers, poss, feats, actv = [], [], [], [], []
    writes = []   # list of dict {point m: (pos, vec)}; point m = P_m
    V = cfg.n_inputs
    # input node = deviation of variable j's embedding from its mean over values;
    # the mean goes into the constant term (so constant offsets carry no influence)
    WE = model.W_E.double()
    mean_j = torch.stack([(WE[2 * j] + WE[2 * j + 1]) / 2 for j in range(V)])     # (V,d)
    in_w = model.input_writes(xt)[0].double() - mean_j[:, None, :]              # (V,T,d)
    for j in range(V):
        for t in range(T):
            types.append("input"); layers.append(-1); poss.append(t); feats.append(j); actv.append(1.0)
            writes.append({0: (t, in_w[j, t])})
    feat_index = {}
    for s in range(L):
        ts, fs = torch.nonzero(acts[s] > 0, as_tuple=True)
        for t, f in zip(ts.tolist(), fs.tolist()):
            a = acts[s][t, f]
            feat_index[(s, t, f)] = len(types)
            types.append("feature"); layers.append(s); poss.append(t); feats.append(f); actv.append(float(a))
            writes.append({dst + 1: (t, a * tc.dec(s, dst)[f].double()) for dst in tc.dsts(s)})
    for l in range(L):
        for t in range(T):
            types.append("error"); layers.append(l); poss.append(t); feats.append(-1); actv.append(1.0)
            writes.append({l + 1: (t, errs[l][t])})
    logit_index = []
    for t in range(T):
        logit_index.append(len(types))
        types.append("logit"); layers.append(L); poss.append(t); feats.append(-1); actv.append(1.0)
        writes.append({})
    N = len(types)
    A = np.zeros((N, N))
    tpre = np.full(N, np.nan); tconst = np.full(N, np.nan)

    # constant writes (independent of nodes): pos-embedding at P_0, decoder biases at P_{l+1},
    # attention constants at Q_l
    const_P = {0: model.pos.double() + mean_j.sum(0)[None, :]}
    for l in range(L):
        const_P[l + 1] = tc.b_dec[l].double().expand(T, d)

    def pull_and_fill(rows: torch.Tensor, row_nodes: list, read_point: tuple, row_const: torch.Tensor):
        """rows: (R, n) read vectors at read_point ('Q', l) or ('P', L)."""
        kind, l = read_point
        RP = {}
        if kind == "Q":
            RP[l] = rows @ Js[l]              # reading Q_l == reading P_l through J_l
            top = l
            const_Q = {l: rows}               # const added at Q_l read directly
        else:
            RP[l] = rows
            top = l
            const_Q = {}
        for m in range(top - 1, -1, -1):
            RP[m] = RP[m + 1] @ Js[m]
        for j in range(0, top if kind == "Q" else l):
            const_Q.setdefault(j, RP[j + 1])
        # node edges
        for si, w in enumerate(writes):
            for m, (t, vec) in w.items():
                if m > top or (kind == "Q" and m > l):
                    continue
                if m not in RP:
                    continue
                blockv = RP[m][:, t * d:(t + 1) * d]
                A[row_nodes, si] += (blockv @ vec).numpy()
        # constant part
        tot = row_const.clone()
        for m, cv in const_P.items():
            if m in RP:
                tot += (RP[m].reshape(-1, T, d) * cv[None]).sum((1, 2))
        for j, R in const_Q.items():
            tot += (R.reshape(-1, T, d) * cs[j][None, None]).sum((1, 2))
        tconst[row_nodes] = tot.numpy()

    # feature targets
    for l in range(L):
        idx = [(t, f) for (s, t, f) in feat_index if s == l]
        if not idx:
            continue
        rows = torch.zeros(len(idx), n, dtype=torch.float64)
        g2 = model.blocks[l].ln2.g.double(); b2 = model.blocks[l].ln2.b.double()
        s2 = c["s2"][l][0, :, 0].double()
        Wl = tc.W_enc[l].double()
        rc = torch.zeros(len(idx), dtype=torch.float64)
        for r, (t, f) in enumerate(idx):
            rows[r, t * d:(t + 1) * d] = C @ (g2 * Wl[f]) / s2[t]
            rc[r] = Wl[f] @ b2 + tc.b_enc[l, f].double()
            tpre[feat_index[(l, t, f)]] = float(pre[l][t, f])
        pull_and_fill(rows, [feat_index[(l, t, f)] for (t, f) in idx], ("Q", l), rc)
    # logit targets
    gf = model.lnf.g.double(); bf = model.lnf.b.double(); sf = c["sf"][0, :, 0].double()
    wU = model.w_U.double()
    rows = torch.zeros(T, n, dtype=torch.float64)
    for t in range(T):
        rows[t, t * d:(t + 1) * d] = C @ (gf * wU) / sf[t]
        tpre[logit_index[t]] = float(logits[0, t])
    pull_and_fill(rows, logit_index, ("P", L), (wU @ bf + model.b_U.double()).expand(T).clone())

    return Graph(A=A, node_type=np.array(types), layer=np.array(layers), pos=np.array(poss),
                 feat=np.array(feats), act=np.array(actv), target_preact=tpre, target_const=tconst,
                 n_layers=L, meta={"kind": tc.cfg.kind, "k": tc.cfg.k})


def conservation_error(g: Graph) -> float:
    """max |sum_in(A) + const - preact| over all targets (should be ~1e-6)."""
    m = ~np.isnan(g.target_preact)
    lhs = g.A[m].sum(1) + g.target_const[m]
    return float(np.max(np.abs(lhs - g.target_preact[m])))


def prune_graph(g: Graph, keep_frac: float) -> np.ndarray:
    """Return boolean mask of feature nodes kept so that they carry `keep_frac`
    of the total (abs-normalised) influence on the logits (Circuit-Tracing-style)."""
    from .depth_metrics import node_influence
    infl = node_influence(g)
    feat = np.where(g.node_type == "feature")[0]
    order = feat[np.argsort(-infl[feat])]
    cum = np.cumsum(infl[order]) / max(infl[feat].sum(), 1e-12)
    keep = np.zeros(len(g.node_type), bool)
    keep[g.node_type != "feature"] = True
    nkeep = int(np.searchsorted(cum, keep_frac) + 1)
    keep[order[:nkeep]] = True
    return keep
