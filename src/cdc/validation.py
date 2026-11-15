"""Validation: from 'reconstructed' to 'validated' (advisor point 3 / TA concern).

1. feature_alignment      which ground-truth variable each feature tracks
2. replacement_run        full replacement model with controllable skip/same-layer writes
3. skip_dependence        interventional test of depth collapse (graph-free)
4. ablation_check         graph-predicted vs actual effect of ablating features
5. probes_and_causal      the decodable x causally-used 2x2 at every residual point,
                          with 1000-permutation probe p-values
6. naive_patch_selectivity  Round-1 style full-vector patching is not variable-selective
"""
from __future__ import annotations

import numpy as np
import torch

from .tasks import high_level, counterfactual_pairs
from .graphs import Graph


# ----------------------------------------------------------- 1. alignment
@torch.no_grad()
def feature_alignment(model, tc, x: np.ndarray, thresh: float = 0.3):
    """Correlate each feature (per layer) with every variable (inputs x_j, hops h_k).
    Returns list of dict rows and {hop: set((layer, feature))} of aligned features."""
    D = model.cfg.D
    H = high_level(x, D)                                   # (N,D,T)
    _, c = model.run_cache(torch.as_tensor(x))
    names = [f"x{j}" for j in range(D + 1)] + [f"h{k}" for k in range(1, D + 1)]
    Vmat = np.concatenate([x, H], axis=1).transpose(0, 2, 1).reshape(-1, len(names)).astype(float)
    Vz = (Vmat - Vmat.mean(0)) / (Vmat.std(0) + 1e-9)
    rows, aligned = [], {k: set() for k in range(1, D + 1)}
    for l in range(model.cfg.n_layers):
        a = tc.encode(c["mlp_in"][l], l).reshape(-1, tc.cfg.n_features).numpy()
        az = (a - a.mean(0)) / (a.std(0) + 1e-9)
        corr = az.T @ Vz / len(az)                           # (F, V)
        for f in range(a.shape[1]):
            if a[:, f].std() == 0:
                rows.append({"layer": l, "feature": f, "label": "dead", "max_abs_corr": 0.0, "density": 0.0})
                continue
            j = int(np.argmax(np.abs(corr[f])))
            lab = names[j]
            r = float(abs(corr[f, j]))
            rows.append({"layer": l, "feature": f, "label": lab, "max_abs_corr": r,
                         "density": float((a[:, f] > 0).mean())})
            if lab.startswith("h") and r >= thresh:
                aligned[int(lab[1:])].add((l, f))
    return rows, aligned


def aligned_nodes(g: Graph, aligned: dict) -> dict:
    out = {}
    for hop, s in aligned.items():
        idx = [i for i in np.where(g.node_type == "feature")[0] if (int(g.layer[i]), int(g.feat[i])) in s]
        out[hop] = np.array(idx, dtype=int)
    return out


# ----------------------------------------------------------- 2. replacement run
def replacement_run(model, tc, x, errors=None, edit=None, same_acts=None, skip_acts=None, record=None):
    """Run the original model with every MLP replaced by the transcoder.
    edit(l, acts) -> acts       : modify activations (ablation/steering)
    same_acts / skip_acts {l: a}: use these activations for same-layer / cross-layer writes
    errors {l: tensor}          : add error terms (local replacement model)"""
    pending = {}

    def fn(m_in, l):
        a = tc.encode(m_in, l)
        if edit is not None:
            a = edit(l, a)
        if record is not None:
            record[l] = a
        for t in tc.dsts(l):
            src = a
            if t == l and same_acts is not None:
                src = same_acts[l]
            if t != l and skip_acts is not None:
                src = skip_acts[l]
            pending[t] = pending.get(t, 0) + src @ tc.dec(l, t)
        out = tc.b_dec[l] + pending.pop(l)
        if errors is not None:
            out = out + errors[l]
        return out

    return model(torch.as_tensor(x), iv={"mlp_fn": {l: fn for l in range(model.cfg.n_layers)}})


# ----------------------------------------------------------- 3. skip dependence
@torch.no_grad()
def skip_dependence(model, tc, x: np.ndarray, seed: int = 0) -> dict:
    """Change the root inputs (x0, x1) only.  Compare the replacement model's output when
    (a) everything updates, (b) cross-layer (skip) writes are frozen at base values
    ('stepwise' route only), (c) same-layer writes are frozen ('skip' route only).
    Scored on informative bits (where the true y changes)."""
    D = model.cfg.D
    rng = np.random.default_rng(seed)
    xs = x.copy()
    xs[:, :2] = x[rng.permutation(len(x))][:, :2]
    yb = high_level(x, D)[:, -1]; ys = high_level(xs, D)[:, -1]
    inf = ys != yb
    base_rec = {}
    replacement_run(model, tc, x, record=base_rec)
    full = (replacement_run(model, tc, xs) > 0).long().numpy()
    step = (replacement_run(model, tc, xs, skip_acts=base_rec) > 0).long().numpy()
    skip = (replacement_run(model, tc, xs, same_acts=base_rec) > 0).long().numpy()
    f = lambda p: float((p == ys)[inf].mean()) if inf.any() else np.nan
    res = {"n_informative": int(inf.sum()), "full_route_acc": f(full),
           "stepwise_only_acc": f(step), "skip_only_acc": f(skip)}
    # interventional skip dependence: share of the effect lost when skip writes are frozen
    res["ISD"] = float(1 - res["stepwise_only_acc"] / max(res["full_route_acc"], 1e-9))
    return res


# ----------------------------------------------------------- 4. ablation check
@torch.no_grad()
def ablation_check(model, tc, g: Graph, x_single: np.ndarray, top: int = 10) -> list[dict]:
    """Graph-predicted (linear) vs actual change in mean logit when zero-ablating a feature."""
    N = len(g.A)
    W = g.A.copy()
    fi = np.where(g.node_type == "feature")[0]
    W[:, fi] = W[:, fi] / g.act[fi][None, :]               # d preact_t / d a_s
    Y = np.where(g.node_type == "logit")[0]
    z = np.zeros(N); z[Y] = 1 / len(Y)
    M = np.zeros((N, N)); M[np.ix_(np.r_[fi, Y], np.arange(N))] = W[np.r_[fi, Y]]
    # gradient of mean logit wrt each node value: g = z M (I - M_ff)^-1 restricted to active features
    Mf = np.zeros((N, N)); Mf[np.ix_(fi, np.arange(N))] = W[fi]
    gvec = np.linalg.solve((np.eye(N) - Mf).T, (z @ M).T)
    pred = -g.act * gvec
    order = fi[np.argsort(-np.abs(pred[fi]))][:top]
    xt = x_single[None]
    _, c = model.run_cache(torch.as_tensor(xt))
    rec = tc.decode_all([tc.encode(c["mlp_in"][l], l) for l in range(model.cfg.n_layers)])
    errs = {l: c["mlp_out"][l] - rec[l] for l in range(model.cfg.n_layers)}
    base = replacement_run(model, tc, xt, errors=errs).mean().item()
    out = []
    for i in order:
        l, t, f = int(g.layer[i]), int(g.pos[i]), int(g.feat[i])

        def edit(ll, a, l=l, t=t, f=f):
            if ll == l:
                a = a.clone(); a[0, t, f] = 0.0
            return a
        new = replacement_run(model, tc, xt, errors=errs, edit=edit).mean().item()
        out.append({"layer": l, "pos": t, "feature": f, "pred": float(pred[i]), "actual": new - base})
    return out


# ----------------------------------------------------------- 5. probes + causal
def _ridge_hat(X, lam=1e-2):
    Xb = np.c_[X, np.ones(len(X))]
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T)


@torch.no_grad()
def probes_and_causal(model, data, n_perm=1000, seed=0, max_tokens=16000):
    """For each residual point P_p and each intermediate h_k (k=1..D-1):
    decodable = ridge-probe test accuracy (+ permutation p-value);
    causal    = IIA on flipped bits when the probe direction's projection is set to
                the (confound-free) source's projection."""
    rng = np.random.default_rng(seed)
    D, L = model.cfg.D, model.cfg.n_layers
    tr, te = data.subset("train"), data.subset("test")
    _, ctr = model.run_cache(torch.as_tensor(tr.x))
    _, cte = model.run_cache(torch.as_tensor(te.x))
    Htr, Hte = high_level(tr.x, D), high_level(te.x, D)
    rows = []
    for p in range(L + 1):
        Xtr = ctr["P"][p].reshape(-1, model.cfg.d_model).numpy().astype(np.float64)
        Xte = cte["P"][p].reshape(-1, model.cfg.d_model).numpy().astype(np.float64)
        sel = rng.choice(len(Xtr), min(max_tokens, len(Xtr)), replace=False)
        Hm = _ridge_hat(Xtr[sel])
        Xteb = np.c_[Xte, np.ones(len(Xte))]
        for k in range(1, D):
            ytr = Htr[:, k - 1].reshape(-1)[sel] * 2.0 - 1
            yte = Hte[:, k - 1].reshape(-1)
            w = Hm @ ytr
            acc = float(((Xteb @ w > 0) == (yte > 0.5)).mean())
            Yp = np.stack([rng.permutation(ytr) for _ in range(n_perm)], 1)
            accp = (((Xteb @ (Hm @ Yp)) > 0) == (yte[:, None] > 0.5)).mean(0)
            pval = float((1 + (accp >= acc).sum()) / (n_perm + 1))
            # causal test along the probe direction
            u = torch.as_tensor(w[:-1] / np.linalg.norm(w[:-1]), dtype=torch.float32)
            xb, xs, y_star, y_b = counterfactual_pairs(te.x, D, k, rng, confound_free=True)
            _, cs = model.run_cache(torch.as_tensor(xs))
            vals = cs["P"][p] @ u
            pi = (model(torch.as_tensor(xb), iv={"proj_set": {p: (u, vals)}}) > 0).long().numpy()
            flip = y_star != y_b
            iia = float((pi == y_star)[flip].mean())
            rows.append({"point": f"P{p}", "p": p, "hop": k, "probe_acc": acc, "perm_mean": float(accp.mean()),
                         "perm_p": pval, "n_perm": n_perm, "causal_iia_flip": iia, "n_flip": int(flip.sum()),
                         "decodable": bool(acc >= 0.95 and pval < 0.01), "causal": bool(iia >= 0.9),
                         "is_designated_site": p == k})
    return rows


# ----------------------------------------------------------- 6. naive baseline
@torch.no_grad()
def naive_patch_selectivity(model, data, seed=0):
    """Round-1 style whole-vector activation patching at P_k.  For each hypothesis
    variable v in {h_k, x_0..x_D}, measure agreement with the counterfactual 'v takes
    the source value'.  A selective method would favour h_k only."""
    rng = np.random.default_rng(seed)
    D = model.cfg.D
    te = data.subset("test")
    x = te.x
    rows = []
    for k in range(1, D):
        xs = x[rng.permutation(len(x))]
        _, cs = model.run_cache(torch.as_tensor(xs))
        pi = (model(torch.as_tensor(x), iv={"resid_set": {k: (list(range(model.cfg.d_model)), cs["P"][k])}}) > 0).long().numpy()
        yb = high_level(x, D)[:, -1]
        hyps = {f"h{k}": high_level(x, D, {k: high_level(xs, D)[:, k - 1]})[:, -1]}
        for j in range(D + 1):
            xm = x.copy(); xm[:, j] = xs[:, j]
            hyps[f"x{j}"] = high_level(xm, D)[:, -1]
        for name, yh in hyps.items():
            flip = yh != yb
            rows.append({"hop": k, "hypothesis": name,
                         "agreement_flip": float((pi == yh)[flip].mean()) if flip.any() else np.nan,
                         "n_flip": int(flip.sum())})
    return rows
