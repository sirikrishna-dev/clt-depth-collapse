"""Strict Interchange Intervention Training (SIIT) and ground-truth depth certification.

SIIT (Gupta et al., InterpBench, NeurIPS 2024) = behaviour loss
  + IIT loss       (interchange at circuit nodes -> high-level counterfactual)
  + strictness loss (resample NON-circuit nodes -> output must not change).

Circuit nodes here: hop k (1..D-1) at residual point P_k, coordinates site_dims(k).
Non-circuit nodes (a random subset is resampled jointly at each step): every attention layer output (the tasks are position-wise),
and, for blocks l <= D-2, the MLP_l output on all coordinates except its own
site (hop l+1).  Round-1 models lacked the strictness term; if you report
them, call them IIT, not SIIT.
"""
from __future__ import annotations

import json
import time



import numpy as np
import torch
import torch.nn.functional as F

from .model import BoolTransformer, ModelConfig, project_reserved_sites
from .tasks import BooleanDataset, high_level, counterfactual_pairs


def _hl_counterfactual(xb: np.ndarray, xs: np.ndarray, D: int, hop: int) -> np.ndarray:
    Hs = high_level(xs, D)
    return high_level(xb, D, overrides={hop: Hs[:, hop - 1, :]})[:, -1, :]


def strict_components(cfg: ModelConfig, include_raw: bool = True):
    """Non-circuit components resampled by the strictness loss.
    ('raw', k): after site k, the raw embeddings of the upstream inputs x_0..x_k and the
    earlier sites are resampled -- downstream layers may only use h_k (no bypass)."""
    comps = [("attn", l) for l in range(cfg.n_layers)]
    comps += [("mlp", l) for l in range(min(cfg.D - 1, cfg.n_layers))]
    if include_raw:
        comps += [("raw", k) for k in range(1, cfg.D)]
    return comps


def raw_bypass_delta(model, xb_t, xs_t, cb, sc, hop):
    """Residual delta at P_hop that swaps upstream raw inputs (x_0..x_hop) and earlier
    sites to the source's values while keeping site `hop` at the base value."""
    cfg = model.cfg
    wb = model.input_writes(xb_t)[:, :hop + 1].sum(1)
    ws = model.input_writes(xs_t)[:, :hop + 1].sum(1)
    delta = ws - wb
    for j in range(1, hop):
        dj = cfg.site_dims(j)
        delta[..., dj] = sc["P"][hop][..., dj] - cb["P"][hop][..., dj]
    delta[..., cfg.site_dims(hop)] = 0.0
    return delta


def mean_ablate_delta(model, xb_t, cb, hop, site_means):
    """Held-out bypass variant (never used in training): upstream raw inputs -> their
    mean embedding, earlier sites -> dataset mean; site `hop` kept."""
    cfg = model.cfg
    WE = model.W_E
    mean_j = torch.stack([(WE[2 * j] + WE[2 * j + 1]) / 2 for j in range(hop + 1)]).sum(0)
    delta = mean_j - model.input_writes(xb_t)[:, :hop + 1].sum(1)
    for j in range(1, hop):
        dj = cfg.site_dims(j)
        delta[..., dj] = site_means[j] - cb["P"][hop][..., dj]
    delta[..., cfg.site_dims(hop)] = 0.0
    return delta


def noncircuit_dims(cfg: ModelConfig, l: int) -> list[int]:
    own = set(cfg.site_dims(l + 1))
    return [i for i in range(cfg.d_model) if i not in own]


def train_siit(cfg: ModelConfig, data: BooleanDataset, seed: int = 0, steps: int = 4000,
               batch: int = 128, lr: float = 1e-3, w_iit: float = 1.0, w_strict: float = 1.0,
               w_site: float = 0.3, p_subset: float = 0.3,
               strict: bool = True, log_every: int = 500, device: str = "cpu", verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BoolTransformer(cfg).to(device)
    project_reserved_sites(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    tr = data.subset("train")
    X = tr.x
    D = cfg.D
    comps = strict_components(cfg)
    history = []
    t0 = time.time()
    for step in range(steps):
        bi = rng.integers(0, len(X), batch)
        xb = X[bi]
        # half confound-free sources, half fully random sources
        xs = X[rng.integers(0, len(X), batch)].copy()
        xb_t = torch.as_tensor(xb, device=device)
        yb = torch.as_tensor(high_level(xb, D)[:, -1, :], device=device).float()
        logits = model(xb_t)
        loss_beh = F.binary_cross_entropy_with_logits(logits, yb)
        loss = loss_beh
        loss_iit = torch.zeros((), device=device)
        # site readout loss: site coordinate is a logit for h_k, so the site holds the
        # binary value of h_k (forces the hop to be computed by a nonlinearity BEFORE the site)
        if D >= 2 and w_site > 0:
            cbt = {}
            model.forward(xb_t, cache=cbt)
            Hb = torch.as_tensor(high_level(xb, D), device=device).float()
            # bounded target (+-2): BCE would push the site value without limit and let it
            # dominate LayerNorm statistics
            loss_site = sum(F.mse_loss(cbt["P"][k][..., cfg.site_dims(k)[0]], 4 * Hb[:, k - 1] - 2)
                            for k in range(1, D)) / (D - 1)
            loss = loss + w_site * loss_site
        loss_strict = torch.zeros((), device=device)
        if D >= 2 or strict:
            with torch.no_grad():
                _, sc = model.run_cache(torch.as_tensor(xs, device=device))
        if D >= 2:
            hop = int(rng.integers(1, D))
            xs_h = xs.copy()
            half = batch // 2
            xs_h[:half, hop + 1:, :] = xb[:half, hop + 1:, :]  # confound-free half
            with torch.no_grad():
                _, sc_h = model.run_cache(torch.as_tensor(xs_h, device=device))
            dims = cfg.site_dims(hop)
            vals = sc_h["P"][hop][..., dims]
            y_star = torch.as_tensor(_hl_counterfactual(xb, xs_h, D, hop), device=device).float()
            lg = model(xb_t, iv={"resid_set": {hop: (dims, vals)}})
            loss_iit = F.binary_cross_entropy_with_logits(lg, y_star)
            loss = loss + w_iit * loss_iit
        if strict and comps:
            # with prob p_subset resample ALL non-circuit components jointly, otherwise one.
            # Single-component resampling alone leaves large input-dependent non-circuit
            # outputs that break the circuit when several are resampled together.
            if rng.random() < p_subset:
                pick = list(comps)
            else:
                pick = [comps[int(rng.integers(0, len(comps)))]]
            iv = {"attn_out": {}, "mlp_out_set": {}, "resid_add": {}}
            cb = None
            for kind, l in pick:
                if kind == "attn":
                    iv["attn_out"][l] = sc["attn_out"][l]
                elif kind == "mlp":
                    dims = noncircuit_dims(cfg, l)
                    iv["mlp_out_set"][l] = (dims, sc["mlp_out"][l][..., dims])
                else:
                    if cb is None:
                        cb = {}
                        with torch.no_grad():
                            model.forward(xb_t, cache=cb)
                    iv["resid_add"][l] = raw_bypass_delta(model, xb_t, torch.as_tensor(xs, device=device), cb, sc, l).detach()
            lg = model(xb_t, iv=iv)
            loss_strict = F.binary_cross_entropy_with_logits(lg, yb)
            loss = loss + w_strict * loss_strict
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        project_reserved_sites(model)
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "loss": float(loss.detach()), "beh": float(loss_beh.detach()),
                   "iit": float(loss_iit.detach()), "strict": float(loss_strict.detach()), "sec": time.time() - t0}
            history.append(rec)
            if verbose:
                print(json.dumps(rec))
    model.eval()
    return model, history


# ----------------------------------------------------------------- certification
@torch.no_grad()
def _pred(model, x, iv=None):
    return (model(torch.as_tensor(x), iv=iv) > 0).long().numpy()


@torch.no_grad()
def certify(model: BoolTransformer, data: BooleanDataset, seed: int = 123, n_rep: int = 3) -> dict:
    """Ground-truth-depth certificate.  Every score reports its denominator."""
    cfg = model.cfg
    D = cfg.D
    te = data.subset("test")
    x, y = te.x, te.y
    rng = np.random.default_rng(seed)
    out = {"D": D, "n_layers": cfg.n_layers, "n_test_examples": int(len(x))}
    p = _pred(model, x)
    out["clean_bit_acc"] = float((p == y).mean())
    out["clean_exact_acc"] = float((p == y).all(1).mean())

    for hop in range(1, D):
        for cf in (False, True):
            agree_all, agree_flip, n_flip, n_all = 0, 0, 0, 0
            for _ in range(n_rep):
                xb, xs, y_star, y_b = counterfactual_pairs(x, D, hop, rng, confound_free=cf)
                _, sc = model.run_cache(torch.as_tensor(xs))
                dims = cfg.site_dims(hop)
                pi = _pred(model, xb, {"resid_set": {hop: (dims, sc["P"][hop][..., dims])}})
                flip = y_star != y_b
                agree_all += (pi == y_star).sum(); n_all += y_star.size
                agree_flip += (pi == y_star)[flip].sum(); n_flip += flip.sum()
            tag = "cf" if cf else "rand"
            out[f"iia_h{hop}_{tag}_all"] = float(agree_all / n_all)
            out[f"iia_h{hop}_{tag}_flip"] = float(agree_flip / max(n_flip, 1))
            out[f"iia_h{hop}_{tag}_n_flip"] = int(n_flip)

    # strictness: resample each non-circuit component; output must equal base
    strict_scores = {}
    for kind, l in strict_components(cfg, include_raw=False):
        xs = x[rng.permutation(len(x))]
        _, sc = model.run_cache(torch.as_tensor(xs))
        if kind == "attn":
            iv = {"attn_out": {l: sc["attn_out"][l]}}
        else:
            dims = noncircuit_dims(cfg, l)
            iv = {"mlp_out_set": {l: (dims, sc["mlp_out"][l][..., dims])}}
        strict_scores[f"{kind}{l}"] = float((_pred(model, x, iv) == y).mean())
    out["strict_scores"] = strict_scores
    out["strict_min"] = min(strict_scores.values()) if strict_scores else 1.0

    # bypass test: after site k, give downstream layers the SOURCE's upstream raw
    # inputs and earlier sites, but keep the BASE value of site k.
    bypass = {}
    for hop in range(1, D):
        xs = x[rng.permutation(len(x))].copy()
        xs[:, hop + 1:, :] = x[:, hop + 1:, :]
        _, sc = model.run_cache(torch.as_tensor(xs))
        _, cb = model.run_cache(torch.as_tensor(x))
        delta = raw_bypass_delta(model, torch.as_tensor(x), torch.as_tensor(xs), cb, sc, hop)
        pi = _pred(model, x, {"resid_add": {hop: delta}})
        site_means = {j: cb["P"][hop][..., cfg.site_dims(j)].mean((0, 1)) for j in range(1, hop)}
        dm = mean_ablate_delta(model, torch.as_tensor(x), cb, hop, site_means)
        pm = _pred(model, x, {"resid_add": {hop: dm}})
        x_mix = x.copy(); x_mix[:, :hop + 1] = xs[:, :hop + 1]
        y_alt = high_level(x_mix, D)[:, -1, :]
        informative = y_alt != y
        bypass[f"h{hop}"] = {"score": float((pi == y)[informative].mean()) if informative.any() else 1.0,
                             "n_informative_bits": int(informative.sum()),
                             "heldout_meanablate_score": float((pm == y).mean())}
    out["bypass"] = bypass
    out["bypass_min"] = min([v["score"] for v in bypass.values()], default=1.0)
    out["bypass_heldout_min"] = min([v["heldout_meanablate_score"] for v in bypass.values()], default=1.0)

    # site binarity: the site is (close to) a function of h_k only
    _, cb = model.run_cache(torch.as_tensor(x))
    Hx = high_level(x, D)
    sep = {}
    for hop in range(1, D):
        sv = cb["P"][hop][..., cfg.site_dims(hop)[0]].numpy()
        h = Hx[:, hop - 1]
        m1, m0 = sv[h == 1].mean(), sv[h == 0].mean()
        sd = np.sqrt((sv[h == 1].var() + sv[h == 0].var()) / 2)
        sep[f"h{hop}"] = float(abs(m1 - m0) / (sd + 1e-9))
    out["site_separation"] = sep
    out["site_separation_min"] = min(sep.values(), default=np.inf)
    # MLP necessity, measured in the CIRCUIT-ONLY model (every non-circuit node
    # resampled jointly from one source example).  The circuit-only model must still solve the task, and
    # mean-ablating any hop-hosting MLP must break it.  The same test on the full model
    # is reported as 'backup' computation (non-circuit nodes that only matter once the
    # circuit is damaged; strictness does not remove these).
    def meanab(t):
        return t.mean((0, 1), keepdim=True).expand_as(t)
    full = list(range(cfg.d_model))
    # circuit-only model: ALL non-circuit components resampled jointly from one source
    xs = x[rng.permutation(len(x))].copy()
    _, sc = model.run_cache(torch.as_tensor(xs))
    circ_iv = {"attn_out": {l: sc["attn_out"][l] for l in range(cfg.n_layers)}, "mlp_out_set": {}, "resid_add": {}}
    for l in range(min(D - 1, cfg.n_layers)):
        nd = noncircuit_dims(cfg, l)
        circ_iv["mlp_out_set"][l] = (nd, sc["mlp_out"][l][..., nd])
    out["circuit_only_acc"] = float((_pred(model, x, circ_iv) == y).mean())
    nec, backup = {}, {}
    for l in range(cfg.n_layers):
        iv = {"attn_out": circ_iv["attn_out"], "mlp_out_set": dict(circ_iv["mlp_out_set"])}
        iv["mlp_out_set"][l] = (full, meanab(cb["mlp_out"][l]))
        nec[f"mlp{l}"] = float((_pred(model, x, iv) == y).mean())
        backup[f"mlp{l}"] = float((_pred(model, x, {"mlp_out_set": {l: (full, meanab(cb["mlp_out"][l]))}}) == y).mean())
    out["mlp_meanablate_acc"] = nec
    out["mlp_meanablate_acc_fullmodel"] = backup
    out["n_necessary_mlps"] = int(sum(v <= 0.9 for v in nec.values()))

    iia_cf = [out[f"iia_h{h}_cf_flip"] for h in range(1, D)]
    out["iia_cf_min"] = min(iia_cf, default=1.0)
    out["certified"] = bool(out["clean_bit_acc"] >= 0.99 and out["iia_cf_min"] >= 0.98
                            and out["strict_min"] >= 0.98 and out["bypass_min"] >= 0.98
                            and out["bypass_heldout_min"] >= 0.98
                            and out["circuit_only_acc"] >= 0.99
                            and out["n_necessary_mlps"] >= D
                            and out["site_separation_min"] >= 3.0)
    return out
