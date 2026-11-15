#!/usr/bin/env python
"""Logit-level bypass diagnostic.

Certification tests bypass-freeness at the DECISION level (sign of the logit).  Attribution
graphs decompose the logit MAGNITUDE.  If later layers read upstream values in a way that
changes the logit's size but never its sign, a graph will (correctly) show shallow paths
in a model whose decisions are certified deep.  For every certified model and hop k we report

  r_bypass(k) = E|Δlogit| under the bypass swap (upstream raw inputs + earlier sites from a
                source, site k kept)  /  E|Δlogit| under the site-k interchange (confound-free)

r_bypass ≈ 0: the logit depends on upstream information only through site k (deep at logit
level).  r_bypass >> 0: magnitude-level shortcuts exist in the model itself.

    python scripts/logit_bypass.py --results results/part_A --ckpt checkpoints/part_A
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cdc.model import load_model  # noqa: E402
from cdc.tasks import make_dataset, counterfactual_pairs  # noqa: E402
from cdc.siit import raw_bypass_delta  # noqa: E402
from cdc.transcoders import load_tc  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/part_A")
    ap.add_argument("--ckpt", default="checkpoints/part_A")
    a = ap.parse_args()
    cert = pd.read_csv(os.path.join(a.results, "ground_truth_certificates.csv"))
    rows = []
    for name in cert[cert.certified].model:
        m, meta = load_model(os.path.join(a.ckpt, name + ".pt"))
        D = m.cfg.D
        x = make_dataset(D).subset("test").x
        rng = np.random.default_rng(0)
        base = m(torch.as_tensor(x))
        for k in range(1, D):
            xb, xs, _, _ = counterfactual_pairs(x, D, k, rng, confound_free=True)
            _, sc = m.run_cache(torch.as_tensor(xs))
            _, cb = m.run_cache(torch.as_tensor(xb))
            dims = m.cfg.site_dims(k)
            iia = m(torch.as_tensor(xb), iv={"resid_set": {k: (dims, sc["P"][k][..., dims])}})
            delta = raw_bypass_delta(m, torch.as_tensor(xb), torch.as_tensor(xs), cb, sc, k)
            byp = m(torch.as_tensor(xb), iv={"resid_add": {k: delta}})
            d_iia = (iia - base).abs().mean().item()
            d_byp = (byp - base).abs().mean().item()
            rows.append({"model": name, "system": "original", "D": D, "seed": meta["seed"], "hop": k,
                         "mean_abs_dlogit_site_swap": d_iia, "mean_abs_dlogit_bypass_swap": d_byp,
                         "r_bypass": d_byp / max(d_iia, 1e-9),
                         "bypass_sign_flip_rate": float(((byp > 0) != (base > 0)).float().mean())})
            # same test inside each replacement model (all MLPs -> transcoder, no error terms).
            # In the replacement, "site k" is the residual coordinate written by the transcoder.
            for kind in ("plt", "clt"):
                p = os.path.join(a.ckpt, f"{name}__{kind}_k8.pt")
                if not os.path.exists(p):
                    continue
                tc = load_tc(p)
                fns = lambda: tc.as_mlp_fns()
                rb = m(torch.as_tensor(xb), iv={"mlp_fn": fns()})
                rs_cache = {}
                m.forward(torch.as_tensor(xs), iv={"mlp_fn": fns()}, cache=rs_cache)
                rb_cache = {}
                m.forward(torch.as_tensor(xb), iv={"mlp_fn": fns()}, cache=rb_cache)
                r_iia = m(torch.as_tensor(xb), iv={"mlp_fn": fns(), "resid_set": {k: (dims, rs_cache["P"][k][..., dims])}})
                dl = raw_bypass_delta(m, torch.as_tensor(xb), torch.as_tensor(xs), rb_cache, rs_cache, k)
                r_byp = m(torch.as_tensor(xb), iv={"mlp_fn": fns(), "resid_add": {k: dl}})
                di, dbp = (r_iia - rb).abs().mean().item(), (r_byp - rb).abs().mean().item()
                rows.append({"model": name, "system": kind, "D": D, "seed": meta["seed"], "hop": k,
                             "mean_abs_dlogit_site_swap": di, "mean_abs_dlogit_bypass_swap": dbp,
                             "r_bypass": dbp / max(di, 1e-9),
                             "bypass_sign_flip_rate": float(((r_byp > 0) != (rb > 0)).float().mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.results, "logit_level_bypass.csv"), index=False)
    print(df.groupby(["D", "hop", "system"])[["r_bypass", "bypass_sign_flip_rate"]].mean().round(4).unstack("system"))


if __name__ == "__main__":
    main()
