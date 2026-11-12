"""Person 1: ground-truth computational depth (D_true) for every original model.

Implements PREREGISTRATION.md Section 4. Run from the repo root:

    python scripts/ground_truth_depth.py
    python scripts/ground_truth_depth.py --quick      # 200 test examples, smoke test

Tests (all on the held-out test split, 1,000 examples x 16 bit tokens):
  1. sanity       clean bit / exact-vector / balanced accuracy and class balance of y
  2. ablation     each MLP output and each attention output, three ways:
                  resample (3 fixed seeds), mean, zero; plus skipping a whole block
  3. decodability logistic-regression probes for u, v, y from the residual stream
                  after every block (trained on train split, scored on test split)
  4. interchange  SIIT models only, at the designated coordinate:
                  u swap; v swap with the corrected protocol (score only bits where
                  donor AND recipient have c = 1); the old "decoupled" v protocol
                  for comparison; an unrelated-coordinate control
  5. chain        3-hop SIIT: u -> v -> y propagation, counterfactual output,
                  and reversal when v is restored
  6. bypass       SIIT: ablate an MLP, restore only the designated site, and
                  measure how much of the damage is undone
  7. y shortcut   all models: swap only the probe direction for y after each
                  non-final block; does the output follow the donor's y?
Every score is reported with its denominator and a 95% Wilson interval.

D_true rule: see decide_dtrue() and PREREGISTRATION.md Section 4.
Key point: a necessary MLP layer is not automatically a computation step. If the
output already follows a donor's y when only the early y direction is swapped,
the later layers carry y rather than compute it.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from das_interchange import load_model  # noqa: E402
from train_transformer import load_split  # noqa: E402
import gt_forward as G  # noqa: E402
from stats_utils import wilson_ci  # noqa: E402

FLIP_T = 0.05          # frozen threshold: flip rate
BA_DROP_T = 0.05       # frozen threshold: balanced-accuracy drop (5 points)
DECODE_T = 0.99        # frozen threshold: "y is decodable"
SUPPORT_T = 0.95       # SIIT interchange/chain/bypass support threshold
RESAMPLE_SEEDS = (0, 1, 2)
PAIR_SEED = 123        # donor/recipient pairing for interchange tests
CONTROL_COORD = 32     # unrelated coordinate used as a negative control


# ---------------------------------------------------------------- helpers
def xor(p, q):
    return (p - q).abs()


def variables(task, X):
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    u = xor(a, b)
    v = u * c
    if task == "2hop":
        return dict(a=a, b=b, c=c, u=u, y=v)
    d = X[:, 3]
    return dict(a=a, b=b, c=c, d=d, u=u, v=v, y=xor(v, d))


def balanced_acc(pred, y):
    pos, neg = y == 1, y == 0
    tpr = (pred[pos] == 1).float().mean().item() if pos.any() else float("nan")
    tnr = (pred[neg] == 0).float().mean().item() if neg.any() else float("nan")
    return 0.5 * (tpr + tnr)


def score(mask_ok, mask_den):
    """Proportion of mask_ok within mask_den, with denominator and Wilson CI."""
    n = int(mask_den.sum())
    k = int((mask_ok & mask_den).sum())
    lo, hi = wilson_ci(k, n)
    return dict(score=k / n if n else float("nan"), k=k, n=n, ci_lo=lo, ci_hi=hi)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def siit_sites(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("siit_sites"), ck.get("siit_heads")


# ---------------------------------------------------------------- 1. sanity
def run_sanity(model, task, X, Y):
    pred = (G.forward(model, X) > 0).float()
    return dict(
        bit_acc=(pred == Y).float().mean().item(),
        exact_vector_acc=(pred == Y).all(dim=1).float().mean().item(),
        balanced_acc=balanced_acc(pred, Y),
        frac_y_is_1=Y.mean().item(),
        majority_baseline_acc=max(Y.mean().item(), 1 - Y.mean().item()),
        n_examples=len(X), n_bits=Y.numel(),
    )


# ---------------------------------------------------------------- 2. ablation
def run_ablations(model, task, X, Y):
    clean_logits, cache = G.forward(model, X, return_cache=True)
    clean = (clean_logits > 0).float()
    clean_ba = balanced_acc(clean, Y)
    rows = []

    def record(kind, L, method, seed, interventions):
        pred = (G.forward(model, X, interventions) > 0).float()
        flips = pred != clean
        f = score(flips, torch.ones_like(flips, dtype=torch.bool))
        ba = balanced_acc(pred, Y)
        rows.append(dict(site=kind, layer=L, method=method, seed=seed,
                         flip_rate=f["score"], flips=f["k"], n_bits=f["n"],
                         flip_ci_lo=f["ci_lo"], flip_ci_hi=f["ci_hi"],
                         bit_acc=(pred == Y).float().mean().item(),
                         balanced_acc=ba, ba_drop=clean_ba - ba))

    for L in range(G.n_layers(model)):
        for kind in ("mlp", "attn"):
            act = cache[(kind, L)]
            for s in RESAMPLE_SEEDS:
                perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(s))
                # avoid self-donation: a row paired with itself is not an ablation
                same = perm == torch.arange(len(X))
                perm[same] = (perm[same] + 1) % len(X)
                donor = act[perm]
                record(kind, L, "resample", s, {(kind, L): lambda m, d=donor: d})
            mean = act.mean(dim=0, keepdim=True).expand_as(act)
            record(kind, L, "mean", -1, {(kind, L): lambda m, d=mean: d})
            record(kind, L, "zero", -1, {(kind, L): lambda m: torch.zeros_like(m)})
        block_in = cache[("resid", L - 1)]
        record("block_skip", L, "identity", -1, {("resid", L): lambda h, d=block_in: d})
    return rows, clean_ba


# ---------------------------------------------------------------- 3. decodability
def run_decodability(model, task, Xtr, Xte):
    _, ctr = G.forward(model, Xtr, return_cache=True)
    _, cte = G.forward(model, Xte, return_cache=True)
    vtr, vte = variables(task, Xtr), variables(task, Xte)
    names = ["u", "y"] if task == "2hop" else ["u", "v", "y"]
    rows = []
    for L in range(-1, G.n_layers(model)):
        Htr = ctr[("resid", L)].reshape(-1, 64).numpy()
        Hte = cte[("resid", L)].reshape(-1, 64).numpy()
        for var in names:
            ytr = vtr[var].reshape(-1).numpy().astype(int)
            yte = vte[var].reshape(-1).numpy().astype(int)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Htr, ytr)
            p = torch.tensor(clf.predict(Hte)).float()
            rows.append(dict(layer=L, var=var,
                             balanced_acc=balanced_acc(p, torch.tensor(yte).float()),
                             n_train=len(ytr), n_test=len(yte)))
    return rows


# ---------------------------------------------------------------- 3b. is early y USED?
def run_y_shortcut(model, task, Xtr, Xte, exclude_coords=()):
    """Causal test of the linearly decodable y signal after each non-final block.

    Fit a probe for y on the residual stream after block L (designated SIIT
    coordinates excluded, so the direction cannot simply be the u/v site), then
    swap ONLY the component along that direction from a donor:
        h' = h + ((h_D - h_R) . w) w
    If the output now follows the donor's y, the model uses a one-step shortcut
    at block L. A random direction (same norm) is the control.
    """
    _, ctr = G.forward(model, Xtr, return_cache=True)
    gen = torch.Generator().manual_seed(PAIR_SEED)
    R, D = Xte, Xte[torch.randperm(len(Xte), generator=gen)]
    _, cR = G.forward(model, R, return_cache=True)
    _, cD = G.forward(model, D, return_cache=True)
    yR, yD = variables(task, R)["y"], variables(task, D)["y"]
    ytr = variables(task, Xtr)["y"].reshape(-1).numpy().astype(int)
    changed = yD != yR
    keep = [i for i in range(64) if i not in set(exclude_coords)]
    rows = []
    for L in range(G.n_layers(model) - 1):
        H = ctr[("resid", L)].reshape(-1, 64).numpy()[:, keep]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(H, ytr)
        w = torch.zeros(64)
        w[keep] = torch.tensor(clf.coef_[0], dtype=torch.float32)
        w = w / w.norm()
        rnd = torch.randn(64, generator=torch.Generator().manual_seed(L))
        rnd[list(exclude_coords)] = 0.0
        rnd = rnd / rnd.norm()
        for name, direc in (("probe_direction", w), ("random_direction_control", rnd)):
            hR, hD = cR[("resid", L)], cD[("resid", L)]
            delta = ((hD - hR) @ direc).unsqueeze(-1) * direc
            pred = (G.forward(model, R, {("resid", L): lambda h, d=delta: h + d}) > 0).float()
            rows.append(dict(test="y_shortcut", protocol=f"after_block{L}_{name}", layer=L,
                             excluded_coords=str(list(exclude_coords)),
                             **score(pred == yD, changed)))
    return rows


# ---------------------------------------------------------------- 4-6. SIIT tests
def swap_coord(coord, donor_h):
    def f(h):
        h = h.clone()
        h[..., coord] = donor_h[..., coord]
        return h
    return f


def run_siit_tests(model, task, X, sites, heads):
    """Interchange, chain and bypass tests at the designated SIIT coordinates."""
    gen = torch.Generator().manual_seed(PAIR_SEED)
    R, D = X, X[torch.randperm(len(X), generator=gen)]
    VR, VD = variables(task, R), variables(task, D)
    clean = (G.forward(model, R) > 0).float()
    _, cR = G.forward(model, R, return_cache=True)
    _, cD = G.forward(model, D, return_cache=True)
    site = {s["var"]: s for s in sites}
    rows = []

    def add(test, protocol, ok, den, **extra):
        rows.append(dict(test=test, protocol=protocol, **score(ok, den), **extra))

    # ---- u interchange
    su = site["u"]
    L, j = su["layer"], su["start"]
    tgt = VD["u"] * VR["c"] if task == "2hop" else xor(VD["u"] * VR["c"], VR["d"])
    pred = (G.forward(model, R, {("resid", L): swap_coord(j, cD[("resid", L)])}) > 0).float()
    changed = tgt != clean
    add("interchange_u", "random_pairs_changed_bits", pred == tgt, changed, layer=L, coord=j)
    ctrl = (G.forward(model, R, {("resid", L): swap_coord(CONTROL_COORD, cD[("resid", L)])}) > 0).float()
    add("interchange_u", "control_coord32_changed_bits", ctrl == tgt, changed, layer=L, coord=CONTROL_COORD)

    if task == "3hop" and "v" in site:
        sv = site["v"]
        Lv, jv = sv["layer"], sv["start"]
        tgt_v = xor(VD["v"], VR["d"])
        pred_v = (G.forward(model, R, {("resid", Lv): swap_coord(jv, cD[("resid", Lv)])}) > 0).float()
        changed_v = tgt_v != clean
        add("interchange_v", "random_pairs_changed_bits", pred_v == tgt_v, changed_v, layer=Lv, coord=jv)
        both_c1 = (VR["c"] == 1) & (VD["c"] == 1)
        add("interchange_v", "corrected_both_c1_changed_bits", pred_v == tgt_v, changed_v & both_c1,
            layer=Lv, coord=jv)
        ctrl_v = (G.forward(model, R, {("resid", Lv): swap_coord(CONTROL_COORD, cD[("resid", Lv)])}) > 0).float()
        add("interchange_v", "control_coord32_both_c1", ctrl_v == tgt_v, changed_v & both_c1,
            layer=Lv, coord=CONTROL_COORD)
        # legacy protocol, reported only to show why it was replaced
        R2, D2 = R.clone(), D.clone()
        R2[:, 2, :], D2[:, 2, :] = 1.0, 0.0
        V2R, V2D = variables(task, R2), variables(task, D2)
        clean2 = (G.forward(model, R2) > 0).float()
        _, c2D = G.forward(model, D2, return_cache=True)
        tgt2 = xor(V2D["v"], V2R["d"])
        pred2 = (G.forward(model, R2, {("resid", Lv): swap_coord(jv, c2D[("resid", Lv)])}) > 0).float()
        add("interchange_v", "legacy_decoupled_recipient_c1_donor_c0", pred2 == tgt2, tgt2 != clean2,
            layer=Lv, coord=jv)

        # ---- chain u -> v -> y, with v read by the SIIT readout head
        w, b = heads[f"v_L{Lv}.weight"], heads[f"v_L{Lv}.bias"]
        logits_u, cache_u = G.forward(model, R, {("resid", L): swap_coord(j, cD[("resid", L)])},
                                      return_cache=True)
        v_read = ((cache_u[("resid", Lv)][..., jv:jv + 1] @ w.T + b).squeeze(-1) > 0).float()
        v_cf = VD["u"] * VR["c"]
        y_cf = xor(v_cf, VR["d"])
        ch = y_cf != VR["y"]
        add("chain", "v_propagation", v_read == v_cf, ch)
        add("chain", "y_counterfactual", (logits_u > 0).float() == y_cf, ch)
        restore = {("resid", L): swap_coord(j, cD[("resid", L)]),
                   ("resid", Lv): swap_coord(jv, cR[("resid", Lv)])}
        y_rest = (G.forward(model, R, restore) > 0).float()
        add("chain", "v_restore_reverses_y", y_rest == clean, ch)

    # ---- bypass: ablate an MLP, restore only the designated site
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    same = perm == torch.arange(len(X))
    perm[same] = (perm[same] + 1) % len(X)
    for s in sites:
        Ls, js = s["layer"], s["start"]
        for Lm in range(0, Ls + 1):
            donor_m = cR[("mlp", Lm)][perm]
            abl = {("mlp", Lm): lambda m, d=donor_m: d}
            p_abl = (G.forward(model, R, abl) > 0).float()
            damaged = p_abl != clean
            fix = dict(abl)
            fix[("resid", Ls)] = swap_coord(js, cR[("resid", Ls)])
            p_fix = (G.forward(model, R, fix) > 0).float()
            add("bypass", f"ablate_mlp{Lm}_restore_{s['var']}_site", p_fix == clean, damaged,
                layer=Ls, coord=js, ablated_mlp=Lm)
    return rows


# ---------------------------------------------------------------- D_true
SHORTCUT_USED_T = 0.95   # early y signal is used: output follows donor y on >= 95% of changed bits
SHORTCUT_NONE_T = 0.05   # early y signal is not used: <= 5%


def decide_dtrue(abl_df, dec_df, siit_df, short_df, n_layers):
    """D_true rule (PREREGISTRATION.md Section 4).

    1. necessary MLP layers: mean resample flip >= 5% or balanced-accuracy drop >= 5 points.
    2. shortcut check: for every block L before the last necessary MLP, swap only the
       probe direction for y. If the output follows the donor on >= 95% of changed
       bits, y is computed by block L and causally used there: D_true = L + 1
       (MLP layers 0..L). If every such score is <= 5%, no shortcut: D_true = number
       of necessary MLP layers. Anything in between: "unresolved" (partial shortcut).
    Decodability alone never changes D_true; it is reported for the decodable-vs-used contrast.
    """
    res = abl_df[(abl_df.site == "mlp") & (abl_df.method == "resample")]
    per_layer = res.groupby("layer")[["flip_rate", "ba_drop"]].mean()
    necessary = [int(L) for L, r in per_layer.iterrows() if r.flip_rate >= FLIP_T or r.ba_drop >= BA_DROP_T]
    attn = abl_df[(abl_df.site == "attn") & (abl_df.method == "resample")].groupby("layer")[["flip_rate", "ba_drop"]].mean()
    y = dec_df[dec_df["var"] == "y"].set_index("layer")["balanced_acc"]
    ready = [int(L) for L, v in y.items() if v >= DECODE_T]
    sc = short_df[short_df.protocol.str.endswith("probe_direction")].set_index("layer")["score"]
    out = dict(necessary_mlp_layers=necessary, n_necessary=len(necessary),
               attn_necessary=[int(L) for L, r in attn.iterrows() if r.flip_rate >= FLIP_T or r.ba_drop >= BA_DROP_T],
               y_first_decodable_after_block=min(ready) if ready else None,
               y_shortcut_scores={int(L): round(float(v), 4) for L, v in sc.items()},
               evidence="n/a", exclusive_sites="n/a")
    if not necessary:
        out.update(D_true="unresolved", reason="no necessary MLP layer")
        return out
    last = max(necessary)
    early = {L: v for L, v in sc.items() if L < last}
    used = [L for L, v in early.items() if v >= SHORTCUT_USED_T]
    partial = [L for L, v in early.items() if SHORTCUT_NONE_T < v < SHORTCUT_USED_T]
    if used:
        L = min(used)
        out.update(D_true=L + 1, reason=f"y computed and used after block {L} (shortcut score {early[L]:.3f}); "
                                        f"later MLPs necessary but carry, not compute")
    elif partial:
        out.update(D_true="unresolved", reason="partial shortcut: " +
                   ", ".join(f"after block {L} = {early[L]:.3f}" for L in partial))
        return out
    else:
        out.update(D_true=len(necessary), reason="no early shortcut used; all necessary MLPs compute")
    if siit_df is None or siit_df.empty:
        out["evidence"] = "ablation+shortcut test"
    else:
        key = siit_df[siit_df.test.isin(["interchange_u", "interchange_v", "chain"]) &
                      ~siit_df.protocol.str.contains("control|legacy")]
        out["evidence"] = ("ablation+shortcut test+interchange/chain" if (key.score >= SUPPORT_T).all()
                           else "ablation+shortcut test (interchange/chain below 0.95)")
        byp = siit_df[siit_df.test == "bypass"]
        out["exclusive_sites"] = ("yes" if (byp.score >= SUPPORT_T).all() else
                                  "no: " + "; ".join(f"{r.protocol}={r.score:.2f}" for r in byp.itertuples()))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "boolean_combined.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "v2_matched" / "person1"))
    ap.add_argument("--quick", action="store_true", help="200 test / 1000 train examples (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()

    import re
    pat = re.compile(r"^[23]hop_L\d+(_siit_k\d+)?_seed\d+\.pt$")   # the 16 originals only
    ckpts = sorted(p for p in (ROOT / "checkpoints").glob("*.pt") if pat.match(p.name))
    sanity, abl, dec, siit, dtrue, shortcut = [], [], [], [], [], []
    for ck in ckpts:
        model, task = load_model(str(ck))
        Xte, Yte = load_split(a.data, task, "test")
        Xtr, _ = load_split(a.data, task, "train")
        if a.quick:
            Xte, Yte, Xtr = Xte[:200], Yte[:200], Xtr[:1000]
        mid = ck.stem
        kind = "siit" if "siit" in mid else "baseline"
        base = dict(model_id=mid, task=task, kind=kind, layers=G.n_layers(model))
        print(f"[{time.time() - t0:6.1f}s] {mid}", flush=True)

        sanity.append(dict(base, **run_sanity(model, task, Xte, Yte)))
        ar, _ = run_ablations(model, task, Xte, Yte)
        abl += [dict(base, **r) for r in ar]
        dr = run_decodability(model, task, Xtr, Xte)
        dec += [dict(base, **r) for r in dr]
        sr = []
        excl = ()
        if kind == "siit":
            sites, heads = siit_sites(ck)
            excl = tuple(sorted({s_["start"] for s_ in sites}))
            sr = run_siit_tests(model, task, Xte, sites, heads)
            siit += [dict(base, **r) for r in sr]
        ys = run_y_shortcut(model, task, Xtr, Xte, excl)
        shortcut += [dict(base, **r) for r in ys]
        dtrue.append(dict(base, **decide_dtrue(pd.DataFrame(ar), pd.DataFrame(dr),
                                               pd.DataFrame(sr) if sr else None, pd.DataFrame(ys),
                                               G.n_layers(model))))

    pd.DataFrame(sanity).to_csv(out / "gt_sanity.csv", index=False)
    pd.DataFrame(abl).to_csv(out / "gt_ablation.csv", index=False)
    pd.DataFrame(dec).to_csv(out / "gt_decodability.csv", index=False)
    pd.DataFrame(siit).to_csv(out / "gt_siit_interchange_chain_bypass.csv", index=False)
    pd.DataFrame(shortcut).to_csv(out / "gt_y_shortcut.csv", index=False)
    dt = pd.DataFrame(dtrue)
    dt["necessary_mlp_layers"] = dt["necessary_mlp_layers"].astype(str)
    dt["attn_necessary"] = dt["attn_necessary"].astype(str)
    dt["y_shortcut_scores"] = dt["y_shortcut_scores"].astype(str)
    dt.to_csv(out / "gt_dtrue.csv", index=False)
    cfg = dict(script="scripts/ground_truth_depth.py", git_commit=git_commit(), quick=a.quick,
               thresholds=dict(flip=FLIP_T, ba_drop=BA_DROP_T, decode=DECODE_T, support=SUPPORT_T,
                               shortcut_used=SHORTCUT_USED_T, shortcut_none=SHORTCUT_NONE_T),
               resample_seeds=list(RESAMPLE_SEEDS), pair_seed=PAIR_SEED, control_coord=CONTROL_COORD,
               data=str(Path(a.data).relative_to(ROOT)) if Path(a.data).is_relative_to(ROOT) else a.data,
               checkpoints=[c.name for c in ckpts], torch=torch.__version__,
               python=platform.python_version(), seconds=round(time.time() - t0, 1))
    (out / "gt_config.json").write_text(json.dumps(cfg, indent=2))
    print(dt[["model_id", "necessary_mlp_layers", "y_shortcut_scores", "D_true", "reason"]].to_string(index=False))
    print(f"\nSaved to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
