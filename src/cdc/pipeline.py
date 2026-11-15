"""Experiment stages.  All outputs are CSV/JSON under cfg['out_dir'];
checkpoints under cfg['ckpt_dir'].  Every stage skips work that already exists."""
from __future__ import annotations

import json
import os

import pandas as pd
import torch

from .tasks import make_dataset
from .model import ModelConfig, save_model, load_model


def _paths(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    return cfg["out_dir"], cfg["ckpt_dir"]


def model_grid(cfg):
    for D in cfg["depths"]:
        for extra in cfg.get("overprovision", [0]):
            for seed in cfg["seeds"]:
                yield D, D + extra, seed


def model_name(D, L, seed):
    return f"siit_D{D}_L{L}_seed{seed}"


def _append_csv(path, rows):
    """Append rows; columns may differ between models (e.g. per-layer metrics), so merge."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path) and os.path.getsize(path) > 1:
        df = pd.concat([pd.read_csv(path), df], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


# ------------------------------------------------------------------------- zoo
def stage_zoo(cfg, device="cpu"):
    from .siit import train_siit, certify
    out, ck = _paths(cfg)
    s = cfg["siit"]
    max_retries = s.get("max_retries", 1)
    for D, L, seed in model_grid(cfg):
        name = model_name(D, L, seed)
        path = os.path.join(ck, name + ".pt")
        cert_path = os.path.join(out, "certificates", name + ".json")
        prev = json.load(open(cert_path)) if os.path.exists(cert_path) else None
        if prev is not None and os.path.exists(path):
            n_att = len(prev.get("attempts", [])) + 1
            if prev["certified"] or n_att > max_retries:
                continue
            steps = s["steps"] * 2 ** n_att        # pre-specified retry rule: double the steps
            attempts = prev.get("attempts", []) + [{k: prev[k] for k in ("certified", "circuit_only_acc",
                                                                         "iia_cf_min", "strict_min", "bypass_min")}
                                                    | {"steps": prev.get("steps", s["steps"])}]
        else:
            steps, attempts = s["steps"], []
        print("training", name, "steps", steps, flush=True)
        data = make_dataset(D)
        mcfg = ModelConfig(D=D, n_layers=L)
        model, hist = train_siit(mcfg, data, seed=seed, steps=steps, batch=s["batch"],
                                 lr=s["lr"], strict=s.get("strict", True), device=device,
                                 p_subset=s.get("p_subset", 0.3), w_site=s.get("w_site", 0.3),
                                 log_every=max(steps // 4, 1))
        model = model.cpu()
        cert = certify(model, data)
        cert.update({"model": name, "seed": seed, "steps": steps, "attempts": attempts, "train_history": hist})
        save_model(model, path, {"name": name, "seed": seed, "siit": s, "certificate": cert})
        os.makedirs(os.path.dirname(cert_path), exist_ok=True)
        json.dump(cert, open(cert_path, "w"), indent=1)
    rows = []
    cdir = os.path.join(out, "certificates")
    for f in sorted(os.listdir(cdir)):
        c = json.load(open(os.path.join(cdir, f)))
        flat = {k: v for k, v in c.items() if not isinstance(v, (dict, list))}
        flat.update({f"strict_{k}": v for k, v in c["strict_scores"].items()})
        flat.update({f"bypass_{k}": v["score"] for k, v in c["bypass"].items()})
        flat.update({f"bypass_heldout_{k}": v["heldout_meanablate_score"] for k, v in c["bypass"].items()})
        flat.update({f"mlp_meanablate_acc_{k}": v for k, v in c["mlp_meanablate_acc"].items()})
        flat.update({f"backup_fullmodel_acc_{k}": v for k, v in c["mlp_meanablate_acc_fullmodel"].items()})
        flat.update({f"site_sep_{k}": v for k, v in c["site_separation"].items()})
        rows.append(flat)
    pd.DataFrame(rows).to_csv(os.path.join(out, "ground_truth_certificates.csv"), index=False)
    print(pd.DataFrame(rows)[["model", "clean_bit_acc", "iia_cf_min", "strict_min", "bypass_min", "circuit_only_acc", "n_necessary_mlps", "certified"]])


def certified_models(cfg):
    out, ck = _paths(cfg)
    df = pd.read_csv(os.path.join(out, "ground_truth_certificates.csv"))
    return df[df.certified].model.tolist(), df


def _models_to_use(cfg):
    names, df = certified_models(cfg)
    skipped = sorted(set(df.model) - set(names))
    if skipped:
        print("NOT certified (excluded from depth analysis):", skipped)
    return names


def tc_name(model, kind, k):
    return f"{model}__{kind}_k{k}"


# ---------------------------------------------------------------- replacements
def stage_replacements(cfg, device="cpu"):
    from .transcoders import make_transcoder, collect_mlp_io, train_transcoder, fidelity, save_tc
    out, ck = _paths(cfg)
    r = cfg["replacements"]
    csv = os.path.join(out, "replacement_fidelity.csv")
    done = set(pd.read_csv(csv).tc) if os.path.exists(csv) else set()
    for name in _models_to_use(cfg):
        model, meta = load_model(os.path.join(ck, name + ".pt"))
        D = model.cfg.D
        data = make_dataset(D)
        todo = [(kind, k) for kind in r["kinds"] for k in r["k_values"] if tc_name(name, kind, k) not in done]
        if not todo:
            continue
        ins, outs = collect_mlp_io(model, data.subset("train").x)
        te = data.subset("test")
        for kind, k in todo:
            tn = tc_name(name, kind, k)
            tc = make_transcoder(kind, model.cfg.n_layers, model.cfg.d_model, r["n_features"], k)
            tc, hist = train_transcoder(tc, ins, outs, steps=r["steps"], batch=r["batch"], lr=r["lr"],
                                        seed=meta["seed"], device=device)
            fid = fidelity(tc, model, te.x, te.y)
            save_tc(tc, os.path.join(ck, tn + ".pt"), {"model": name, "kind": kind, "k": k, "hist": hist,
                                                         "budget": {"n_features_per_layer": tc.cfg.n_features,
                                                                    "k": k, "steps": r["steps"],
                                                                    "batch": r["batch"], "lr": r["lr"]}})
            row = {"tc": tn, "model": name, "D": D, "n_layers": model.cfg.n_layers, "seed": meta["seed"],
                   "kind": kind, "k": k, "n_features": tc.cfg.n_features, "steps": r["steps"], **fid}
            _append_csv(csv, [row])
            print(tn, {kk: round(v, 4) for kk, v in fid.items() if kk in ("rel_mse_mean", "e2e_bit_acc", "e2e_kl")}, flush=True)


# ---------------------------------------------------------------------- graphs
def _pruned_metrics(g, keep_frac):
    from .graphs import prune_graph
    from .depth_metrics import graph_metrics
    import copy
    keep = prune_graph(g, keep_frac)
    g2 = copy.copy(g)
    A = g.A.copy(); A[~keep, :] = 0; A[:, ~keep] = 0
    g2.A = A
    m = graph_metrics(g2)
    return {f"prune{int(keep_frac*100)}_root_D_feat": m["root_D_feat"],
            f"prune{int(keep_frac*100)}_n_features": int(keep[g.node_type == "feature"].sum())}


def stage_graphs(cfg, device="cpu"):
    from .transcoders import load_tc
    from .graphs import build_graph, conservation_error
    from .depth_metrics import graph_metrics, ground_truth_graph
    from .validation import feature_alignment, aligned_nodes
    out, ck = _paths(cfg)
    csv = os.path.join(out, "graph_metrics_per_example.csv")
    acsv = os.path.join(out, "feature_alignment.csv")
    done = set(pd.read_csv(csv).tc) if os.path.exists(csv) else set()
    n_ex = cfg["graphs"]["n_examples"]
    fid = pd.read_csv(os.path.join(out, "replacement_fidelity.csv"))
    for _, fr in fid.iterrows():
        if fr.tc in done:
            continue
        model, meta = load_model(os.path.join(ck, fr.model + ".pt"))
        tc = load_tc(os.path.join(ck, fr.tc + ".pt"))
        te = make_dataset(model.cfg.D).subset("test")
        arows, aligned = feature_alignment(model, tc, te.x)
        _append_csv(acsv, [{"tc": fr.tc, **a} for a in arows])
        rows = []
        for i in range(n_ex):
            g = build_graph(model, tc, te.x[i])
            m = graph_metrics(g, aligned_nodes(g, aligned))
            m.update(_pruned_metrics(g, 0.8)); m.update(_pruned_metrics(g, 0.9))
            m.update({"tc": fr.tc, "model": fr.model, "D": fr.D, "n_layers": fr.n_layers, "seed": fr.seed,
                      "kind": fr.kind, "k": fr.k, "example": i, "conservation_err": conservation_error(g)})
            rows.append(m)
        _append_csv(csv, rows)
        df = pd.DataFrame(rows)
        print(fr.tc, "D_feat(root) = %.3f  D=%d  skip=%.3f  maxcons=%.1e" % (
            df.root_D_feat.mean(), fr.D, df.skip_mass.mean(), df.conservation_err.max()), flush=True)
    gt = []
    for D in cfg["depths"]:
        m = graph_metrics(ground_truth_graph(D))
        gt.append({"D": D, **m})
    pd.DataFrame(gt).to_csv(os.path.join(out, "ground_truth_graph_metrics.csv"), index=False)


# -------------------------------------------------------------------- validate
def stage_validate(cfg, device="cpu"):
    from .transcoders import load_tc
    from .graphs import build_graph
    from .validation import skip_dependence, ablation_check, naive_patch_selectivity
    out, ck = _paths(cfg)
    fid = pd.read_csv(os.path.join(out, "replacement_fidelity.csv"))
    sd_csv = os.path.join(out, "skip_dependence.csv")
    ab_csv = os.path.join(out, "ablation_check.csv")
    nv_csv = os.path.join(out, "naive_patching.csv")
    done = set(pd.read_csv(sd_csv).tc) if os.path.exists(sd_csv) else set()
    for _, fr in fid.iterrows():
        if fr.tc in done:
            continue
        model, _ = load_model(os.path.join(ck, fr.model + ".pt"))
        tc = load_tc(os.path.join(ck, fr.tc + ".pt"))
        te = make_dataset(model.cfg.D).subset("test")
        sd = skip_dependence(model, tc, te.x)
        _append_csv(sd_csv, [{"tc": fr.tc, "model": fr.model, "D": fr.D, "seed": fr.seed, "kind": fr.kind,
                              "k": fr.k, **sd}])
        ab = []
        for i in range(5):
            g = build_graph(model, tc, te.x[i])
            for r in ablation_check(model, tc, g, te.x[i]):
                ab.append({"tc": fr.tc, "D": fr.D, "kind": fr.kind, "k": fr.k, "seed": fr.seed, "example": i, **r})
        _append_csv(ab_csv, ab)
        print(fr.tc, sd, flush=True)
    nv_done = set(pd.read_csv(nv_csv).model) if os.path.exists(nv_csv) else set()
    for name in _models_to_use(cfg):
        if name in nv_done:
            continue
        model, meta = load_model(os.path.join(ck, name + ".pt"))
        rows = naive_patch_selectivity(model, make_dataset(model.cfg.D))
        _append_csv(nv_csv, [{"model": name, "D": model.cfg.D, "seed": meta["seed"], **r} for r in rows])


# ---------------------------------------------------------------------- probes
def stage_probes(cfg, device="cpu"):
    from .validation import probes_and_causal
    out, ck = _paths(cfg)
    csv = os.path.join(out, "probes_causal_2x2.csv")
    done = set(pd.read_csv(csv).model) if os.path.exists(csv) else set()
    for name in _models_to_use(cfg):
        if name in done:
            continue
        model, meta = load_model(os.path.join(ck, name + ".pt"))
        if model.cfg.D < 2:
            continue
        rows = probes_and_causal(model, make_dataset(model.cfg.D), n_perm=cfg["probes"]["n_perm"])
        _append_csv(csv, [{"model": name, "D": model.cfg.D, "seed": meta["seed"], **r} for r in rows])
        print(name, "probes done", flush=True)
