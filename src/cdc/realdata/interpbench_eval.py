"""Tier 2 (advisor point f): InterpBench SIIT models (Gupta et al., NeurIPS 2024 D&B).

InterpBench releases SIIT-trained transformers with KNOWN circuits
  code:   https://github.com/FlyingPumba/InterpBench   (MIT)
  models: https://huggingface.co/cybershiptrooper/InterpBench   (CC-BY)

Plan for this tier:
  1. List the released cases; keep those whose ground-truth circuit contains a chain of
     >= 2 MLP nodes (true MLP depth >= 2), plus IOI as the realistic case.
  2. For each kept case: train PLT and CLT on the model's MLPs at the same budget
     (same features/layer, same TopK k, same steps), using the case's own input sampler.
  3. Build attribution graphs with the autograd engine (generic_tf.build_graph) and
     compute D_feat / skip mass / IMF exactly as for the Boolean zoo; compare D_feat to
     the circuit's MLP depth.

Run on Colab (internet needed):
    pip install huggingface_hub
    python -m cdc.realdata.interpbench_eval --list
    python -m cdc.realdata.interpbench_eval --case <CASE_ID> --out results/interpbench

VERIFY before relying on this script (not runnable in the authoring environment):
  * the folder layout inside the HF repo (this script prints it -- adjust `find_files`);
  * how each case's inputs are generated: InterpBench's repo provides the task / dataset
    code for every case -- install it (`pip install git+https://github.com/FlyingPumba/InterpBench`
    or clone it) and plug its sampler into `sample_inputs`;
  * the ground-truth circuit (list of nodes) for each case, to compute its MLP depth.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from ..transcoders import make_transcoder, train_transcoder
from ..depth_metrics import graph_metrics
from .generic_tf import from_transformerlens_state_dict

REPO = "cybershiptrooper/InterpBench"


def list_repo():
    from huggingface_hub import list_repo_files
    files = list_repo_files(REPO)
    cases = sorted({f.split("/")[0] for f in files if "/" in f})
    return files, cases


def find_files(case: str, local_dir: str):
    from huggingface_hub import snapshot_download
    path = snapshot_download(REPO, allow_patterns=[f"{case}/*"], local_dir=local_dir)
    root = os.path.join(path, case)
    found = {"weights": None, "cfg": None, "meta": []}
    for dirpath, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dirpath, fn)
            if fn.endswith((".pth", ".pt")) and "ll_model" in fn:
                found["weights"] = full
            elif ("cfg" in fn or "config" in fn) and fn.endswith((".pkl", ".json")):
                found["cfg"] = full
            else:
                found["meta"].append(full)
    return found


def load_case(found: dict):
    if found["cfg"].endswith(".json"):
        cfg = json.load(open(found["cfg"]))
    else:
        cfg = pickle.load(open(found["cfg"], "rb"))
        cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    sd = torch.load(found["weights"], map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "embed.W_E" not in sd:
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd.state_dict()
    return from_transformerlens_state_dict(cfg, sd), cfg


def sample_inputs(case: str, cfg: dict, n: int, seed: int = 0) -> torch.Tensor:
    """REPLACE with the InterpBench sampler for this case.  The fallback (uniform random tokens)
    is only for smoke-testing the pipeline and must not be used for results."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg["d_vocab"], (n, cfg["n_ctx"]), generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--out", default="results/interpbench")
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-graphs", type=int, default=50)
    ap.add_argument("--mlp-depth", type=int, required=False,
                    help="MLP depth of the case's ground-truth circuit (from its circuit definition)")
    a = ap.parse_args()
    if a.list:
        files, cases = list_repo()
        print(len(cases), "cases:", cases)
        return
    os.makedirs(a.out, exist_ok=True)
    found = find_files(a.case, os.path.join(a.out, "hf"))
    print("files:", json.dumps(found, indent=1)[:2000])
    model, cfg = load_case(found)
    X = sample_inputs(a.case, cfg, a.n_train)
    ins, outs = model.mlp_io(X)
    rows = []
    for kind in ("plt", "clt"):
        tc = make_transcoder(kind, model.cfg.n_layers, model.cfg.d_model, a.n_features, a.k)
        tc, _ = train_transcoder(tc, ins, outs, steps=a.steps)
        for p in tc.parameters():
            p.requires_grad_(False)
        for i in range(a.n_graphs):
            toks = X[i]
            logits, _ = model.run(toks)
            tgt = int(logits[-1].argmax())
            g, info = model.build_graph(toks, tc, tgt)
            m = graph_metrics(g, roots=np.arange(len(toks)))
            rows.append({"case": a.case, "kind": kind, "example": i, "mlp_depth": a.mlp_depth,
                         "coverage": info["coverage"], **m})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"interpbench_{a.case}.csv"), index=False)
    print(df.groupby("kind")[["root_signed_D_feat", "root_D_feat", "skip_mass", "coverage"]].mean())


if __name__ == "__main__":
    main()
