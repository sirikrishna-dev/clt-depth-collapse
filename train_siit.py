"""Train models whose intermediate variable (u, v) is forced into a fixed subspace
and forced to be *used* (SIIT-style interchange-intervention training).

Run from the repo root (next to src/, data/, checkpoints/):
  python train_siit.py --task 2hop --layers 2 --seed 0
  python train_siit.py --task 3hop --layers 3 --seed 0      # needs >=3 layers for u AND v

Sites (hidden state of each bit-position token, after the given encoder block):
  2hop            : u in coords [0:k) after block 0
  3hop, layers>=3 : u in coords [0:k) after block 0, v in coords [0:k) after block 1
  3hop, layers==2 : only v in coords [0:k) after block 0 (two blocks cannot hold three hops)

Loss = clean BCE
     + lambda_swap * BCE(output after swapping the site coords from a donor input,
                         the output the circuit gives if only that variable is swapped)
     + lambda_aux  * BCE(linear readout of the site coords -> the variable)

Checkpoints are saved under a new name (..._siit_...) so the old zoo is untouched, in the
same format das_interchange.py loads. Always re-check a finished model with
  python das_interchange.py --ckpt <ckpt> --layers 0 --ks 1 2 --steps 400 --decouple
(a learned-rotation test that does not know where we put the variable).
"""
import argparse, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")
from train_transformer import BitVectorTransformer, load_split

bce = F.binary_cross_entropy_with_logits


def xor(p, q):
    return (p - q).abs()


def sites_for(task, layers, k):
    if task == "2hop":
        return [dict(var="u", layer=0, start=0, end=k)]
    if layers >= 3:
        return [dict(var="u", layer=0, start=0, end=k), dict(var="v", layer=1, start=0, end=k)]
    return [dict(var="v", layer=0, start=0, end=k)]


def var_value(var, X):
    u = xor(X[:, 0], X[:, 1])
    return u if var == "u" else u * X[:, 2]


def clean_y(task, X):
    v = var_value("v", X)
    return v if task == "2hop" else xor(v, X[:, 3])


def cf_target(task, var, R, D):
    """Output if only `var` is swapped in from donor D into recipient R."""
    uD = xor(D[:, 0], D[:, 1])
    if task == "2hop":
        return uD * R[:, 2]
    if var == "u":
        return xor(uD * R[:, 2], R[:, 3])
    return xor(uD * D[:, 2], R[:, 3])  # v from donor, d from recipient


def decouple_rows(R, D, frac):
    """For a fraction of rows force c=1 (recipient) and c=0 (donor), so swapping the
    donor's y can never imitate swapping u."""
    R, D = R.clone(), D.clone()
    m = torch.rand(len(R), device=R.device) < frac
    R[m, 2, :] = 1.0
    D[m, 2, :] = 0.0
    return R, D


def all_hidden(model, x):
    h = model.input_proj(x.transpose(1, 2))
    hs = []
    for blk in model.encoder.layers:
        h = blk(h)
        hs.append(h)
    return hs


def swap_forward(model, x, donor_hs, site):
    h = model.input_proj(x.transpose(1, 2))
    for i, blk in enumerate(model.encoder.layers):
        h = blk(h)
        if i == site["layer"]:
            s, e = site["start"], site["end"]
            h = torch.cat([h[..., :s], donor_hs[i][..., s:e], h[..., e:]], dim=-1)
    return model.head(h).squeeze(-1)


def losses(model, heads, sites, task, R, D, frac, lam_swap, lam_aux):
    l_clean = bce(model(R), clean_y(task, R))
    l_swap, l_aux = 0.0, 0.0
    for s in sites:
        Rs, Ds = decouple_rows(R, D, frac) if s["var"] == "u" else (R, D)
        hsD = all_hidden(model, Ds)
        logits = swap_forward(model, Rs, hsD, s)
        l_swap = l_swap + bce(logits, cf_target(task, s["var"], Rs, Ds))
        key = f"{s['var']}_L{s['layer']}"
        read = heads[key](hsD[s["layer"]][..., s["start"]:s["end"]]).squeeze(-1)
        l_aux = l_aux + bce(read, var_value(s["var"], Ds))
    total = l_clean + lam_swap * l_swap + lam_aux * l_aux
    return total, l_clean.item(), float(l_swap), float(l_aux)


@torch.no_grad()
def iia(model, task, site, R, D):
    """Accuracy on the bits where the swap should change the output."""
    tgt = cf_target(task, site["var"], R, D)
    clean = clean_y(task, R)
    pred = (swap_forward(model, R, all_hidden(model, D), site) > 0).float()
    mask = tgt != clean
    return (pred == tgt)[mask].float().mean().item() if mask.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/boolean_combined.csv")
    ap.add_argument("--task", choices=["2hop", "3hop"], default="2hop")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=1, help="width of the variable's subspace")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-swap", type=float, default=1.0)
    ap.add_argument("--lambda-aux", type=float, default=0.5)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    trX, _ = load_split(a.data, a.task, "train")
    vaX, _ = load_split(a.data, a.task, "validation")
    teX, _ = load_split(a.data, a.task, "test")
    trX, vaX, teX = trX.to(dev), vaX.to(dev), teX.to(dev)

    sites = sites_for(a.task, a.layers, a.k)
    model = BitVectorTransformer(trX.shape[1], a.layers).to(dev)
    heads = nn.ModuleDict({f"{s['var']}_L{s['layer']}": nn.Linear(a.k, 1) for s in sites}).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=a.lr)
    loader = DataLoader(TensorDataset(trX), batch_size=a.batch_size, shuffle=True)

    best, best_state, best_heads = float("inf"), None, None
    for ep in range(a.epochs):
        model.train()
        for (x,) in loader:
            D = x[torch.randperm(len(x), device=x.device)]
            loss, *_ = losses(model, heads, sites, a.task, x, D, 0.5, a.lambda_swap, a.lambda_aux)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(1)
            Dv = vaX[torch.randperm(len(vaX), generator=g).to(dev)]
            vl, lc, ls, la = losses(model, heads, sites, a.task, vaX, Dv, 0.5, a.lambda_swap, a.lambda_aux)
        if vl.item() < best:
            best = vl.item()
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            best_heads = {k_: v.detach().cpu().clone() for k_, v in heads.state_dict().items()}
        print(f"epoch={ep+1} val_total={vl.item():.4f} clean={lc:.4f} swap={ls:.4f} aux={la:.4f}", flush=True)

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        preds = (model(teX) > 0).float()
        Y = clean_y(a.task, teX)
        bit_acc = (preds == Y).float().mean().item()
        exact = (preds == Y).all(dim=1).float().mean().item()
        g = torch.Generator(device="cpu").manual_seed(7)
        Dt = teX[torch.randperm(len(teX), generator=g).to(dev)]
        Rdec, Ddec = decouple_rows(teX, Dt, 1.0)
        report = {}
        for s in sites:
            ctrl = dict(s, start=32, end=32 + a.k)  # control: same swap on unrelated coords
            report[f"{s['var']}_L{s['layer']}"] = dict(
                iia_random_pairs=iia(model, a.task, s, teX, Dt),
                iia_decoupled=iia(model, a.task, s, Rdec, Ddec),
                control_other_coords_decoupled=iia(model, a.task, ctrl, Rdec, Ddec),
            )

    os.makedirs(a.out, exist_ok=True)
    model_id = f"{a.task}_L{a.layers}_siit_k{a.k}_seed{a.seed}"
    ckpt = os.path.join(a.out, model_id + ".pt")
    torch.save({"state_dict": model.state_dict(), "task": a.task, "layers": a.layers, "seed": a.seed,
                "n_bits": 16, "siit_sites": sites, "siit_heads": best_heads}, ckpt)
    meta = {
        "model_id": model_id, "source": "siit_trained_transformer", "task": a.task,
        "hop_depth": 2 if a.task == "2hop" else 3, "layers": a.layers, "seed": a.seed,
        "checkpoint": ckpt, "test_bit_accuracy": bit_acc, "test_exact_vector_accuracy": exact,
        "intermediates": {f"{s['var']}": {"definition": "XOR(a,b)" if s["var"] == "u" else "AND(u,c)",
                                          "location": f"after encoder block {s['layer']}, hidden dims [{s['start']}:{s['end']}) per bit token",
                                          "how_placed": "interchange-intervention training"} for s in sites},
        "training_interchange_check": report,
        "validated": False,
        "validation_todo": "run das_interchange.py --decouple on this checkpoint; set validated only if u beats the donor-y baseline",
    }
    os.makedirs("metadata", exist_ok=True)
    with open(os.path.join("metadata", model_id + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join("metadata", "model_zoo.jsonl"), "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()