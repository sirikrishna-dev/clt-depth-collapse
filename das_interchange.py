"""Subspace interchange test: is the intermediate (u, v) a causally used variable?

Run from the repo root, e.g.
  python das_interchange.py --ckpt checkpoints/2hop_L2_seed0.pt
Layer index: -1 = after input_proj (negative control), 0 = after encoder block 1,
1 = after block 2 (final).

For each hypothesis H (u, a, b, c, ... y) we learn an orthogonal rotation of the
64-dim hidden state and swap the first k rotated coordinates (at every bit-position
token) from a donor input into a recipient input. The target is the output the
model *should* give if only variable H were swapped. We report accuracy on the
bits where that target differs from the recipient's clean output (acc_flip).
"""
import argparse, sys
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "src")
from train_transformer import BitVectorTransformer, load_split


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    task = ck["task"]
    m = BitVectorTransformer(3 if task == "2hop" else 4, ck["layers"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, task


class Swap(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.rot = nn.utils.parametrizations.orthogonal(nn.Linear(d, d, bias=False))
        self.k = k

    def forward(self, hR, hD):
        W = self.rot.weight  # orthogonal; rotated coords z = h @ W.T, back: h = z @ W
        zR, zD = hR @ W.T, hD @ W.T
        z = torch.cat([zD[..., : self.k], zR[..., self.k:]], dim=-1)
        return z @ W


def hidden(model, x, layer):
    h = model.input_proj(x.transpose(1, 2))
    if layer == -1:
        return h
    for i, l in enumerate(model.encoder.layers):
        h = l(h)
        if i == layer:
            return h


def run(model, x, layer, hD, swap):
    h = model.input_proj(x.transpose(1, 2))
    if layer == -1:
        h = swap(h, hD)
    for i, l in enumerate(model.encoder.layers):
        h = l(h)
        if i == layer:
            h = swap(h, hD)
    return model.head(h).squeeze(-1)


def xor(p, q):
    return (p - q).abs()


def targets(task, hyp, R, D):
    g = lambda T, i: T[:, i, :]
    aR, bR, cR = g(R, 0), g(R, 1), g(R, 2)
    aD, bD, cD = g(D, 0), g(D, 1), g(D, 2)
    dR = g(R, 3) if task == "3hop" else None
    dD = g(D, 3) if task == "3hop" else None
    uR, uD = xor(aR, bR), xor(aD, bD)

    def out(u, c, d=None):
        v = u * c
        return v if task == "2hop" else xor(v, d)

    clean = out(uR, cR, dR)
    t = {
        "u": out(uD, cR, dR),
        "a": out(xor(aD, bR), cR, dR),
        "b": out(xor(aR, bD), cR, dR),
        "c": out(uR, cD, dR),
        "y": out(uD, cD, dD),
    }
    if task == "3hop":
        t["v"] = xor(uD * cD, dR)
        t["d"] = xor(uR * cR, dD)
    return t[hyp], clean


def decouple_pair(R, D):
    """Force c=1 in the recipient and c=0 in the donor, so swapping the donor's
    y (which is 0 everywhere) can no longer imitate swapping u."""
    R, D = R.clone(), D.clone()
    R[:, 2, :] = 1.0
    D[:, 2, :] = 0.0
    return R, D


def fit_eval(model, task, layer, hyp, k, Xtr, Xte, steps, seed, decouple=False, identity_init=False):
    torch.manual_seed(seed)
    swap = Swap(64, k)

    if identity_init:
        with torch.no_grad():
            swap.rot.weight = torch.eye(64)

    opt = torch.optim.Adam(swap.parameters(), lr=3e-3)
    for _ in range(steps):
        R = Xtr[torch.randint(len(Xtr), (512,))]
        D = Xtr[torch.randint(len(Xtr), (512,))]
        if decouple:
            R, D = decouple_pair(R, D)
        tgt, _ = targets(task, hyp, R, D)
        with torch.no_grad():
            hD = hidden(model, D, layer)
        loss = F.binary_cross_entropy_with_logits(run(model, R, layer, hD, swap), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = torch.Generator().manual_seed(123)
    R, D = Xte, Xte[torch.randperm(len(Xte), generator=g)]
    if decouple:
        R, D = decouple_pair(R, D)
    tgt, clean = targets(task, hyp, R, D)
    with torch.no_grad():
        hD = hidden(model, D, layer)
        pred = (run(model, R, layer, hD, swap) > 0).float()
    mask = tgt != clean
    return dict(
        acc_all=(pred == tgt).float().mean().item(),
        acc_flip=(pred == tgt)[mask].float().mean().item() if mask.any() else float("nan"),
        n_flip=int(mask.sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/boolean_combined.csv")
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, 0, 1])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--decouple", action="store_true", help="recipient c=1, donor c=0")
    ap.add_argument("--identity-init", action="store_true", help="Initialize DAS rotation to identity")

    ap.add_argument("--out", default="results/das_interchange.csv")
    a = ap.parse_args()
    model, task = load_model(a.ckpt)
    Xtr, _ = load_split(a.data, task, "train")
    Xte, _ = load_split(a.data, task, "test")
    hyps = ["u", "a", "b", "c", "y"] if task == "2hop" else ["u", "v", "a", "b", "c", "d", "y"]
    rows = []
    for layer in a.layers:
        for hyp in hyps:
            for k in a.ks:
                r = fit_eval(model, task, layer, hyp, k, Xtr, Xte, a.steps, seed=0, decouple=a.decouple, identity_init=a.identity_init)
                rows.append(dict(ckpt=a.ckpt, layer=layer, hyp=hyp, k=k, **r))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.pivot_table(index=["layer", "hyp"], columns="k", values="acc_flip").round(3))


if __name__ == "__main__":
    main()