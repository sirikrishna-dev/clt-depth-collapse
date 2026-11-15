"""Per-layer (PLT) and cross-layer (CLT) transcoders with an identical budget.

Both map MLP input (post-LN2) at layer l to MLP output.  Feature activations
use TopK (k active features per token per layer), so L0 is matched exactly
between PLT and CLT instead of being left to an L1 coefficient (Round 1 used
L1 = 1e-4 for both, which does not match L0).

  PLT: feature at layer l decodes only to MLP_l.
  CLT: feature at layer l decodes to every MLP_l' with l' >= l (Anthropic 2025).

Budget modes (see docs/hypotheses.md):
  'features' (primary): same n_features per layer, same k, same steps/lr/data.
  'params'  (secondary): PLT gets more features per layer so its total parameter
                         count equals the CLT's ('plt_parammatched').
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TCConfig:
    kind: str            # 'plt' | 'clt'
    n_layers: int
    d_model: int
    n_features: int      # per layer
    k: int               # TopK per token per layer

    def to_dict(self):
        return asdict(self)


def topk_relu(pre: torch.Tensor, k: int) -> torch.Tensor:
    pre = torch.relu(pre)
    if k >= pre.shape[-1]:
        return pre
    vals, idx = pre.topk(k, dim=-1)
    out = torch.zeros_like(pre)
    return out.scatter(-1, idx, vals)


class Transcoder(nn.Module):
    def __init__(self, cfg: TCConfig):
        super().__init__()
        self.cfg = cfg
        L, d, Fn = cfg.n_layers, cfg.d_model, cfg.n_features
        self.W_enc = nn.Parameter(torch.randn(L, Fn, d) / math.sqrt(d))
        self.b_enc = nn.Parameter(torch.zeros(L, Fn))
        # decoder W_dec[(src, dst)] : (Fn, d); PLT only has src == dst
        self.pairs = [(s, t) for s in range(L) for t in range(L)
                      if (t == s if cfg.kind == "plt" else t >= s)]
        self.W_dec = nn.ParameterDict({
            f"{s}_{t}": nn.Parameter(self.W_enc.data[s].clone() * (0.1 if t != s else 1.0) / math.sqrt(Fn / d))
            for s, t in self.pairs})
        self.b_dec = nn.Parameter(torch.zeros(L, d))

    def dec(self, s, t):
        return self.W_dec[f"{s}_{t}"]

    def dsts(self, s):
        return [t for (a, t) in self.pairs if a == s]

    def encode(self, mlp_in_l: torch.Tensor, l: int) -> torch.Tensor:
        pre = mlp_in_l @ self.W_enc[l].T + self.b_enc[l]
        return topk_relu(pre, self.cfg.k)

    def preact(self, mlp_in_l, l):
        return mlp_in_l @ self.W_enc[l].T + self.b_enc[l]

    def decode_all(self, acts: list[torch.Tensor]) -> list[torch.Tensor]:
        """acts[l]: (..., Fn).  Returns reconstructions of every MLP output."""
        outs = [self.b_dec[t].expand(*acts[0].shape[:-1], -1).clone() for t in range(self.cfg.n_layers)]
        for s, t in self.pairs:
            outs[t] = outs[t] + acts[s] @ self.dec(s, t)
        return outs

    def forward(self, mlp_ins: list[torch.Tensor]):
        acts = [self.encode(mlp_ins[l], l) for l in range(self.cfg.n_layers)]
        return self.decode_all(acts), acts

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -------- online use inside the original model (full replacement model)
    def as_mlp_fns(self, state: dict | None = None, frozen_acts: dict | None = None,
                   skip_from_frozen: bool = False, errors: list | None = None):
        """Return {l: fn} usable as model iv['mlp_fn'].  Features of layer l are computed
        from the *current* MLP input at layer l; their writes to later layers are stored
        in `state` and added when those layers run.

        frozen_acts={l: acts} uses fixed activations at layer l (for interventions).
        skip_from_frozen=True: the cross-layer (skip) writes of every layer use
        frozen_acts (base activations) while same-layer writes use current ones.
        errors: optional per-layer error tensors added to every output (local model).
        """
        state = {} if state is None else state
        state.setdefault("pending", {})
        L = self.cfg.n_layers

        def fn(m_in, l):
            if frozen_acts is not None and l in frozen_acts and not skip_from_frozen:
                a = frozen_acts[l]
            else:
                a = self.encode(m_in, l)
            state.setdefault("acts", {})[l] = a
            for t in self.dsts(l):
                src_a = a
                if skip_from_frozen and t != l and frozen_acts is not None:
                    src_a = frozen_acts[l]
                w = src_a @ self.dec(l, t)
                state["pending"][t] = state["pending"].get(t, 0) + w
            out = self.b_dec[l] + state["pending"].pop(l)
            if errors is not None:
                out = out + errors[l]
            return out

        return {l: fn for l in range(L)}


def make_transcoder(kind: str, n_layers: int, d_model: int, n_features: int, k: int) -> Transcoder:
    if kind == "plt_parammatched":
        # match total parameters of a CLT with n_features per layer
        n_pairs_clt = n_layers * (n_layers + 1) // 2
        per_feat_plt = 2 * d_model * n_layers + n_layers  # enc + dec + b_enc per layer
        clt_params = n_layers * n_features * d_model + n_pairs_clt * n_features * d_model + n_layers * n_features
        nf = int(round(clt_params / per_feat_plt))
        return Transcoder(TCConfig("plt", n_layers, d_model, max(nf, n_features), k))
    return Transcoder(TCConfig(kind, n_layers, d_model, n_features, k))


@torch.no_grad()
def collect_mlp_io(model, x: np.ndarray, batch: int = 500, device="cpu"):
    ins = [[] for _ in model.blocks]
    outs = [[] for _ in model.blocks]
    for i in range(0, len(x), batch):
        _, c = model.run_cache(torch.as_tensor(x[i:i + batch], device=device))
        for l in range(len(model.blocks)):
            ins[l].append(c["mlp_in"][l].reshape(-1, model.cfg.d_model))
            outs[l].append(c["mlp_out"][l].reshape(-1, model.cfg.d_model))
    return [torch.cat(a) for a in ins], [torch.cat(a) for a in outs]


def train_transcoder(tc: Transcoder, ins, outs, steps=3000, batch=1024, lr=1e-3, seed=0,
                     device="cpu", verbose=False):
    torch.manual_seed(seed)
    tc = tc.to(device)
    ins = [a.to(device) for a in ins]
    outs = [a.to(device) for a in outs]
    var = [o.var(0).sum().clamp_min(1e-8) for o in outs]
    with torch.no_grad():
        for l in range(tc.cfg.n_layers):
            tc.b_dec[l] = outs[l].mean(0)
    opt = torch.optim.Adam(tc.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = ins[0].shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g).to(device)
        rec, _ = tc([a[idx] for a in ins])
        loss = sum(((r - o[idx]) ** 2).sum(-1).mean() / v for r, o, v in zip(rec, outs, var)) / tc.cfg.n_layers
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % max(steps // 5, 1) == 0 or step == steps - 1:
            hist.append({"step": step, "nmse": float(loss.detach()), "sec": time.time() - t0})
            if verbose:
                print(hist[-1])
    return tc.eval().cpu(), hist


@torch.no_grad()
def fidelity(tc: Transcoder, model, x: np.ndarray, y: np.ndarray) -> dict:
    """Per-layer reconstruction + end-to-end replacement metrics."""
    ins, outs = collect_mlp_io(model, x)
    rec, acts = tc(ins)
    res = {}
    for l in range(tc.cfg.n_layers):
        err = ((rec[l] - outs[l]) ** 2).sum(-1)
        res[f"rel_mse_L{l}"] = float(err.mean() / outs[l].var(0).sum())
        res[f"rel_l2_L{l}"] = float((rec[l] - outs[l]).norm() / outs[l].norm())
        res[f"dead_frac_L{l}"] = float(((acts[l] > 0).float().mean(0) == 0).float().mean())
        res[f"l0_L{l}"] = float((acts[l] > 0).float().sum(-1).mean())
    res["rel_mse_mean"] = float(np.mean([res[f"rel_mse_L{l}"] for l in range(tc.cfg.n_layers)]))
    xt = torch.as_tensor(x)
    clean = model(xt)
    repl = model(xt, iv={"mlp_fn": tc.as_mlp_fns()})
    pc, pr = torch.sigmoid(clean), torch.sigmoid(repl)
    kl = pc * (torch.log(pc + 1e-9) - torch.log(pr + 1e-9)) + (1 - pc) * (torch.log(1 - pc + 1e-9) - torch.log(1 - pr + 1e-9))
    res["e2e_bit_acc"] = float(((repl > 0).long().numpy() == y).mean())
    res["e2e_agree_with_model"] = float(((repl > 0) == (clean > 0)).float().mean())
    res["e2e_kl"] = float(kl.mean())
    res["n_params"] = tc.n_params()
    return res


def save_tc(tc: Transcoder, path, meta):
    torch.save({"cfg": tc.cfg.to_dict(), "state": tc.state_dict(), "meta": meta}, path)


def load_tc(path) -> Transcoder:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tc = Transcoder(TCConfig(**ck["cfg"]))
    tc.load_state_dict(ck["state"])
    return tc.eval()
