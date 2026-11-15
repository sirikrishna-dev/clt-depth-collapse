"""Small pre-LN transformer with explicit per-head weights and intervention hooks.

Residual-stream points:
    P_0       = token embedding + positional embedding
    Q_l       = P_l + Attn_l(LN1_l(P_l))
    P_{l+1}   = Q_l + MLP_l(LN2_l(Q_l))
    logit_t   = w_U . LNf(P_L)_t + b_U          (one logit per bit position)

Hop k (k = 1..D-1) of the SIIT high-level model is aligned to the residual
point P_k (after block k-1) at coordinates site_dims(k).  The last hop (y) is
computed by the final block and read by the unembedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import N_POS


@dataclass
class ModelConfig:
    D: int                 # task depth (number of Boolean hops)
    n_layers: int          # transformer blocks (= D for the main zoo)
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 128
    site_width: int = 1    # coordinates per intermediate site (Round 1 used k=1)
    n_pos: int = N_POS
    reserve_sites: bool = True  # only MLP_{k-1} may write hop k's site coordinates

    @property
    def n_inputs(self) -> int:
        return self.D + 1

    def site_dims(self, hop: int) -> list[int]:
        w = self.site_width
        return list(range((hop - 1) * w, hop * w))

    def to_dict(self):
        return asdict(self)


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x, return_scale=False):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        sigma = torch.sqrt((xc ** 2).mean(-1, keepdim=True) + self.eps)
        out = xc / sigma * self.g + self.b
        return (out, sigma) if return_scale else out


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.WQ = nn.Parameter(torch.randn(h, d, self.dh) / math.sqrt(d))
        self.WK = nn.Parameter(torch.randn(h, d, self.dh) / math.sqrt(d))
        self.WV = nn.Parameter(torch.randn(h, d, self.dh) / math.sqrt(d))
        self.WO = nn.Parameter(torch.randn(h, self.dh, d) / math.sqrt(d))
        self.bQ = nn.Parameter(torch.zeros(h, self.dh))
        self.bK = nn.Parameter(torch.zeros(h, self.dh))
        self.bV = nn.Parameter(torch.zeros(h, self.dh))
        self.bO = nn.Parameter(torch.zeros(d))

    def forward(self, x):  # x: (B,T,d), already layer-normed; bidirectional attention
        q = torch.einsum("btd,hde->bhte", x, self.WQ) + self.bQ[None, :, None]
        k = torch.einsum("btd,hde->bhte", x, self.WK) + self.bK[None, :, None]
        v = torch.einsum("btd,hde->bhte", x, self.WV) + self.bV[None, :, None]
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)  # (B,h,T,T)
        z = att @ v
        out = torch.einsum("bhte,hed->btd", z, self.WO) + self.bO
        return out, att


class MLP(nn.Module):
    def __init__(self, d, dm):
        super().__init__()
        self.fc1 = nn.Linear(d, dm)
        self.fc2 = nn.Linear(dm, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_mlp)


class BoolTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.randn(2 * cfg.n_inputs, cfg.d_model) * 0.5)
        self.pos = nn.Parameter(torch.randn(cfg.n_pos, cfg.d_model) * 0.1)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.lnf = LayerNorm(cfg.d_model)
        self.w_U = nn.Parameter(torch.randn(cfg.d_model) / math.sqrt(cfg.d_model))
        self.b_U = nn.Parameter(torch.zeros(()))

    # ---------------------------------------------------------------- embedding
    def onehot(self, x):  # x: (B, D+1, T) bits -> (B, T, 2(D+1))
        B, V, T = x.shape
        idx = (2 * torch.arange(V, device=x.device)[None, :, None] + x).permute(0, 2, 1)  # (B,T,V)
        oh = torch.zeros(B, T, 2 * V, device=x.device)
        oh.scatter_(2, idx, 1.0)
        return oh

    def input_writes(self, x):
        """Per-variable embedding writes: (B, V, T, d)."""
        B, V, T = x.shape
        rows = 2 * torch.arange(V, device=x.device)[None, :, None] + x  # (B,V,T)
        return self.W_E[rows]

    def embed(self, x):
        return self.onehot(x) @ self.W_E + self.pos

    # ------------------------------------------------------------------ forward
    def forward(self, x, iv: dict | None = None, cache: dict | None = None):
        """x: (B, D+1, T) int bits.  Returns logits (B, T).

        iv (interventions), all optional:
          'resid_set':  {p: (dims, values (B,T,len(dims)))}  overwrite coords at P_p
          'resid_add':  {p: delta (B,T,d)}                   add to P_p
          'proj_set':   {p: (u (d,), values (B,T))}          set projection on unit u at P_p
          'attn_out':   {l: tensor (B,T,d)}                  replace attention output of layer l
          'mlp_out_set':{l: (dims, values (B,T,len))}        overwrite coords of MLP_l output
          'mlp_fn':     {l: f(mlp_in (B,T,d), l) -> (B,T,d)} replace MLP_l (replacement models)
        cache: dict filled with P, Q, mlp_in, mlp_out, attn patterns, LN scales.
        """
        iv = iv or {}
        h = self.embed(x)
        h = self._at_point(h, 0, iv)
        if cache is not None:
            cache["P"] = [h]
            for key in ("Q", "mlp_in", "mlp_out", "attn_out", "att", "s1", "s2"):
                cache[key] = []
        for l, blk in enumerate(self.blocks):
            a_in, s1 = blk.ln1(h, return_scale=True)
            a_out, att = blk.attn(a_in)
            if l in iv.get("attn_out", {}):
                a_out = iv["attn_out"][l]
            q = h + a_out
            m_in, s2 = blk.ln2(q, return_scale=True)
            if l in iv.get("mlp_fn", {}):
                m_out = iv["mlp_fn"][l](m_in, l)
            else:
                m_out = blk.mlp(m_in)
            if l in iv.get("mlp_out_set", {}):
                dims, vals = iv["mlp_out_set"][l]
                m_out = m_out.clone()
                m_out[..., dims] = vals
            h = q + m_out
            h = self._at_point(h, l + 1, iv)
            if cache is not None:
                cache["Q"].append(q); cache["mlp_in"].append(m_in); cache["mlp_out"].append(m_out)
                cache["attn_out"].append(a_out); cache["att"].append(att)
                cache["s1"].append(s1); cache["s2"].append(s2); cache["P"].append(h)
        f, sf = self.lnf(h, return_scale=True)
        if cache is not None:
            cache["sf"] = sf
        return f @ self.w_U + self.b_U

    @staticmethod
    def _at_point(h, p, iv):
        if p in iv.get("resid_set", {}):
            dims, vals = iv["resid_set"][p]
            h = h.clone()
            h[..., dims] = vals
        if p in iv.get("resid_add", {}):
            h = h + iv["resid_add"][p]
        if p in iv.get("proj_set", {}):
            u, vals = iv["proj_set"][p]
            cur = h @ u
            h = h + (vals - cur)[..., None] * u
        return h

    @torch.no_grad()
    def run_cache(self, x, iv=None):
        cache = {}
        logits = self.forward(x, iv=iv, cache=cache)
        return logits, cache


@torch.no_grad()
def project_reserved_sites(model: "BoolTransformer"):
    """Enforce the site reservation by zeroing every write into a site coordinate except
    MLP_{k-1} -> site k.  Called at init and after every optimiser step, so the forward
    pass and the attribution-graph code see the same (projected) weights."""
    cfg = model.cfg
    if not cfg.reserve_sites or cfg.D < 2:
        return
    all_sites = [i for k in range(1, cfg.D) for i in cfg.site_dims(k)]
    model.W_E[:, all_sites] = 0
    model.pos[:, all_sites] = 0
    for l, blk in enumerate(model.blocks):
        blk.attn.WO[..., all_sites] = 0
        blk.attn.bO[all_sites] = 0
        own = set(cfg.site_dims(l + 1)) if l + 1 < cfg.D else set()
        other = [i for i in all_sites if i not in own]
        blk.mlp.fc2.weight[other, :] = 0
        blk.mlp.fc2.bias[other] = 0


def save_model(model: BoolTransformer, path, meta: dict):
    torch.save({"cfg": model.cfg.to_dict(), "state": model.state_dict(), "meta": meta}, path)


def load_model(path, map_location="cpu") -> tuple[BoolTransformer, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = BoolTransformer(ModelConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck.get("meta", {})
