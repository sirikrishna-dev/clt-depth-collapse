"""Generic transformer (GPT-2 / TransformerLens-style weights) with autograd-based
attribution graphs, for the real-data tiers (InterpBench models, pretrained LMs).

The Boolean models use explicit (T*d)^2 matrices (graphs.py).  That does not scale to
d=768, so here edges are computed by back-propagating through the *local replacement
model* (frozen attention patterns + frozen LayerNorm scales + transcoder features +
error terms) exactly as in Circuit Tracing:

    edge(s -> t) = write_s . d(preact_t)/d(residual at s's write point)

with all other features held constant (direct effects).  The local model is affine in
the writes, so incoming edges + constant reproduce each pre-activation exactly; this is
unit-tested on a random small model in tests/test_generic_tf.py.

Status: code paths are tested here on random weights.  Runs on real GPT-2 /
InterpBench checkpoints must be done on Colab (needs internet) -- see notebooks/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..graphs import Graph


@dataclass
class TFConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    act: str = "gelu_new"      # gelu_new | gelu | relu
    causal: bool = True
    use_ln: bool = True


def _act(x, kind):
    if kind == "relu":
        return torch.relu(x)
    if kind == "gelu":
        return F.gelu(x)
    return F.gelu(x, approximate="tanh")


class GenericTransformer:
    """Weights in TransformerLens layout:
    W_E (V,d) W_pos (n_ctx,d); per layer: ln1_w, ln1_b, W_Q/W_K/W_V (H,d,dh), b_Q/b_K/b_V (H,dh),
    W_O (H,dh,d), b_O (d), ln2_w, ln2_b, W_in (d,m), b_in (m), W_out (m,d), b_out (d);
    lnf_w, lnf_b, W_U (d,V), b_U (V)."""

    def __init__(self, cfg: TFConfig, w: dict, device="cpu"):
        self.cfg = cfg
        self.w = {k: v.to(device) for k, v in w.items()}
        self.device = device

    # ------------------------------------------------------------------ pieces
    def _ln(self, x, wkey, bkey, frozen_scale=None):
        if not self.cfg.use_ln:
            return x, torch.ones_like(x[..., :1])
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        s = torch.sqrt((xc ** 2).mean(-1, keepdim=True) + 1e-5) if frozen_scale is None else frozen_scale
        return xc / s * self.w[wkey] + self.w[bkey], s

    def _attn(self, l, x, frozen_pattern=None):
        w = self.w
        q = torch.einsum("td,hde->hte", x, w[f"{l}.W_Q"]) + w[f"{l}.b_Q"][:, None]
        k = torch.einsum("td,hde->hte", x, w[f"{l}.W_K"]) + w[f"{l}.b_K"][:, None]
        v = torch.einsum("td,hde->hte", x, w[f"{l}.W_V"]) + w[f"{l}.b_V"][:, None]
        if frozen_pattern is None:
            sc = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            if self.cfg.causal:
                T = x.shape[0]
                sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), -1e9)
            A = torch.softmax(sc, -1)
        else:
            A = frozen_pattern
        z = A @ v
        return torch.einsum("hte,hed->td", z, w[f"{l}.W_O"]) + w[f"{l}.b_O"], A

    def mlp(self, l, x):
        return _act(x @ self.w[f"{l}.W_in"] + self.w[f"{l}.b_in"], self.cfg.act) @ self.w[f"{l}.W_out"] + self.w[f"{l}.b_out"]

    # ----------------------------------------------------------------- forward
    @torch.no_grad()
    def run(self, tokens: torch.Tensor):
        """tokens (T,) -> logits (T,V), cache."""
        T = tokens.shape[0]
        r = self.w["W_E"][tokens] + self.w["W_pos"][:T]
        c = {"A": [], "s1": [], "s2": [], "mlp_in": [], "mlp_out": []}
        for l in range(self.cfg.n_layers):
            a_in, s1 = self._ln(r, f"{l}.ln1_w", f"{l}.ln1_b")
            ao, A = self._attn(l, a_in)
            r = r + ao
            m_in, s2 = self._ln(r, f"{l}.ln2_w", f"{l}.ln2_b")
            mo = self.mlp(l, m_in)
            r = r + mo
            for k_, v_ in (("A", A), ("s1", s1), ("s2", s2), ("mlp_in", m_in), ("mlp_out", mo)):
                c[k_].append(v_)
        f, sf = self._ln(r, "lnf_w", "lnf_b")
        c["sf"] = sf
        return f @ self.w["W_U"] + self.w["b_U"], c

    @torch.no_grad()
    def mlp_io(self, tokens_batch: torch.Tensor):
        """tokens_batch (B,T) -> lists of (B*T,d) MLP inputs and outputs (for transcoder training)."""
        ins = [[] for _ in range(self.cfg.n_layers)]
        outs = [[] for _ in range(self.cfg.n_layers)]
        for toks in tokens_batch:
            _, c = self.run(toks)
            for l in range(self.cfg.n_layers):
                ins[l].append(c["mlp_in"][l]); outs[l].append(c["mlp_out"][l])
        return [torch.cat(a) for a in ins], [torch.cat(a) for a in outs]

    # ------------------------------------------------------ local replacement
    def _local_forward(self, c, tc, emb_leaf, out_leaves, acts_const, errs, delta_leaves=None, masks=None):
        """Affine local replacement model.  Returns (feature preacts per layer, final resid).
        emb_leaf (T,d): input writes at P_0.  out_leaves[l] (T,d): extra write at P_{l+1}.
        acts_const[l]: frozen activations (direct-effect mode) unless delta_leaves given,
        in which case activations are live-linear: a = mask*preact + delta (total-effect mode)."""
        L = self.cfg.n_layers
        T = emb_leaf.shape[0]
        r = emb_leaf + self.w["W_pos"][:T]
        pre_all, acts = [], []
        pending = {}
        for l in range(L):
            a_in, _ = self._ln(r, f"{l}.ln1_w", f"{l}.ln1_b", frozen_scale=c["s1"][l])
            ao, _ = self._attn(l, a_in, frozen_pattern=c["A"][l])
            r = r + ao
            m_in, _ = self._ln(r, f"{l}.ln2_w", f"{l}.ln2_b", frozen_scale=c["s2"][l])
            pre = tc.preact(m_in, l)
            pre_all.append(pre)
            a = masks[l] * pre + delta_leaves[l] if delta_leaves is not None else acts_const[l]
            for t in tc.dsts(l):
                pending[t] = pending.get(t, 0) + a @ tc.dec(l, t)
            r = r + tc.b_dec[l] + pending.pop(l) + errs[l] + out_leaves[l]
        return pre_all, r

    @torch.no_grad()
    def run_replacement(self, tokens, tc, errs=None, edit=None):
        """Full replacement model (attention recomputed) with optional error terms and
        activation edits edit(l, acts) -> acts.  Returns logits (T,V)."""
        T = tokens.shape[0]
        r = self.w["W_E"][tokens] + self.w["W_pos"][:T]
        pending = {}
        for l in range(self.cfg.n_layers):
            a_in, _ = self._ln(r, f"{l}.ln1_w", f"{l}.ln1_b")
            ao, _ = self._attn(l, a_in)
            r = r + ao
            m_in, _ = self._ln(r, f"{l}.ln2_w", f"{l}.ln2_b")
            a = tc.encode(m_in, l)
            if edit is not None:
                a = edit(l, a)
            for t in tc.dsts(l):
                pending[t] = pending.get(t, 0) + a @ tc.dec(l, t)
            r = r + tc.b_dec[l] + pending.pop(l) + (0 if errs is None else errs[l])
        f, _ = self._ln(r, "lnf_w", "lnf_b")
        return f @ self.w["W_U"] + self.w["b_U"]

    def build_graph(self, tokens: torch.Tensor, tc, target_token: int, max_features: int = 300,
                    keep_all: bool = False) -> tuple[Graph, dict]:
        """Attribution graph for the logit of `target_token` at the last position."""
        cfg = self.cfg
        L, d = cfg.n_layers, cfg.d_model
        T = tokens.shape[0]
        logits, c = self.run(tokens)
        with torch.no_grad():
            acts = [tc.encode(c["mlp_in"][l], l) for l in range(L)]
            rec = tc.decode_all(acts)
            errs = [c["mlp_out"][l] - rec[l] for l in range(L)]
        emb = self.w["W_E"][tokens].clone()
        masks = [(a > 0).float() for a in acts]

        def logit_fn(r):
            f, _ = self._ln(r, "lnf_w", "lnf_b", frozen_scale=c["sf"])
            return f[-1] @ self.w["W_U"][:, target_token] + self.w["b_U"][target_token]

        # ---- 1. rank features by total effect on the logit (live-linear features)
        deltas = [torch.zeros_like(a, requires_grad=True) for a in acts]
        zero_out = [torch.zeros(T, d, device=self.device) for _ in range(L)]
        _, r = self._local_forward(c, tc, emb, zero_out, None, errs, delta_leaves=deltas, masks=masks)
        grads = torch.autograd.grad(logit_fn(r), deltas)
        score = [(acts[l] * grads[l]).abs() for l in range(L)]
        cand = [(float(score[l][t, f]), l, t, f) for l in range(L) for t, f in zip(*torch.nonzero(acts[l] > 0, as_tuple=True))]
        cand.sort(reverse=True)
        tot_score = sum(s for s, *_ in cand) + 1e-12
        chosen = cand if keep_all else cand[:max_features]
        coverage = sum(s for s, *_ in chosen) / tot_score
        chosen = sorted([(l, int(t), int(f)) for _, l, t, f in chosen])

        # ---- 2. nodes
        types, layers, poss, feats, actv = [], [], [], [], []
        for t in range(T):
            types.append("input"); layers.append(-1); poss.append(t); feats.append(t); actv.append(1.0)
        fidx = {}
        for (l, t, f) in chosen:
            fidx[(l, t, f)] = len(types)
            types.append("feature"); layers.append(l); poss.append(t); feats.append(f); actv.append(float(acts[l][t, f]))
        for l in range(L):
            for t in range(T):
                types.append("error"); layers.append(l); poss.append(t); feats.append(-1); actv.append(1.0)
        types.append("logit"); layers.append(L); poss.append(T - 1); feats.append(-1); actv.append(1.0)
        N = len(types)
        A = np.zeros((N, N))
        tpre = np.full(N, np.nan); tconst = np.full(N, np.nan)

        # ---- 3. direct effects: one backward per target (features frozen)
        emb_leaf = emb.clone().requires_grad_(True)
        out_leaves = [torch.zeros(T, d, device=self.device, requires_grad=True) for _ in range(L)]
        pre_all, r = self._local_forward(c, tc, emb_leaf, out_leaves, [a.detach() for a in acts], errs)
        targets = [(fidx[k], pre_all[k[0]][k[1], k[2]]) for k in chosen] + [(N - 1, logit_fn(r))]
        leaves = [emb_leaf] + out_leaves
        # Vectorised edge computation (identical maths to the per-element loop, one GPU->CPU copy per target).
        n_in, n_feat = T, len(fidx)
        by_layer = {}
        for (l, t, f), si in fidx.items():
            by_layer.setdefault(l, []).append((t, f, si))
        by_layer = {l: (torch.tensor([x[0] for x in v], device=self.device),
                        torch.tensor([x[1] for x in v], device=self.device),
                        np.array([x[2] for x in v])) for l, v in by_layer.items()}
        err_stack = torch.stack(errs)                                   # (L, T, d)
        for ti, val in targets:
            gs = torch.autograd.grad(val, leaves, retain_graph=True, allow_unused=True)
            gs = [torch.zeros(T, d, device=self.device) if g is None else g for g in gs]
            # inputs
            A[ti, :n_in] = (emb * gs[0]).sum(-1).detach().cpu().numpy()
            # features: write a * W_dec[s->dst] at point dst+1  (gs[dst+1])
            for l, (tt, ff, cols) in by_layer.items():
                tot = torch.zeros(len(cols), device=self.device)
                for dst in tc.dsts(l):
                    tot = tot + (tc.dec(l, dst)[ff] * gs[dst + 1][tt]).sum(-1)
                A[ti, cols] = (acts[l][tt, ff] * tot).detach().cpu().numpy()
            # error nodes
            ge = torch.stack(gs[1:L + 1])                                # (L, T, d)
            A[ti, n_in + n_feat:n_in + n_feat + L * T] = (err_stack * ge).sum(-1).reshape(-1).detach().cpu().numpy()
            tpre[ti] = float(val.detach())
        # constant term: local model with every node write removed (exactly affine)
        with torch.no_grad():
            pre0, r0 = self._local_forward(c, tc, torch.zeros_like(emb), [torch.zeros(T, d, device=self.device)] * L,
                                           [torch.zeros_like(a) for a in acts], [torch.zeros_like(e) for e in errs])
        for (l, t, f), si in fidx.items():
            tconst[si] = float(pre0[l][t, f])
        tconst[N - 1] = float(logit_fn(r0))
        g = Graph(A=A, node_type=np.array(types), layer=np.array(layers), pos=np.array(poss), feat=np.array(feats),
                  act=np.array(actv), target_preact=tpre, target_const=tconst, n_layers=L,
                  meta={"coverage": coverage, "n_candidates": len(cand), "target_token": target_token})
        return g, {"coverage": coverage, "logit": float(logits[-1, target_token])}


# ---------------------------------------------------------------- loaders
def from_hf_gpt2(name="gpt2", device="cpu"):
    """Load GPT-2 family weights from Hugging Face `transformers` into TL layout."""
    from transformers import GPT2LMHeadModel
    return from_hf_gpt2_model(GPT2LMHeadModel.from_pretrained(name), device)


def from_hf_gpt2_model(hf, device="cpu"):
    """Convert an already-loaded HF GPT2LMHeadModel (tested offline against HF's own forward)."""
    sd = hf.state_dict()
    hc = hf.config
    d, H = hc.n_embd, hc.n_head
    dh = d // H
    w = {"W_E": sd["transformer.wte.weight"], "W_pos": sd["transformer.wpe.weight"],
         "lnf_w": sd["transformer.ln_f.weight"], "lnf_b": sd["transformer.ln_f.bias"],
         "W_U": sd["transformer.wte.weight"].T.contiguous(), "b_U": torch.zeros(hc.vocab_size)}
    for l in range(hc.n_layer):
        p = f"transformer.h.{l}."
        Wqkv, bqkv = sd[p + "attn.c_attn.weight"], sd[p + "attn.c_attn.bias"]
        for j, nm in enumerate("QKV"):
            Wj = Wqkv[:, j * d:(j + 1) * d]
            w[f"{l}.W_{nm}"] = Wj.reshape(d, H, dh).permute(1, 0, 2).contiguous()
            w[f"{l}.b_{nm}"] = bqkv[j * d:(j + 1) * d].reshape(H, dh)
        w[f"{l}.W_O"] = sd[p + "attn.c_proj.weight"].reshape(H, dh, d)
        w[f"{l}.b_O"] = sd[p + "attn.c_proj.bias"]
        w[f"{l}.ln1_w"], w[f"{l}.ln1_b"] = sd[p + "ln_1.weight"], sd[p + "ln_1.bias"]
        w[f"{l}.ln2_w"], w[f"{l}.ln2_b"] = sd[p + "ln_2.weight"], sd[p + "ln_2.bias"]
        w[f"{l}.W_in"], w[f"{l}.b_in"] = sd[p + "mlp.c_fc.weight"], sd[p + "mlp.c_fc.bias"]
        w[f"{l}.W_out"], w[f"{l}.b_out"] = sd[p + "mlp.c_proj.weight"], sd[p + "mlp.c_proj.bias"]
    cfg = TFConfig(hc.n_layer, d, H, 4 * d, hc.vocab_size, hc.n_positions, "gelu_new", True, True)
    return GenericTransformer(cfg, {k: v.float() for k, v in w.items()}, device)


def from_transformerlens_state_dict(cfg_dict: dict, sd: dict, device="cpu"):
    """Load a TransformerLens HookedTransformer state dict (e.g. InterpBench `ll_model`).
    VERIFY key names against the downloaded checkpoint (print sd.keys())."""
    L, d = cfg_dict["n_layers"], cfg_dict["d_model"]
    H = cfg_dict["n_heads"]
    dh = cfg_dict.get("d_head", d // H)
    d_mlp = cfg_dict.get("d_mlp", 4 * d)
    use_ln = cfg_dict.get("normalization_type", "LN") not in (None, "None")
    w = {"W_E": sd["embed.W_E"], "W_pos": sd["pos_embed.W_pos"], "W_U": sd["unembed.W_U"], "b_U": sd["unembed.b_U"]}
    one, zero = torch.ones(d), torch.zeros(d)
    w["lnf_w"] = sd.get("ln_final.w", one); w["lnf_b"] = sd.get("ln_final.b", zero)
    for l in range(L):
        p = f"blocks.{l}."
        for nm in "QKV":
            w[f"{l}.W_{nm}"] = sd[p + f"attn.W_{nm}"]
            w[f"{l}.b_{nm}"] = sd.get(p + f"attn.b_{nm}", torch.zeros(H, dh))
        w[f"{l}.W_O"] = sd[p + "attn.W_O"]; w[f"{l}.b_O"] = sd.get(p + "attn.b_O", zero)
        w[f"{l}.ln1_w"] = sd.get(p + "ln1.w", one); w[f"{l}.ln1_b"] = sd.get(p + "ln1.b", zero)
        w[f"{l}.ln2_w"] = sd.get(p + "ln2.w", one); w[f"{l}.ln2_b"] = sd.get(p + "ln2.b", zero)
        w[f"{l}.W_in"] = sd.get(p + "mlp.W_in", torch.zeros(d, d_mlp)); w[f"{l}.b_in"] = sd.get(p + "mlp.b_in", torch.zeros(d_mlp))
        w[f"{l}.W_out"] = sd.get(p + "mlp.W_out", torch.zeros(d_mlp, d)); w[f"{l}.b_out"] = sd.get(p + "mlp.b_out", zero)
    act = cfg_dict.get("act_fn", "gelu")
    cfg = TFConfig(L, d, H, d_mlp, w["W_E"].shape[0], w["W_pos"].shape[0],
                   "relu" if act == "relu" else ("gelu_new" if act == "gelu_new" else "gelu"),
                   cfg_dict.get("attention_dir", "causal") == "causal", use_ln)
    return GenericTransformer(cfg, {k: v.float() for k, v in w.items()}, device)


def random_small(n_layers=3, d=32, H=4, d_mlp=64, V=20, n_ctx=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    dh = d // H
    w = {"W_E": r(V, d), "W_pos": r(n_ctx, d), "lnf_w": 1 + r(d) * 0.1, "lnf_b": r(d) * 0.1, "W_U": r(d, V), "b_U": r(V) * 0.1}
    for l in range(n_layers):
        for nm in "QKV":
            w[f"{l}.W_{nm}"] = r(H, d, dh); w[f"{l}.b_{nm}"] = r(H, dh) * 0.1
        w[f"{l}.W_O"] = r(H, dh, d); w[f"{l}.b_O"] = r(d) * 0.1
        for nm in ("ln1", "ln2"):
            w[f"{l}.{nm}_w"] = 1 + r(d) * 0.1; w[f"{l}.{nm}_b"] = r(d) * 0.1
        w[f"{l}.W_in"] = r(d, d_mlp); w[f"{l}.b_in"] = r(d_mlp) * 0.1
        w[f"{l}.W_out"] = r(d_mlp, d); w[f"{l}.b_out"] = r(d) * 0.1
    return GenericTransformer(TFConfig(n_layers, d, H, d_mlp, V, n_ctx), w)
