"""Tier 3 (TA: "validate your methodology on real data"): two-hop factual recall in GPT-2.

    2-hop prompt:  "Fact: the capital of the state containing Dallas is"  ->  " Austin"
    1-hop control: "Fact: the capital of Texas is"                        ->  " Austin"

The 2-hop prompt needs an extra step (Dallas -> Texas) that the 1-hop prompt does not,
so we know the intermediate entity. The script runs PLT and CLT transcoders trained at an
identical budget (same features per layer, same TopK k, same steps, data, seed and lr) and
asks four validation questions:

  V1  Depth sensitivity: is D_feat(2-hop, from the city tokens) > D_feat(1-hop, from the
      state tokens)? Paired by state. A depth metric that cannot separate these on a real
      model is not measuring depth.
  V2  Causal specificity: zero-ablating the state-selective features (selected
      leave-one-out: never using the prompt being tested) lowers the answer logit much
      more than ablating the same number of random active features in the same layers.
  V3  Graph vs intervention: does the graph's claimed mediation through state features
      (IMF) track the measured ablation drop across prompts? (Spearman)
  V4  CLT vs PLT on the same prompts: D_feat, skip mass, IMF, ablation drop, paired tests.

Plus fidelity: reconstruction NMSE on held-out WikiText, and whether the replacement
model (no error terms) keeps the original model's answer.

Run on Colab (GPU, internet):
    python -m cdc.realdata.lm_twohop --model gpt2 --out results/lm_twohop
Every number in the paper's Sec. 7 must come from the CSVs this writes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from ..depth_metrics import graph_metrics
from ..stats import bootstrap_ci, paired_signflip_p
from ..transcoders import make_transcoder

# (city, state, capital): the city is never the capital itself. Several cities per state
# allow leave-one-out selection of state-selective features.
TRIPLES = [
    ("Dallas", "Texas", "Austin"), ("Houston", "Texas", "Austin"), ("San Antonio", "Texas", "Austin"),
    ("El Paso", "Texas", "Austin"), ("Fort Worth", "Texas", "Austin"),
    ("Los Angeles", "California", "Sacramento"), ("San Francisco", "California", "Sacramento"),
    ("San Diego", "California", "Sacramento"), ("Fresno", "California", "Sacramento"), ("Oakland", "California", "Sacramento"),
    ("Miami", "Florida", "Tallahassee"), ("Orlando", "Florida", "Tallahassee"), ("Tampa", "Florida", "Tallahassee"),
    ("Jacksonville", "Florida", "Tallahassee"),
    ("Buffalo", "New York", "Albany"), ("Rochester", "New York", "Albany"), ("Brooklyn", "New York", "Albany"),
    ("Chicago", "Illinois", "Springfield"), ("Peoria", "Illinois", "Springfield"),
    ("Philadelphia", "Pennsylvania", "Harrisburg"), ("Pittsburgh", "Pennsylvania", "Harrisburg"),
    ("Seattle", "Washington", "Olympia"), ("Spokane", "Washington", "Olympia"), ("Tacoma", "Washington", "Olympia"),
    ("Savannah", "Georgia", "Atlanta"), ("Augusta", "Georgia", "Atlanta"),
    ("Cleveland", "Ohio", "Columbus"), ("Cincinnati", "Ohio", "Columbus"), ("Toledo", "Ohio", "Columbus"),
    ("Detroit", "Michigan", "Lansing"), ("Grand Rapids", "Michigan", "Lansing"),
    ("Las Vegas", "Nevada", "Carson City"), ("Reno", "Nevada", "Carson City"),
    ("Portland", "Oregon", "Salem"), ("Eugene", "Oregon", "Salem"),
    ("St. Louis", "Missouri", "Jefferson City"), ("Kansas City", "Missouri", "Jefferson City"),
    ("New Orleans", "Louisiana", "Baton Rouge"), ("Shreveport", "Louisiana", "Baton Rouge"),
    ("Louisville", "Kentucky", "Frankfort"), ("Lexington", "Kentucky", "Frankfort"),
    ("Baltimore", "Maryland", "Annapolis"), ("Bethesda", "Maryland", "Annapolis"),
    ("Tucson", "Arizona", "Phoenix"), ("Flagstaff", "Arizona", "Phoenix"),
    ("Memphis", "Tennessee", "Nashville"), ("Knoxville", "Tennessee", "Nashville"),
    ("Boulder", "Colorado", "Denver"), ("Colorado Springs", "Colorado", "Denver"),
    ("Milwaukee", "Wisconsin", "Madison"), ("Green Bay", "Wisconsin", "Madison"),
]
TEMPLATE_2HOP = "Fact: the capital of the state containing {city} is"
TEMPLATE_1HOP = "Fact: the capital of {state} is"
N_RANDOM_DRAWS = 5


# ------------------------------------------------------------------ transcoder training
def wikitext_batches(tok, batch, seq_len, split="train"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    buf = []
    for ex in ds:
        buf.extend(tok(ex["text"])["input_ids"])
        while len(buf) >= batch * seq_len:
            chunk = torch.tensor(buf[:batch * seq_len]).view(batch, seq_len)
            del buf[:batch * seq_len]
            yield chunk


@torch.no_grad()
def mlp_io_batch(gm, x):
    """x (B,T) token ids -> per-layer (B*T,d) MLP inputs and outputs from the generic engine."""
    L = gm.cfg.n_layers
    ins, outs = [[] for _ in range(L)], [[] for _ in range(L)]
    for row in x:
        _, c = gm.run(row)
        for l in range(L):
            ins[l].append(c["mlp_in"][l]); outs[l].append(c["mlp_out"][l])
    return [torch.cat(a) for a in ins], [torch.cat(a) for a in outs]


class HFMLPIO:
    """Batched MLP input/output capture from a Hugging Face GPT-2 (fast path for training).
    mlp_in = ln_2 output, mlp_out = MLP output: the same points the generic engine uses."""

    def __init__(self, hf):
        self.hf, self.cap = hf, {}
        for l, blk in enumerate(hf.transformer.h):
            blk.mlp.register_forward_pre_hook(lambda m, a, l=l: self.cap.__setitem__(("in", l), a[0].detach()))
            blk.mlp.register_forward_hook(lambda m, a, o, l=l: self.cap.__setitem__(("out", l), o.detach()))
        self.L, self.d = hf.config.n_layer, hf.config.n_embd

    @torch.no_grad()
    def __call__(self, x):
        self.hf(x)
        return ([self.cap[("in", l)].reshape(-1, self.d).float() for l in range(self.L)],
                [self.cap[("out", l)].reshape(-1, self.d).float() for l in range(self.L)])


def train_lm_transcoder(gm, batches, kind, n_features, k, steps, lr=1e-3, seed=0, log_every=200, io_fn=None):
    """Identical recipe for PLT and CLT: same data stream (re-created per kind), same seed, lr, steps.
    io_fn(x) -> (mlp_ins, mlp_outs); defaults to the (slower, per-sequence) generic engine."""
    torch.manual_seed(seed)
    io_fn = io_fn or (lambda x: mlp_io_batch(gm, x))
    L, d = gm.cfg.n_layers, gm.cfg.d_model
    tc = make_transcoder(kind, L, d, n_features, k).to(gm.device)
    opt = torch.optim.Adam(tc.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    var, hist = None, []
    for step in range(steps):
        x = next(batches).to(gm.device)
        ins, outs = io_fn(x)
        if var is None:
            var = [o.var(0).sum().clamp_min(1e-6) for o in outs]
            with torch.no_grad():
                for l in range(L):
                    tc.b_dec[l] = outs[l].mean(0)
        rec, _ = tc(ins)
        loss = sum(((r - o) ** 2).sum(-1).mean() / v for r, o, v in zip(rec, outs, var)) / L
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "train_nmse": loss.item()})
            print(kind, hist[-1], flush=True)
    for p in tc.parameters():
        p.requires_grad_(False)
    return tc.eval(), hist


@torch.no_grad()
def heldout_nmse(gm, tc, batches, n_batches=4, io_fn=None):
    io_fn = io_fn or (lambda x: mlp_io_batch(gm, x))
    L = gm.cfg.n_layers
    num, den = np.zeros(L), np.zeros(L)
    for _ in range(n_batches):
        ins, outs = io_fn(next(batches).to(gm.device))
        rec, _ = tc(ins)
        for l in range(L):
            num[l] += float(((rec[l] - outs[l]) ** 2).sum())
            den[l] += float(((outs[l] - outs[l].mean(0)) ** 2).sum())
    return (num / np.maximum(den, 1e-12)).tolist()


# ------------------------------------------------------------------ prompts
def span_positions(tok, prompt, span):
    """Token positions of `span` inside `prompt`.

    GPT-2 BPE attaches the leading space to the word (" Dallas"), so the prefix must be
    tokenised WITHOUT its trailing space; tokenising "... containing " would add a lone
    space token and shift the start by one (for a one-token city, the span would be empty).
    """
    pre = prompt.split(span)[0]
    n0 = len(tok(pre.rstrip(" "))["input_ids"]) if pre.strip() else 0
    n1 = len(tok(pre + span)["input_ids"])
    if n1 <= n0:
        raise ValueError(f"empty root span for {span!r} in {prompt!r}")
    return list(range(n0, n1))


def answer_rank(gm, ids, tgt):
    logits, _ = gm.run(ids)
    return int((logits[-1] > logits[-1, tgt]).sum()) + 1, logits


def build_prompts(gm, tok, triples, max_rank):
    rows = []
    for city, state, cap in triples:
        tgt = tok(" " + cap)["input_ids"][0]
        for hop, p, span in (("2hop", TEMPLATE_2HOP.format(city=city), city),
                             ("1hop", TEMPLATE_1HOP.format(state=state), state)):
            ids = torch.tensor(tok(p)["input_ids"], device=gm.device)
            rank, _ = answer_rank(gm, ids, tgt)
            rows.append(dict(hop=hop, city=city, state=state, capital=cap, prompt=p, root_span=span,
                             target_token=tgt, rank=rank))
    pr = pd.DataFrame(rows).drop_duplicates(subset=["hop", "prompt"]).reset_index(drop=True)
    ok1 = set(pr[(pr.hop == "1hop") & (pr["rank"] <= max_rank)].state)
    # keep a 2-hop prompt only if the model also knows the state's capital (1-hop), so a
    # failure cannot come from not knowing the capital at all
    pr["kept"] = ((pr["rank"] <= max_rank) & pr.state.isin(ok1))
    return pr


# ------------------------------------------------------------------ analysis
def state_selective_features(acts, states, i_exclude, state):
    """Features that fire much more on this state's OTHER 2-hop prompts than on other states' prompts.
    acts: (P, L, F) max-over-positions activations. Leave-one-out: prompt i_exclude is never used."""
    idx = np.arange(len(states))
    m_in = (states == state) & (idx != i_exclude)
    m_out = (states != state) & (idx != i_exclude)
    if m_in.sum() < 1 or m_out.sum() < 1:
        return set()
    mu_in, mu_out = acts[m_in].mean(0), acts[m_out].mean(0)
    fires = (acts[m_in] > 0).mean(0)
    sel = (mu_in > 2 * mu_out + 1e-3) & (fires >= 0.5)
    return set(zip(*map(lambda a: a.tolist(), np.nonzero(sel))))


@torch.no_grad()
def ablation_drop(gm, tc, ids, tgt, feats_by_layer):
    """Answer-logit drop in the replacement model (with error terms, so the unedited run is exact)
    when the given features are zeroed at every position."""
    _, c = gm.run(ids)
    acts = [tc.encode(c["mlp_in"][l], l) for l in range(gm.cfg.n_layers)]
    rec = tc.decode_all(acts)
    errs = [c["mlp_out"][l] - rec[l] for l in range(gm.cfg.n_layers)]

    def edit(l, a_):
        fs = feats_by_layer.get(l, [])
        if fs:
            a_ = a_.clone(); a_[:, fs] = 0
        return a_
    base = float(gm.run_replacement(ids, tc, errs)[-1, tgt])
    abl = float(gm.run_replacement(ids, tc, errs, edit)[-1, tgt])
    return base - abl, base


@torch.no_grad()
def replacement_keeps_answer(gm, tc, ids, tgt):
    """Replacement model WITHOUT error terms: is the answer still top-1 / what is its rank?"""
    orig_logits, _ = gm.run(ids)
    rep = gm.run_replacement(ids, tc, None)
    return dict(orig_top1_kept=int(rep[-1].argmax() == orig_logits[-1].argmax()),
                answer_rank_replacement=int((rep[-1] > rep[-1, tgt]).sum()) + 1)


def analyse(gm, tok, tcs, prompts, max_graph_features=300, seed=0):
    rng = np.random.default_rng(seed)
    L = gm.cfg.n_layers
    keep = prompts[prompts.kept].reset_index(drop=True)
    two = keep[keep.hop == "2hop"].reset_index(drop=True)
    rows = []
    for kind, tc in tcs.items():
        # max-over-positions activations of every kept 2-hop prompt (for leave-one-out selection)
        acts = []
        for _, r in two.iterrows():
            ids = torch.tensor(tok(r.prompt)["input_ids"], device=gm.device)
            _, c = gm.run(ids)
            acts.append(torch.stack([tc.encode(c["mlp_in"][l], l).max(0).values for l in range(L)]).cpu().numpy())
        acts = np.stack(acts) if acts else np.zeros((0, L, 1))
        states = two.state.values
        for _, r in keep.iterrows():
            t0 = time.time()
            ids = torch.tensor(tok(r.prompt)["input_ids"], device=gm.device)
            tgt = int(r.target_token)
            g, info = gm.build_graph(ids, tc, tgt, max_features=max_graph_features)
            roots = np.array(span_positions(tok, r.prompt, r.root_span))
            row = dict(kind=kind, hop=r.hop, city=r.city, state=r.state, n_tokens=len(ids),
                       coverage=info["coverage"], **replacement_keeps_answer(gm, tc, ids, tgt))
            if r.hop == "2hop":
                i = int(np.where((two.city == r.city).values)[0][0])
                sel = state_selective_features(acts, states, i, r.state)
                aligned = {1: np.array([j for j in np.where(g.node_type == "feature")[0]
                                        if (int(g.layer[j]), int(g.feat[j])) in sel], dtype=int)}
                met = graph_metrics(g, aligned, roots=roots)
                by_layer = {}
                for (l, f) in sel:
                    by_layer.setdefault(int(l), []).append(int(f))
                drop, base = ablation_drop(gm, tc, ids, tgt, by_layer)
                # random control: same number of features per layer, drawn from features active in
                # this prompt at that layer (excluding the selected ones)
                _, c = gm.run(ids)
                rnd = []
                for _ in range(N_RANDOM_DRAWS):
                    rb = {}
                    for l, fs in by_layer.items():
                        active = torch.nonzero(tc.encode(c["mlp_in"][l], l).max(0).values > 0).flatten().tolist()
                        pool = [f for f in active if f not in set(fs)]
                        if pool:
                            rb[l] = rng.choice(pool, size=min(len(fs), len(pool)), replace=False).tolist()
                    rnd.append(ablation_drop(gm, tc, ids, tgt, rb)[0])
                row.update(n_state_features=len(sel), n_state_features_in_graph=len(aligned[1]),
                           state_ablation_logit_drop=drop, random_ablation_logit_drop=float(np.mean(rnd)),
                           answer_logit=base)
            else:
                met = graph_metrics(g, None, roots=roots)
            row.update(met)
            row["seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            print(kind, r.hop, r.city if r.hop == "2hop" else r.state,
                  "D_feat=%.2f" % met["root_signed_D_feat"], "IMF=%s" % met.get("IMF_h1"), flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    """Validation tests V1-V4 with bootstrap CIs and exact/MC sign-flip p-values."""
    out = []

    def add(test, kind, a, b=None, label=""):
        a = np.asarray(a, float)
        if b is None:
            m, lo, hi = bootstrap_ci(a); p = paired_signflip_p(a, np.zeros_like(a))
        else:
            b = np.asarray(b, float); m, lo, hi = bootstrap_ci(a - b); p = paired_signflip_p(a, b)
        out.append(dict(test=test, kind=kind, quantity=label, n=int(np.sum(~np.isnan(a))), mean=m, ci_lo=lo, ci_hi=hi,
                        p_signflip=p))

    for kind, d in df.groupby("kind"):
        two, one = d[d.hop == "2hop"], d[d.hop == "1hop"]
        # V1: paired by state (2-hop averaged over that state's cities)
        s2 = two.groupby("state").root_signed_D_feat.mean()
        s1 = one.set_index("state").root_signed_D_feat
        common = s2.index.intersection(s1.index)
        add("V1_depth_sensitivity", kind, s2[common].values, s1[common].values, "D_feat(2hop) - D_feat(1hop), per state")
        s2a = two.groupby("state").root_D_feat.mean(); s1a = one.set_index("state").root_D_feat
        add("V1_depth_sensitivity", kind, s2a[common].values, s1a[common].values,
            "abs-normalised D_feat(2hop) - D_feat(1hop), per state")
        # V2: specificity, per prompt
        t = two.dropna(subset=["state_ablation_logit_drop"])
        t = t[t.n_state_features > 0]
        add("V2_causal_specificity", kind, t.state_ablation_logit_drop.values, t.random_ablation_logit_drop.values,
            "logit drop: state features - random features")
        # V3: graph claim vs intervention
        if len(t) >= 4 and t["IMF_h1"].notna().sum() >= 4:
            from scipy.stats import spearmanr
            rr, pp = spearmanr(t["IMF_h1"], t.state_ablation_logit_drop, nan_policy="omit")
            out.append(dict(test="V3_graph_vs_intervention", kind=kind, quantity="Spearman(IMF, ablation drop)",
                            n=len(t), mean=float(rr), ci_lo=np.nan, ci_hi=np.nan, p_signflip=float(pp)))
    # V4: CLT vs PLT on the same prompts
    if set(df.kind) >= {"plt", "clt"}:
        key = ["hop", "city", "state"]
        w = df.pivot_table(index=key, columns="kind",
                           values=["root_signed_D_feat", "skip_mass", "IMF_h1", "state_ablation_logit_drop", "error_influence"])
        for q in ["root_signed_D_feat", "skip_mass", "IMF_h1", "state_ablation_logit_drop", "error_influence"]:
            for hop in ("2hop", "1hop"):
                if q in w and hop in w.index.get_level_values(0):
                    sub = w[q].xs(hop, level="hop").dropna()
                    if len(sub):
                        add("V4_clt_minus_plt", hop, sub["clt"].values, sub["plt"].values, q)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="results/lm_twohop")
    ap.add_argument("--n-features", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-rank", type=int, default=10, help="keep prompts whose answer token rank <= this")
    ap.add_argument("--max-graph-features", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from .generic_tf import from_hf_gpt2
    tok = GPT2TokenizerFast.from_pretrained(a.model)
    gm = from_hf_gpt2(a.model, device)
    io_fn = HFMLPIO(GPT2LMHeadModel.from_pretrained(a.model).to(device).eval())
    # sanity: HF capture and the generic engine must agree (same weights, same hook points)
    chk = torch.tensor(tok("The capital of Texas is Austin.")["input_ids"], device=device)
    _, c = gm.run(chk)
    hi, ho = io_fn(chk[None])
    gap = max(float((hi[l] - c["mlp_in"][l]).abs().max()) for l in range(gm.cfg.n_layers))
    print(f"HF vs generic engine max |mlp_in| gap: {gap:.2e}")
    assert gap < 1e-3, "HF capture and generic engine disagree; do not trust the run"

    prompts = build_prompts(gm, tok, TRIPLES, a.max_rank)
    prompts.to_csv(os.path.join(a.out, "prompts.csv"), index=False)
    n2 = int((prompts.kept & (prompts.hop == "2hop")).sum()); n1 = int((prompts.kept & (prompts.hop == "1hop")).sum())
    print(f"kept {n2} two-hop and {n1} one-hop prompts (answer rank <= {a.max_rank}, state capital known)")
    if n2 < 8:
        print("WARNING: fewer than 8 two-hop prompts survive. Rerun with --model gpt2-medium "
              "or --max-rank 20, and report both runs.")

    tcs, fid = {}, []
    for kind in ("plt", "clt"):
        path = os.path.join(a.out, f"tc_{kind}.pt")
        if os.path.exists(path):
            tc = torch.load(path, weights_only=False).to(device); hist = []
        else:
            tc, hist = train_lm_transcoder(gm, wikitext_batches(tok, a.batch, a.seq_len, "train"), kind,
                                           a.n_features, a.k, a.steps, seed=a.seed, io_fn=io_fn)
            torch.save(tc, path)
        tcs[kind] = tc
        nm = heldout_nmse(gm, tc, wikitext_batches(tok, a.batch, a.seq_len, "validation"), io_fn=io_fn)
        fid.append(dict(kind=kind, n_features_per_layer=a.n_features, k=a.k, steps=a.steps, n_params=tc.n_params(),
                        heldout_nmse_mean=float(np.mean(nm)), heldout_nmse_max=float(np.max(nm)),
                        **{f"heldout_nmse_L{l}": v for l, v in enumerate(nm)}))
        pd.DataFrame(hist).to_csv(os.path.join(a.out, f"train_curve_{kind}.csv"), index=False)
    fid = pd.DataFrame(fid)
    fid.to_csv(os.path.join(a.out, "lm_fidelity.csv"), index=False)
    print(fid[["kind", "n_params", "heldout_nmse_mean", "heldout_nmse_max"]])

    df = analyse(gm, tok, tcs, prompts, a.max_graph_features, seed=a.seed)
    df.to_csv(os.path.join(a.out, "lm_twohop_graph_metrics.csv"), index=False)
    cols = ["root_signed_D_feat", "root_D_feat", "skip_mass", "IMF_h1", "error_influence", "coverage",
            "state_ablation_logit_drop", "random_ablation_logit_drop", "orig_top1_kept"]
    df.groupby(["kind", "hop"])[[c for c in cols if c in df]].mean().to_csv(os.path.join(a.out, "lm_twohop_summary.csv"))
    tests = summarise(df)
    tests.to_csv(os.path.join(a.out, "lm_validation_tests.csv"), index=False)
    print(tests.to_string(index=False))
    json.dump(dict(vars(a), device=device, torch=torch.__version__), open(os.path.join(a.out, "args.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
