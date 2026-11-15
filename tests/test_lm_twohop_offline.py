"""Offline end-to-end test of the real-data (GPT-2 two-hop) pipeline.

No internet: a random GPT-2-shaped model and a word-level tokenizer that copies GPT-2's
leading-space convention (" Dallas" is one token; a trailing space is its own token).
Checks that every stage runs, the root spans are right, and every validation test is produced.
"""
import re

import numpy as np
import pandas as pd
import torch

from cdc.realdata import lm_twohop as lm
from cdc.realdata.generic_tf import random_small


class StubTok:
    """GPT-2-like: words carry their leading space; a trailing space is a separate token."""

    def __init__(self, V):
        self.V, self.vocab = V, {}

    def __call__(self, text):
        pieces = re.findall(r" ?[^ ]+| $", text)
        ids = [self.vocab.setdefault(p, (len(self.vocab) * 7 + 3) % self.V) for p in pieces]
        return {"input_ids": ids}


def fake_batches(V, batch, seq_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, V, (batch, seq_len), generator=g)


def test_span_positions_gpt2_convention():
    tok = StubTok(500)
    p = lm.TEMPLATE_2HOP.format(city="Dallas")
    pos = lm.span_positions(tok, p, "Dallas")
    ids = tok(p)["input_ids"]
    assert pos == [len(ids) - 2]                      # " Dallas" is second-to-last, before " is"
    p2 = lm.TEMPLATE_2HOP.format(city="San Antonio")
    assert len(lm.span_positions(tok, p2, "San Antonio")) == 2
    p1 = lm.TEMPLATE_1HOP.format(state="New York")
    assert len(lm.span_positions(tok, p1, "New York")) == 2


def test_pipeline_end_to_end():
    torch.manual_seed(0)
    V = 300
    gm = random_small(n_layers=3, d=32, H=4, d_mlp=64, V=V, n_ctx=32)
    tok = StubTok(V)
    triples = lm.TRIPLES[:10]                          # Texas x5, California x5
    prompts = lm.build_prompts(gm, tok, triples, max_rank=V, filter_mode="vocab")   # keep everything
    assert set(prompts.hop) == {"2hop", "1hop"} and prompts.kept.all()
    pc = lm.build_prompts(gm, tok, triples, max_rank=V, filter_mode="candidates")
    assert (pc.cand_rank >= 1).all() and (pc.kept == ((pc.cand_rank == 1) & pc.state.isin(
        set(pc[(pc.hop == "1hop") & (pc.cand_rank == 1)].state)))).all()
    tcs = {}
    for kind in ("plt", "clt"):
        tc, hist = lm.train_lm_transcoder(gm, fake_batches(V, 4, 16), kind, n_features=64, k=4, steps=30, log_every=10)
        tcs[kind] = tc
        nm = lm.heldout_nmse(gm, tc, fake_batches(V, 4, 16, seed=1), n_batches=1)
        assert len(nm) == 3 and all(np.isfinite(nm))
    df = lm.analyse(gm, tok, tcs, prompts, max_graph_features=40)
    assert set(df.kind) == {"plt", "clt"} and set(df.hop) == {"2hop", "1hop"}
    two = df[df.hop == "2hop"]
    assert two.state_ablation_logit_drop.notna().all()
    assert two.random_ablation_logit_drop.notna().all()
    assert np.isfinite(df.root_signed_D_feat).all()
    tests = lm.summarise(df)
    assert {"V1_depth_sensitivity", "V2_causal_specificity", "V4_clt_minus_plt"} <= set(tests.test)


def test_span_positions_with_fewshot_prefix():
    tok = StubTok(500)
    p = lm.FEWSHOT_2HOP + lm.TEMPLATE_2HOP.format(city="Dallas")
    ids = tok(p)["input_ids"]
    assert lm.span_positions(tok, p, "Dallas") == [len(ids) - 2]
    p1 = lm.FEWSHOT_1HOP + lm.TEMPLATE_1HOP.format(state="Texas")
    assert lm.span_positions(tok, p1, "Texas") == [len(tok(p1)["input_ids"]) - 2]
    for city, state, cap in lm.TRIPLES:          # no tested name may appear in a demo line
        for name in (city, state, cap):
            assert name not in lm.FEWSHOT_2HOP and name not in lm.FEWSHOT_1HOP, name


def test_summarise_survives_no_two_hop():
    df = pd.DataFrame([dict(kind="plt", hop="1hop", city="x", state="Ohio", root_signed_D_feat=1.4,
                            root_D_feat=1.4, skip_mass=0.1, IMF_h1=np.nan, error_influence=0.1)])
    t = lm.summarise(df)
    assert (t.test == "NOT_RUN").all()


def test_leave_one_out_never_uses_tested_prompt():
    acts = np.zeros((3, 1, 4))
    acts[0, 0, 1] = 5.0                                 # only prompt 0 fires feature 1
    states = np.array(["Texas", "Texas", "Ohio"])
    assert (0, 1) in lm.state_selective_features(acts, states, i_exclude=1, state="Texas")
    assert (0, 1) not in lm.state_selective_features(acts, states, i_exclude=0, state="Texas")
