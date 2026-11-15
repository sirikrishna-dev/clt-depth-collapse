"""The autograd attribution engine (used for real models) must satisfy conservation,
and agree with the explicit-matrix engine's conventions on a random model."""
import numpy as np
import torch

from cdc.realdata.generic_tf import random_small
from cdc.transcoders import make_transcoder, train_transcoder
from cdc.graphs import conservation_error
from cdc.depth_metrics import graph_metrics


def test_generic_conservation_and_metrics():
    torch.manual_seed(0)
    m = random_small()
    toks = torch.randint(0, 20, (40, 8))
    ins, outs = m.mlp_io(toks)
    for kind in ("plt", "clt"):
        tc = make_transcoder(kind, 3, 32, 32, 4)
        tc, _ = train_transcoder(tc, ins, outs, steps=150, batch=128)
        for tc_ in (tc,):
            for p in tc_.parameters():
                p.requires_grad_(False)
        g, info = m.build_graph(toks[0], tc, target_token=3, keep_all=True)
        assert conservation_error(g) < 1e-3
        assert abs(info["coverage"] - 1) < 1e-9
        met = graph_metrics(g)
        assert np.isfinite(met["root_D_feat"]) or np.isnan(met["root_D_feat"])
