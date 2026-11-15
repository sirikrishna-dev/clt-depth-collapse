import numpy as np
import torch

from cdc.transcoders import make_transcoder, fidelity
from cdc.graphs import build_graph, conservation_error
from cdc.depth_metrics import graph_metrics
from cdc.validation import replacement_run, skip_dependence, ablation_check


def test_matched_budget():
    plt = make_transcoder("plt", 3, 64, 128, 8)
    clt = make_transcoder("clt", 3, 64, 128, 8)
    assert plt.cfg.n_features == clt.cfg.n_features and plt.cfg.k == clt.cfg.k
    x = torch.randn(50, 64)
    for tc in (plt, clt):
        for l in range(3):
            assert int((tc.encode(x, l) > 0).sum(-1).max()) <= 8
    pm = make_transcoder("plt_parammatched", 3, 64, 128, 8)
    assert abs(pm.n_params() - clt.n_params()) / clt.n_params() < 0.02
    assert len(plt.pairs) == 3 and len(clt.pairs) == 6


def test_replacement_run_matches_as_mlp_fns(tiny_d2):
    model, data, tcs = tiny_d2
    x = data.subset("test").x[:20]
    for tc in tcs.values():
        a = replacement_run(model, tc, x)
        b = model(torch.as_tensor(x), iv={"mlp_fn": tc.as_mlp_fns()})
        assert torch.allclose(a, b, atol=1e-5)


def test_graph_conservation(tiny_d2):
    model, data, tcs = tiny_d2
    x = data.subset("test").x
    for tc in tcs.values():
        for i in range(3):
            g = build_graph(model, tc, x[i])
            assert conservation_error(g) < 1e-3
            m = graph_metrics(g)
            assert 0 <= m["root_D_feat"] <= model.cfg.n_layers + 1


def test_plt_has_no_cross_layer_decoders(tiny_d2):
    _, _, tcs = tiny_d2
    assert all(s == t for s, t in tcs["plt"].pairs)
    assert any(t > s for s, t in tcs["clt"].pairs)


def test_validation_runs(tiny_d2):
    model, data, tcs = tiny_d2
    te = data.subset("test")
    for tc in tcs.values():
        sd = skip_dependence(model, tc, te.x[:200])
        assert 0 <= sd["full_route_acc"] <= 1
        g = build_graph(model, tc, te.x[0])
        rows = ablation_check(model, tc, g, te.x[0], top=3)
        assert len(rows) == 3
    # PLT has no skip writes, so freezing them cannot change anything
    sd = skip_dependence(model, tcs["plt"], te.x[:200])
    assert abs(sd["full_route_acc"] - sd["stepwise_only_acc"]) < 1e-9
