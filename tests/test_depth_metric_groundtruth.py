import numpy as np

from cdc.depth_metrics import (ground_truth_graph, hop_profile, depth_stats, root_sources,
                               graph_metrics, normalized_adjacency)
from cdc.graphs import Graph


def test_ground_truth_graph_depth_equals_D():
    for D in range(1, 6):
        g = ground_truth_graph(D)
        st = depth_stats(hop_profile(g, root_sources(g)))
        assert abs(st["D_feat"] - D) < 1e-9
        assert abs(st["D_eff"] - D) < 1e-9


def _toy(skip: bool):
    """inputs x0,x1 -> f0 (layer 0) -> f1 (layer 1) -> f2 (layer 2) -> y.
    skip=True adds a dominant CLT-style edge f0 -> y (a collapsed one-hop path)."""
    types = ["input", "input", "feature", "feature", "feature", "logit"]
    layer = [-1, -1, 0, 1, 2, 3]
    A = np.zeros((6, 6))
    A[2, 0] = A[2, 1] = 1
    A[3, 2] = 1
    A[4, 3] = 1
    A[5, 4] = 1
    if skip:
        A[5, 2] = 9.0
    return Graph(A=A, node_type=np.array(types), layer=np.array(layer), pos=np.zeros(6, int),
                 feat=np.array([0, 1, 0, 0, 0, -1]), act=np.ones(6), target_preact=np.full(6, np.nan),
                 target_const=np.full(6, np.nan), n_layers=3, meta={})


def test_known_skip_gives_collapse():
    deep = graph_metrics(_toy(False))
    coll = graph_metrics(_toy(True))
    assert abs(deep["root_D_feat"] - 3) < 1e-9
    assert coll["root_D_feat"] < 1.5          # 90% of influence goes through the 1-hop path
    assert coll["skip_mass"] > deep["skip_mass"]


def test_normalization_rows_sum_to_one():
    Ah = normalized_adjacency(_toy(True).A)
    s = Ah.sum(1)
    assert np.allclose(s[s > 0], 1)
