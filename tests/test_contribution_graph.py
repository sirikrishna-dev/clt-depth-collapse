
import pytest
import torch

from src.metrics.contribution_graph import build_contribution_graph
from src.replacements.models import (
    PerLayerTranscoder,
    CrossLayerTranscoder,
)


def test_plt_builds_feature_to_write_edges():
    model = PerLayerTranscoder(
        n_layers=2,
        d_model=3,
        n_features=2,
    )

    features = [
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[3.0, 4.0]]]),
    ]

    graph = build_contribution_graph(
        model, features, "plt"
    )

    assert graph.metadata["causal_edges"] is False

    expected = torch.linalg.vector_norm(
        features[0][0, 0, 0]
        * model.transcoders[0].decoder.weight[:, 0]
    ).item()

    matching_edges = [
        weight
        for source, target, weight in graph.edges
        if source == "plt:L0:F0"
        and target == "plt:write:L0"
    ]

    assert len(matching_edges) == 1
    assert matching_edges[0] == pytest.approx(expected)


def test_clt_includes_allowed_downstream_write():
    model = CrossLayerTranscoder(
        n_layers=2,
        d_model=3,
        n_features=2,
        write_span=2,
    )

    features = [
        torch.ones(1, 1, 2),
        torch.ones(1, 1, 2),
    ]

    graph = build_contribution_graph(
        model, features, "clt"
    )

    assert graph.metadata["causal_edges"] is False

    matching_edges = [
        (source, target, weight)
        for source, target, weight in graph.edges
        if source == "clt:L0:F0"
        and target == "clt:write:L1"
    ]

    assert len(matching_edges) == 1
    assert matching_edges[0][2] >= 0.0


def test_invalid_replacement_type_is_rejected():
    with pytest.raises(ValueError):
        build_contribution_graph(None, [], "unknown")
