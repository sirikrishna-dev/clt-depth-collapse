
import pytest
import torch

from src.metrics.feature_nodes import add_feature_activation_nodes
from src.metrics.graph_utils import AttributionGraph


def test_adds_feature_nodes_and_metadata():
    features = [
        torch.tensor([[[0.5, 0.0, 1.2]]]),
        torch.tensor([[[0.1, 0.7]]]),
    ]

    graph = AttributionGraph()

    ids = add_feature_activation_nodes(
        graph, features, replacement_type="plt"
    )

    assert ids == [
        ["plt:L0:F0", "plt:L0:F1", "plt:L0:F2"],
        ["plt:L1:F0", "plt:L1:F1"],
    ]
    assert len(graph.nodes) == 5
    assert graph.node_metadata["plt:L0:F0"]["activation"] == 0.5
    assert graph.node_metadata["plt:L1:F1"]["feature_id"] == 1
    assert graph.metadata["replacement_type"] == "plt"


def test_selects_requested_sample_and_token():
    features = [
        torch.tensor([
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ])
    ]

    graph = AttributionGraph()

    add_feature_activation_nodes(
        graph,
        features,
        replacement_type="clt",
        sample_index=1,
        token_index=0,
    )

    assert graph.node_metadata["clt:L0:F0"]["activation"] == 0.5
    assert graph.node_metadata["clt:L0:F1"]["activation"] == pytest.approx(0.6)


def test_rejects_invalid_replacement_type():
    with pytest.raises(ValueError):
        add_feature_activation_nodes(
            AttributionGraph(),
            [torch.zeros(1, 1, 2)],
            replacement_type="unknown",
        )


def test_rejects_wrong_activation_shape():
    with pytest.raises(ValueError):
        add_feature_activation_nodes(
            AttributionGraph(),
            [torch.zeros(1, 2)],
            replacement_type="plt",
        )


def test_rejects_out_of_range_sample():
    with pytest.raises(IndexError):
        add_feature_activation_nodes(
            AttributionGraph(),
            [torch.zeros(1, 1, 2)],
            replacement_type="plt",
            sample_index=1,
        )


def test_rejects_non_finite_activation():
    with pytest.raises(ValueError):
        add_feature_activation_nodes(
            AttributionGraph(),
            [torch.tensor([[[float("nan")]]])],
            replacement_type="plt",
        )
