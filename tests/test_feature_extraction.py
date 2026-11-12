
import pytest
import torch

from src.metrics.feature_nodes import add_feature_activation_nodes
from src.metrics.feature_extraction import extract_feature_graph
from src.metrics.graph_utils import AttributionGraph
from src.replacements.models import (
    PerLayerTranscoder,
    CrossLayerTranscoder,
)


def test_feature_nodes_are_created():
    graph = AttributionGraph()
    features = [
        torch.tensor([[[0.5, 0.8]]]),
        torch.tensor([[[0.2, 0.9]]]),
    ]

    node_ids = add_feature_activation_nodes(
        graph, features, "plt"
    )

    assert len(node_ids) == 2
    assert len(node_ids[0]) == 2
    assert "plt:L0:F0" in graph.nodes
    assert graph.node_metadata["plt:L0:F0"]["activation"] == pytest.approx(0.5)
    assert len(graph.edges) == 0


def test_plt_feature_extraction():
    model = PerLayerTranscoder(
        n_layers=2, d_model=3, n_features=4
    )

    inputs = [
        torch.randn(2, 3, 3),
        torch.randn(2, 3, 3),
    ]

    graph = extract_feature_graph(model, inputs, "plt")

    assert len(graph.nodes) == 8
    assert graph.metadata["replacement_type"] == "plt"
    assert graph.metadata["extraction_method"] == "replacement_model_forward"
    assert len(graph.edges) == 0


def test_clt_feature_extraction():
    model = CrossLayerTranscoder(
        n_layers=2,
        d_model=3,
        n_features=4,
        write_span=2,
    )

    inputs = [
        torch.randn(2, 3, 3),
        torch.randn(2, 3, 3),
    ]

    graph = extract_feature_graph(model, inputs, "clt")

    assert len(graph.nodes) == 8
    assert graph.metadata["replacement_type"] == "clt"
    assert len(graph.edges) == 0


def test_invalid_replacement_type_is_rejected():
    with pytest.raises(ValueError):
        extract_feature_graph(None, [], "unknown")


def test_invalid_sample_index_is_rejected():
    graph = AttributionGraph()
    features = [torch.ones(1, 1, 2)]

    with pytest.raises(IndexError):
        add_feature_activation_nodes(
            graph, features, "plt", sample_index=2
        )


def test_non_finite_activations_are_rejected():
    graph = AttributionGraph()
    features = [torch.tensor([[[float("nan")]]])]

    with pytest.raises(ValueError):
        add_feature_activation_nodes(
            graph, features, "plt"
        )
