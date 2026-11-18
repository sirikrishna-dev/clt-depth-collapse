
import pytest
import torch
from torch import nn

from src.metrics.feature_extraction import extract_feature_graph


class DummyReplacement(nn.Module):
    def forward(self, layer_inputs):
        features = [
            torch.relu(torch.cat([x[..., :1], x[..., 1:2]], dim=-1))
            for x in layer_inputs
        ]
        writes = [torch.zeros_like(x) for x in layer_inputs]
        return writes, features


def test_extracts_features_from_model_forward():
    model = DummyReplacement()

    layer_inputs = [
        torch.tensor([[[0.5, 0.2]]]),
        torch.tensor([[[0.0, 0.8]]]),
    ]

    graph = extract_feature_graph(
        model, layer_inputs, replacement_type="plt"
    )

    assert len(graph.nodes) == 4
    assert graph.node_metadata["plt:L0:F0"]["activation"] == pytest.approx(0.5)
    assert graph.node_metadata["plt:L1:F1"]["activation"] == pytest.approx(0.8)
    assert graph.metadata["extraction_method"] == "replacement_model_forward"


def test_rejects_wrong_number_of_feature_layers():
    class WrongLayerCount(nn.Module):
        def forward(self, layer_inputs):
            return [], [torch.zeros(1, 1, 2)]

    with pytest.raises(ValueError, match="Number of feature layers"):
        extract_feature_graph(
            WrongLayerCount(),
            [torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)],
            replacement_type="clt",
        )


def test_rejects_invalid_model_output():
    class WrongOutput(nn.Module):
        def forward(self, layer_inputs):
            return torch.zeros(1)

    with pytest.raises(ValueError, match="must return"):
        extract_feature_graph(
            WrongOutput(),
            [torch.zeros(1, 1, 2)],
            replacement_type="plt",
        )
