
"""Extract learned feature activations from a PLT/CLT model."""

import torch

from .feature_nodes import add_feature_activation_nodes
from .graph_utils import AttributionGraph


def extract_feature_graph(
    model,
    layer_inputs,
    replacement_type: str,
    sample_index: int = 0,
    token_index: int = 0,
) -> AttributionGraph:
    """Run a replacement model and create nodes from its feature activations.

    The model must return (reconstructed_writes, features_by_layer).
    This creates feature nodes only; it does not infer attribution edges.
    """
    if replacement_type not in {"plt", "clt"}:
        raise ValueError("replacement_type must be 'plt' or 'clt'.")

    if not isinstance(layer_inputs, (list, tuple)) or not layer_inputs:
        raise ValueError("layer_inputs must be a non-empty list or tuple.")

    if not all(isinstance(tensor, torch.Tensor) for tensor in layer_inputs):
        raise TypeError("Every layer input must be a torch.Tensor.")

    model.eval()

    with torch.no_grad():
        result = model(list(layer_inputs))

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError(
            "Model must return (reconstructed_writes, features_by_layer)."
        )

    _, features_by_layer = result

    if not isinstance(features_by_layer, (list, tuple)):
        raise ValueError("Model feature activations must be a list or tuple.")

    if len(features_by_layer) != len(layer_inputs):
        raise ValueError(
            "Number of feature layers must match number of input layers."
        )

    graph = AttributionGraph()

    add_feature_activation_nodes(
        graph=graph,
        features_by_layer=features_by_layer,
        replacement_type=replacement_type,
        sample_index=sample_index,
        token_index=token_index,
    )

    graph.metadata["extraction_method"] = "replacement_model_forward"
    return graph

"""Extract learned feature activations from a PLT/CLT model."""

import torch

from .feature_nodes import add_feature_activation_nodes
from .graph_utils import AttributionGraph


def extract_feature_graph(
    model,
    layer_inputs,
    replacement_type: str,
    sample_index: int = 0,
    token_index: int = 0,
) -> AttributionGraph:
    """Run a replacement model and create nodes from its feature activations.

    The model must return (reconstructed_writes, features_by_layer).
    This creates feature nodes only; it does not infer attribution edges.
    """
    if replacement_type not in {"plt", "clt"}:
        raise ValueError("replacement_type must be 'plt' or 'clt'.")

    if not isinstance(layer_inputs, (list, tuple)) or not layer_inputs:
        raise ValueError("layer_inputs must be a non-empty list or tuple.")

    if not all(isinstance(tensor, torch.Tensor) for tensor in layer_inputs):
        raise TypeError("Every layer input must be a torch.Tensor.")

    model.eval()

    with torch.no_grad():
        result = model(list(layer_inputs))

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError(
            "Model must return (reconstructed_writes, features_by_layer)."
        )

    _, features_by_layer = result

    if not isinstance(features_by_layer, (list, tuple)):
        raise ValueError("Model feature activations must be a list or tuple.")

    if len(features_by_layer) != len(layer_inputs):
        raise ValueError(
            "Number of feature layers must match number of input layers."
        )

    graph = AttributionGraph()

    add_feature_activation_nodes(
        graph=graph,
        features_by_layer=features_by_layer,
        replacement_type=replacement_type,
        sample_index=sample_index,
        token_index=token_index,
    )

    graph.metadata["extraction_method"] = "replacement_model_forward"
    return graph
