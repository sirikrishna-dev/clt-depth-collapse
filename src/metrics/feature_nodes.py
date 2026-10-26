
"""Build attribution-graph nodes from PLT/CLT feature activations."""

from math import isfinite

from .graph_utils import AttributionGraph


def add_feature_activation_nodes(
    graph: AttributionGraph,
    features_by_layer,
    replacement_type: str,
    sample_index: int = 0,
    token_index: int = 0,
) -> list[list[str]]:
    """Add one node per learned feature for one sample and token.

    Each layer's feature tensor must have shape:
        [batch_size, sequence_length, n_features]

    This function creates nodes only. It does not infer attribution edges.
    """
    if not isinstance(graph, AttributionGraph):
        raise TypeError("graph must be an AttributionGraph.")

    if replacement_type not in {"plt", "clt"}:
        raise ValueError("replacement_type must be 'plt' or 'clt'.")

    if not features_by_layer:
        raise ValueError("features_by_layer must not be empty.")

    for name, index in (
        ("sample_index", sample_index),
        ("token_index", token_index),
    ):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    layer_node_ids = []

    for layer_idx, activations in enumerate(features_by_layer):
        if not hasattr(activations, "shape") or len(activations.shape) != 3:
            raise ValueError(
                f"Layer {layer_idx} activations must have shape "
                "[batch, sequence, features]."
            )

        batch_size, sequence_length, n_features = activations.shape

        if batch_size == 0 or sequence_length == 0 or n_features == 0:
            raise ValueError("Activation dimensions must be non-zero.")

        if sample_index >= batch_size:
            raise IndexError("sample_index is outside the activation batch.")

        if token_index >= sequence_length:
            raise IndexError("token_index is outside the sequence.")

        current_layer_ids = []

        for feature_idx in range(n_features):
            value = activations[
                sample_index, token_index, feature_idx
            ].item()

            if isinstance(value, bool) or not isfinite(value):
                raise ValueError("Feature activations must be finite numbers.")

            node_id = (
                f"{replacement_type}:L{layer_idx}:F{feature_idx}"
            )

            graph.add_node(
                node_id,
                layer=layer_idx,
                feature_id=feature_idx,
                activation=float(value),
                hook_point=f"{replacement_type}_feature_activation",
            )

            current_layer_ids.append(node_id)

        layer_node_ids.append(current_layer_ids)

    graph.metadata["replacement_type"] = replacement_type
    graph.metadata["sample_index"] = sample_index
    graph.metadata["token_index"] = token_index

    return layer_node_ids
