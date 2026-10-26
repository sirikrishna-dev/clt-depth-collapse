
"""Build feature-to-write contribution graphs for PLT and CLT models.

Edges represent the magnitude of each feature's contribution to a
reconstructed MLP write. They do not represent causal influence between
features.
"""

from math import isfinite

import torch

from src.metrics.graph_utils import AttributionGraph


def build_contribution_graph(
    model,
    features_by_layer,
    replacement_type,
    sample_index=0,
    token_index=0,
):
    """Build a feature-to-write graph from a trained PLT or CLT model.

    For each feature, its decoder contribution is:
        feature_activation * decoder_weight[:, feature_index]

    The edge weight is the L2 norm of this contribution vector.
    Consequently, edge weights are nonnegative contribution magnitudes,
    not signed causal attributions.
    """
    if replacement_type not in {"plt", "clt"}:
        raise ValueError("replacement_type must be 'plt' or 'clt'.")

    if len(features_by_layer) != model.n_layers:
        raise ValueError("Feature-layer count does not match model.n_layers.")

    for name, index in (
        ("sample_index", sample_index),
        ("token_index", token_index),
    ):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")

    graph = AttributionGraph()
    graph.metadata.update(
        {
            "replacement_type": replacement_type,
            "edge_semantics": "l2_norm_of_feature_decoder_contribution",
            "causal_edges": False,
        }
    )

    for source_layer, features in enumerate(features_by_layer):
        if features.ndim != 3:
            raise ValueError(
                "Feature tensors must have shape [batch, sequence, features]."
            )

        if sample_index >= features.shape[0]:
            raise IndexError("sample_index is outside the feature batch.")

        if token_index >= features.shape[1]:
            raise IndexError("token_index is outside the sequence.")

        if not torch.isfinite(features).all():
            raise ValueError("Feature activations must be finite.")

        if replacement_type == "plt":
            decoders = {
                source_layer: model.transcoders[source_layer].decoder
            }
        else:
            decoders = {
                target_layer: model.decoders[
                    f"{source_layer}_{target_layer}"
                ]
                for target_layer in range(
                    source_layer,
                    min(
                        model.n_layers,
                        source_layer + model.write_span,
                    ),
                )
            }

        for target_layer, decoder in decoders.items():
            write_node = f"{replacement_type}:write:L{target_layer}"
            graph.add_node(write_node, layer=target_layer)
            graph.node_metadata[write_node].update({
                "node_type": "reconstructed_write",
                "replacement_type": replacement_type,
            })
            activation_vector = features[
                sample_index, token_index
            ].detach()

            weights = decoder.weight.detach()

            if weights.ndim != 2 or weights.shape[1] != activation_vector.numel():
                raise ValueError("Decoder dimensions do not match feature count.")

            for feature_index, activation in enumerate(activation_vector):
                feature_node = (
                    f"{replacement_type}:L{source_layer}:F{feature_index}"
                )

                graph.add_node(
                feature_node,
                layer=source_layer,
                feature_id=feature_index,
                activation=float(activation.item()),
            )
                graph.node_metadata[feature_node].update({
                    "node_type": "feature",
                    "replacement_type": replacement_type,
                })

                contribution = activation * weights[:, feature_index]
                magnitude = torch.linalg.vector_norm(contribution).item()

                if not isfinite(magnitude):
                    raise ValueError("Feature contribution magnitude is not finite.")

                graph.add_edge(feature_node, write_node, magnitude)

    graph.metadata.update(
        {
            "sample_index": sample_index,
            "token_index": token_index,
            "interpretation": (
                "Feature-to-write contribution magnitudes only; "
                "no feature-to-feature causal paths are asserted."
            ),
        }
    )

    return graph
