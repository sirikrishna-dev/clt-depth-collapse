
"""Build feature-to-write contribution graphs for PLT and CLT models.

Edges represent the magnitude of a feature's decoder contribution
to a reconstructed MLP write. They are NOT causal edges.
"""

import torch

from src.metrics.graph_utils import AttributionGraph


def build_contribution_graph(
    model,
    features_by_layer,
    replacement_type,
    sample_index=0,
    token_index=0,
):
    if replacement_type not in {"plt", "clt"}:
        raise ValueError("replacement_type must be 'plt' or 'clt'")

    if len(features_by_layer) != model.n_layers:
        raise ValueError(
            f"Expected {model.n_layers} feature tensors, "
            f"got {len(features_by_layer)}"
        )

    for name, index in (
        ("sample_index", sample_index),
        ("token_index", token_index),
    ):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    graph = AttributionGraph()
    graph.metadata = {
        "replacement_type": replacement_type,
        "edge_semantics": "l2_norm_of_feature_decoder_contribution",
        "causal_edges": False,
    }

    for source_layer, features in enumerate(features_by_layer):
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(
                "Each feature tensor must have shape "
                "[batch, sequence, features]"
            )

        if sample_index >= features.shape[0]:
            raise IndexError("sample_index is outside the feature tensor")

        if token_index >= features.shape[1]:
            raise IndexError("token_index is outside the feature tensor")

        if not torch.isfinite(features).all():
            raise ValueError("Feature tensor contains non-finite values")

        if replacement_type == "plt":
            decoder_targets = {
                source_layer: model.transcoders[source_layer].decoder
            }
        else:
            decoder_targets = {
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

        activation_vector = features[
            sample_index, token_index
        ].detach()

        for target_layer, decoder in decoder_targets.items():
            write_node = (
                f"{replacement_type}:write:L{target_layer}"
            )
            graph.add_node(
                write_node,
                layer=target_layer,
                hook_point="reconstructed_mlp_write",
            )

            weights = decoder.weight.detach()

            if weights.ndim != 2:
                raise ValueError("Decoder weights must be a 2D tensor")

            if weights.shape[1] != activation_vector.numel():
                raise ValueError(
                    "Decoder feature dimension does not match "
                    "the activation dimension"
                )

            for feature_index in range(activation_vector.numel()):
                feature_node = (
                    f"{replacement_type}:L{source_layer}:F{feature_index}"
                )

                activation = activation_vector[feature_index].item()

                graph.add_node(
                    feature_node,
                    layer=source_layer,
                    feature_id=feature_index,
                    activation=activation,
                    hook_point="transcoder_feature",
                )

                contribution = (
                    activation_vector[feature_index]
                    * weights[:, feature_index]
                )

                magnitude = torch.linalg.vector_norm(
                    contribution
                ).item()

                if not torch.isfinite(torch.tensor(magnitude)):
                    raise ValueError(
                        "Non-finite contribution magnitude encountered"
                    )

                graph.add_edge(
                    feature_node,
                    write_node,
                    magnitude,
                )

    graph.metadata.update({
        "sample_index": sample_index,
        "token_index": token_index,
        "interpretation": (
            "Feature-to-write contribution magnitudes only; "
            "no feature-to-feature causal paths are asserted."
        ),
    })

    return graph
