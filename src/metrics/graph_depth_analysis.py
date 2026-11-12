
"""Graph depth and skipped-layer analysis.

This module measures graph structure only.
It does not establish that edges are causal.
"""

from math import isfinite


def _get_layer(graph, node):
    """Return a node's layer from graph metadata, if available."""
    metadata = graph.node_metadata.get(node, {})
    layer = metadata.get("layer")

    if layer is None:
        return None

    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise ValueError(f"Invalid layer metadata for node {node!r}")

    return layer


def analyze_graph_depth(graph, input_nodes, output_nodes):
    """Analyse input-to-output path lengths and skipped-layer edges.

    Path depth is reported as the number of graph edges.
    A skipped-layer edge is an edge whose target layer is more than
    one layer after its source layer.

    Returns a dictionary suitable for JSON serialization.
    """
    if not input_nodes or not output_nodes:
        raise ValueError("Input and output node lists cannot be empty.")

    paths = graph.enumerate_simple_paths(input_nodes, output_nodes)

    path_lengths = []
    path_details = []

    for nodes, weights in paths:
        edge_count = len(weights)
        path_lengths.append(edge_count)

        path_details.append({
            "nodes": list(nodes),
            "edge_count": edge_count,
            "path_weight_product": _path_weight_product(weights),
        })

    skipped_edges = []

    for source, target, weight in graph.edges:
        source_layer = _get_layer(graph, source)
        target_layer = _get_layer(graph, target)

        if (
            source_layer is not None
            and target_layer is not None
            and target_layer > source_layer + 1
        ):
            skipped_edges.append({
                "source": source,
                "target": target,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "layers_skipped": target_layer - source_layer - 1,
                "weight": weight,
            })

    if not path_lengths:
        return {
            "path_exists": False,
            "number_of_paths": 0,
            "shortest_path_depth": None,
            "longest_path_depth": None,
            "unique_path_depths": [],
            "path_details": [],
            "skipped_layer_edges": skipped_edges,
            "interpretation": (
                "No input-to-output path was found. "
                "This is not evidence of zero computational depth."
            ),
        }

    return {
        "path_exists": True,
        "number_of_paths": len(path_lengths),
        "shortest_path_depth": min(path_lengths),
        "longest_path_depth": max(path_lengths),
        "unique_path_depths": sorted(set(path_lengths)),
        "path_details": path_details,
        "skipped_layer_edges": skipped_edges,
        "interpretation": (
            "Structural path lengths only; edge causality must be "
            "validated separately."
        ),
    }


def _path_weight_product(weights):
    """Return the product of absolute edge weights along a path."""
    product = 1.0

    for weight in weights:
        product *= abs(weight)

        if not isfinite(product):
            raise ValueError("Path weight product is not finite.")

    return product
