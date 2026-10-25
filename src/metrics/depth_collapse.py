
from math import isfinite

from .graph_utils import AttributionGraph


def collapse(graph, ground_truth) -> float:
    """
    Calculate the fraction of path attribution mass on one-edge paths.

    graph: AttributionGraph
    ground_truth: dict containing 'inputs', 'outputs', and 'depth'

    Path mass is the product of absolute edge weights.
    Paths are grouped by edge count: one edge vs. two or more.
    """
    if not isinstance(graph, AttributionGraph):
        raise TypeError("graph must be an AttributionGraph.")

    required = {"inputs", "outputs", "depth"}
    if not required.issubset(ground_truth):
        raise ValueError(f"ground_truth must contain {required}.")

    depth = ground_truth["depth"]
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("Ground-truth depth must be a positive integer.")

    paths = graph.enumerate_simple_paths(
        ground_truth["inputs"], ground_truth["outputs"]
    )

    one_hop_mass = 0.0
    multi_hop_mass = 0.0

    for path, weights in paths:
        if not weights:
            continue

        mass = 1.0
        for weight in weights:
            mass *= abs(weight)

        if not isfinite(mass):
            raise ValueError("Path attribution mass overflowed.")

        if len(weights) == 1:
            one_hop_mass += mass
        else:
            multi_hop_mass += mass

    total_mass = one_hop_mass + multi_hop_mass

    if total_mass == 0:
        return float("nan")

    return one_hop_mass / total_mass
