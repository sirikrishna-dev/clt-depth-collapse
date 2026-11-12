
"""Basic directed graph utilities for attribution analysis."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite


@dataclass
class AttributionNode:
    node_id: str
    layer: int | None = None
    feature_id: str | int | None = None
    activation: float | None = None
    hook_point: str | None = None


@dataclass
class AttributionGraph:
    edges: list = field(default_factory=list)
    nodes: set = field(default_factory=set)
    node_metadata: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_node(
        self,
        node,
        layer=None,
        feature_id=None,
        activation=None,
        hook_point=None,
    ):
        if isinstance(node, AttributionNode):
            node_id = node.node_id
            layer = node.layer
            feature_id = node.feature_id
            activation = node.activation
            hook_point = node.hook_point
        else:
            node_id = node

        if layer is not None:
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer < 0
            ):
                raise ValueError("layer must be a non-negative integer")

        if activation is not None:
            if (
                isinstance(activation, bool)
                or not isinstance(activation, (int, float))
                or not isfinite(activation)
            ):
                raise ValueError("activation must be finite and numeric")

        self.nodes.add(node_id)

        details = {
            "layer": layer,
            "feature_id": feature_id,
            "activation": activation,
            "hook_point": hook_point,
        }

        if any(value is not None for value in details.values()):
            self.node_metadata[node_id] = details

    def add_edge(self, source, target, weight):
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not isfinite(weight)
        ):
            raise ValueError("Edge weight must be finite and numeric")

        self.nodes.update((source, target))
        self.edges.append((source, target, float(weight)))

    def adjacency(self):
        graph = defaultdict(list)

        for source, target, weight in self.edges:
            graph[source].append((target, weight))

        return graph

    def enumerate_simple_paths(self, inputs, outputs):
        adjacency = self.adjacency()
        output_nodes = set(outputs)
        paths = []

        def dfs(node, visited, path, weights):
            if node in output_nodes:
                paths.append((path.copy(), weights.copy()))
                return

            for target, weight in adjacency[node]:
                if target not in visited:
                    visited.add(target)
                    path.append(target)
                    weights.append(weight)

                    dfs(target, visited, path, weights)

                    weights.pop()
                    path.pop()
                    visited.remove(target)

        for start in inputs:
            if start in self.nodes:
                dfs(start, {start}, [start], [])

        return paths
