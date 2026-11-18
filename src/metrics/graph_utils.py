
from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite


@dataclass
class AttributionNode:
    """A feature node in an attribution graph."""

    node_id: str
    layer: int | None = None
    feature_id: str | int | None = None
    activation: float | None = None
    hook_point: str | None = None


@dataclass
class AttributionGraph:
    """Directed graph with signed attribution weights."""

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
        """Add a node, optionally recording its feature metadata."""

        if isinstance(node, AttributionNode):
            node_id = node.node_id
            details = {
                "layer": node.layer,
                "feature_id": node.feature_id,
                "activation": node.activation,
                "hook_point": node.hook_point,
            }
        else:
            node_id = node
            details = {
                "layer": layer,
                "feature_id": feature_id,
                "activation": activation,
                "hook_point": hook_point,
            }

        # Validate the layer value, including values inside AttributionNode.
        node_layer = details["layer"]
        if node_layer is not None and (
            not isinstance(node_layer, int)
            or isinstance(node_layer, bool)
            or node_layer < 0
        ):
            raise ValueError("Layer must be an integer.")

        # Validate activation values.
        node_activation = details["activation"]
        if node_activation is not None:
            if (
                not isinstance(node_activation, (int, float))
                or isinstance(node_activation, bool)
            ):
                raise ValueError("Activation must be a finite number.")

            if not isfinite(node_activation):
                raise ValueError("Activation must be a finite number.")

            details["activation"] = float(node_activation)

        self.nodes.add(node_id)

        if any(value is not None for value in details.values()):
            self.node_metadata[node_id] = details

    def add_edge(self, source, target, weight):
        """Add a directed edge with a finite signed attribution weight."""

        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
        ):
            raise ValueError("Edge weight must be a finite number.")

        if not isfinite(weight):
            raise ValueError("Edge weight must be a finite number.")

        self.nodes.update((source, target))
        self.edges.append((source, target, float(weight)))

    def adjacency(self):
        """Return the directed adjacency list with edge weights."""

        graph = defaultdict(list)

        for source, target, weight in self.edges:
            graph[source].append((target, weight))

        return graph

    def enumerate_simple_paths(self, inputs, outputs):
        """Return simple input-to-output paths and their edge weights."""

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
