
from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isfinite


@dataclass
class AttributionGraph:
    """Directed graph with signed attribution weights."""

    edges: list = field(default_factory=list)
    nodes: set = field(default_factory=set)

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, source, target, weight):
        if not isfinite(weight):
            raise ValueError("Edge weight must be finite.")

        self.nodes.update((source, target))
        self.edges.append((source, target, float(weight)))

    def adjacency(self):
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
