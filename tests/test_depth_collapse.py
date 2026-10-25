
import math
import unittest

from src.metrics.graph_utils import AttributionGraph
from src.metrics.depth_collapse import collapse


class TestDepthCollapse(unittest.TestCase):

    def test_two_hop_graph(self):
        graph = AttributionGraph()
        graph.add_edge("input", "X", 1.0)
        graph.add_edge("X", "output", 1.0)

        score = collapse(
            graph,
            {"inputs": ["input"], "outputs": ["output"], "depth": 2},
        )
        self.assertAlmostEqual(score, 0.0)

    def test_direct_path(self):
        graph = AttributionGraph()
        graph.add_edge("input", "output", -2.0)

        score = collapse(
            graph,
            {"inputs": ["input"], "outputs": ["output"], "depth": 2},
        )
        self.assertAlmostEqual(score, 1.0)

    def test_mixed_paths(self):
        graph = AttributionGraph()
        graph.add_edge("input", "output", 1.0)
        graph.add_edge("input", "X", 1.0)
        graph.add_edge("X", "output", 1.0)

        score = collapse(
            graph,
            {"inputs": ["input"], "outputs": ["output"], "depth": 2},
        )
        self.assertAlmostEqual(score, 0.5)

    def test_zero_mass(self):
        graph = AttributionGraph()
        graph.add_edge("input", "output", 0.0)

        score = collapse(
            graph,
            {"inputs": ["input"], "outputs": ["output"], "depth": 2},
        )
        self.assertTrue(math.isnan(score))

    def test_invalid_depth(self):
        graph = AttributionGraph()
        with self.assertRaises(ValueError):
            collapse(
                graph,
                {"inputs": ["input"], "outputs": ["output"], "depth": 0},
            )


if __name__ == "__main__":
    unittest.main()
