
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

    def test_disconnected_output_returns_nan(self):
        graph = AttributionGraph()
        graph.add_edge("input", "middle", 1.0)
        graph.add_node("output")

        score = collapse(
            graph,
            {"inputs": ["input"], "outputs": ["output"], "depth": 2},
        )
        self.assertTrue(math.isnan(score))

    def test_cycle_does_not_create_infinite_paths(self):
        graph = AttributionGraph()
        graph.add_edge("input", "A", 1.0)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "A", 1.0)
        graph.add_edge("B", "output", 1.0)

        score = collapse(
            graph,
            {"inputs": ["input"], "outputs": ["output"], "depth": 2},
        )
        self.assertAlmostEqual(score, 0.0)

    def test_non_finite_edge_weight_is_rejected(self):
        graph = AttributionGraph()
        with self.assertRaises(ValueError):
            graph.add_edge("input", "output", float("inf"))

    def test_depth_bool_is_rejected(self):
        graph = AttributionGraph()
        with self.assertRaises(ValueError):
            collapse(
                graph,
                {"inputs": ["input"], "outputs": ["output"], "depth": True},
            )

    def test_total_path_mass_overflow_is_rejected(self):
        graph = AttributionGraph()
        graph.add_edge("input", "output", 1e308)
        graph.add_edge("input", "output", 1e308)

        with self.assertRaises(ValueError):
            collapse(
                graph,
                {"inputs": ["input"], "outputs": ["output"], "depth": 2},
            )


if __name__ == "__main__":
    unittest.main()
