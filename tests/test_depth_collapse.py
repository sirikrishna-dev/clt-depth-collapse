from src.metrics.graph_utils import AttributionNode
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

    def test_invalid_layer_in_attribution_node(self):
        graph = AttributionGraph()
        node = AttributionNode(node_id="feature", layer=1.5)

        with self.assertRaises(ValueError):
            graph.add_node(node)

    def test_non_finite_node_activation_is_rejected(self):
        graph = AttributionGraph()

        with self.assertRaises(ValueError):
            graph.add_node("feature", activation=float("nan"))

    def test_string_edge_weight_is_rejected(self):
        graph = AttributionGraph()

        with self.assertRaises(ValueError):
            graph.add_edge("input", "output", "invalid")

    def test_boolean_edge_weight_is_rejected(self):
        graph = AttributionGraph()

        with self.assertRaises(ValueError):
            graph.add_edge("input", "output", True)


    def test_multiple_inputs_and_outputs(self):
        graph = AttributionGraph()
        graph.add_edge("input1", "output1", 2.0)
        graph.add_edge("input2", "middle", 1.0)
        graph.add_edge("middle", "output2", 3.0)

        score = collapse(
            graph,
            {
                "inputs": ["input1", "input2"],
                "outputs": ["output1", "output2"],
                "depth": 2,
            },
        )

        self.assertAlmostEqual(score, 2.0 / 5.0)

    def test_duplicate_edges_are_counted_separately(self):
        graph = AttributionGraph()
        graph.add_edge("input", "output", 1.0)
        graph.add_edge("input", "output", 2.0)

        score = collapse(
            graph,
            {
                "inputs": ["input"],
                "outputs": ["output"],
                "depth": 2,
            },
        )

        self.assertAlmostEqual(score, 1.0)

    def test_attribution_node_metadata_is_preserved(self):
        graph = AttributionGraph()
        node = AttributionNode(
            node_id="feature_0",
            layer=1,
            feature_id=0,
            activation=0.75,
            hook_point="mlp",
        )

        graph.add_node(node)

        self.assertIn("feature_0", graph.nodes)
        self.assertEqual(graph.node_metadata["feature_0"]["layer"], 1)
        self.assertEqual(graph.node_metadata["feature_0"]["feature_id"], 0)
        self.assertAlmostEqual(
            graph.node_metadata["feature_0"]["activation"], 0.75
        )
        self.assertEqual(
            graph.node_metadata["feature_0"]["hook_point"], "mlp"
        )

    def test_negative_layer_in_attribution_node_is_rejected(self):
        graph = AttributionGraph()
        node = AttributionNode(node_id="feature", layer=-1)

        # Negative layer indices are invalid for this project.
        with self.assertRaises(ValueError):
            graph.add_node(node)

if __name__ == "__main__":
    unittest.main()
