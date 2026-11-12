
import pytest

from src.metrics.graph_utils import AttributionGraph
from src.metrics.graph_depth_analysis import analyze_graph_depth


def test_shortest_and_longest_path_depth():
    graph = AttributionGraph()

    graph.add_edge("input", "a", 0.8)
    graph.add_edge("a", "output", 0.5)

    graph.add_edge("input", "b", 0.4)
    graph.add_edge("b", "c", 0.6)
    graph.add_edge("c", "output", 0.7)

    result = analyze_graph_depth(
        graph, ["input"], ["output"]
    )

    assert result["path_exists"] is True
    assert result["number_of_paths"] == 2
    assert result["shortest_path_depth"] == 2
    assert result["longest_path_depth"] == 3
    assert result["unique_path_depths"] == [2, 3]


def test_skipped_layer_edge_detection():
    graph = AttributionGraph()

    graph.add_node("feature_L0", layer=0)
    graph.add_node("feature_L2", layer=2)
    graph.add_edge("feature_L0", "feature_L2", 0.8)

    result = analyze_graph_depth(
        graph, ["feature_L0"], ["feature_L2"]
    )

    skipped = result["skipped_layer_edges"]

    assert len(skipped) == 1
    assert skipped[0]["layers_skipped"] == 1


def test_no_input_to_output_path():
    graph = AttributionGraph()

    graph.add_edge("input", "middle", 0.5)
    graph.add_node("output")

    result = analyze_graph_depth(
        graph, ["input"], ["output"]
    )

    assert result["path_exists"] is False
    assert result["number_of_paths"] == 0
    assert result["shortest_path_depth"] is None
    assert result["longest_path_depth"] is None


def test_path_weight_product():
    graph = AttributionGraph()

    graph.add_edge("input", "middle", -0.5)
    graph.add_edge("middle", "output", 0.4)

    result = analyze_graph_depth(
        graph, ["input"], ["output"]
    )

    assert result["path_details"][0]["path_weight_product"] == pytest.approx(
        0.2
    )


def test_empty_input_nodes_are_rejected():
    graph = AttributionGraph()

    with pytest.raises(ValueError):
        analyze_graph_depth(graph, [], ["output"])


def test_invalid_layer_metadata_is_rejected():
    graph = AttributionGraph()

    graph.nodes.update(["input", "output"])

    # Deliberately inject invalid metadata to test the analysis function.
    graph.node_metadata["input"] = {"layer": -1}
    graph.node_metadata["output"] = {"layer": 2}

    graph.add_edge("input", "output", 0.5)

    with pytest.raises(ValueError):
        analyze_graph_depth(graph, ["input"], ["output"])