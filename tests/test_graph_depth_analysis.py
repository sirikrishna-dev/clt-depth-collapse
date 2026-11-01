
import pytest

from src.metrics.graph_utils import AttributionGraph
from src.metrics.graph_depth_analysis import analyze_graph_depth


def build_test_graph():
    graph = AttributionGraph()

    graph.add_node("input", layer=0)
    graph.add_node("stage_u", layer=1)
    graph.add_node("stage_v", layer=2)
    graph.add_node("output", layer=3)

    graph.add_edge("input", "stage_u", 0.9)
    graph.add_edge("stage_u", "stage_v", 0.8)
    graph.add_edge("stage_v", "output", 0.7)

    return graph


def test_three_stage_chain_has_depth_three():
    graph = build_test_graph()

    result = analyze_graph_depth(
        graph,
        input_nodes=["input"],
        output_nodes=["output"],
    )

    assert result["path_exists"] is True
    assert result["number_of_paths"] == 1
    assert result["shortest_path_depth"] == 3
    assert result["longest_path_depth"] == 3


def test_direct_shortcut_is_detected():
    graph = build_test_graph()
    graph.add_edge("input", "output", 0.5)

    result = analyze_graph_depth(
        graph,
        input_nodes=["input"],
        output_nodes=["output"],
    )

    assert result["unique_path_depths"] == [1, 3]
    assert result["shortest_path_depth"] == 1
    assert result["longest_path_depth"] == 3


def test_missing_path_is_reported():
    graph = AttributionGraph()
    graph.add_node("input", layer=0)
    graph.add_node("output", layer=3)

    result = analyze_graph_depth(
        graph,
        input_nodes=["input"],
        output_nodes=["output"],
    )

    assert result["path_exists"] is False
    assert result["shortest_path_depth"] is None


def test_empty_inputs_are_rejected():
    graph = build_test_graph()

    with pytest.raises(ValueError):
        analyze_graph_depth(graph, [], ["output"])
