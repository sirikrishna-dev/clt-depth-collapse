
"""Generate PLT/CLT feature-ablation attribution graphs for SIIT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.metrics.graph_utils import AttributionGraph
from src.replacements.activation_cache import capture_mlp_io
from src.replacements.checkpoint_io import load_replacement_checkpoint
from src.replacements.model_adapter import load_original_checkpoint


def get_connections(model, model_type):
    if model_type == "plt":
        return [
            (layer, layer, transcoder.decoder)
            for layer, transcoder in enumerate(model.transcoders)
        ]

    return [
        (int(key.split("_")[0]), int(key.split("_")[1]), decoder)
        for key, decoder in model.decoders.items()
    ]


def build_attribution_graph(model, model_type, layer_inputs):
    graph = AttributionGraph()

    graph.metadata.update({
        "replacement_type": model_type,
        "edge_semantics": (
            "mean_absolute_destination_write_change_after_feature_ablation"
        ),
        "causal_edges": False,
        "causal_scope": "replacement_model_only",
        "aggregation": "mean over test examples, tokens and write dimensions",
    })

    with torch.no_grad():
        if model_type == "plt":
            features = [
                torch.relu(transcoder.encoder(x))
                for transcoder, x in zip(model.transcoders, layer_inputs)
            ]
        else:
            features = [
                torch.relu(encoder(x))
                for encoder, x in zip(model.encoders, layer_inputs)
            ]

        connections = get_connections(model, model_type)
        rows = []
        connection_summary = []

        for source, target, decoder in connections:
            source_features = features[source]
            n_features = source_features.shape[-1]

            write_node = f"{model_type}:write:L{target}"

            graph.add_node(
                write_node,
                layer=target,
                hook_point="predicted_destination_mlp_write",
            )

            connection_effects = []

            for feature_idx in range(n_features):
                original_feature = source_features[..., feature_idx]
                decoder_column = decoder.weight[:, feature_idx]

                delta = (
                    original_feature.unsqueeze(-1)
                    * decoder_column.view(1, 1, -1)
                )

                mean_effect = delta.abs().mean().item()
                max_effect = delta.abs().max().item()
                active_fraction = (
                    (original_feature > 0).float().mean().item()
                )

                feature_node = (
                    f"{model_type}:L{source}:F{feature_idx}"
                )

                graph.add_node(
                    feature_node,
                    layer=source,
                    feature_id=feature_idx,
                    activation=original_feature.mean().item(),
                    hook_point=f"{model_type}_feature_activation",
                )

                graph.add_edge(
                    feature_node,
                    write_node,
                    mean_effect,
                )

                connection_effects.append(mean_effect)

                rows.append({
                    "model_type": model_type,
                    "source_layer": source,
                    "target_layer": target,
                    "feature_id": feature_idx,
                    "mean_absolute_write_change": mean_effect,
                    "max_absolute_write_change": max_effect,
                    "active_fraction": active_fraction,
                })

            connection_summary.append({
                "source_layer": source,
                "target_layer": target,
                "mean_feature_effect": (
                    sum(connection_effects) / len(connection_effects)
                ),
                "max_feature_effect": max(connection_effects),
                "nonzero_feature_count": sum(
                    value > 1e-12 for value in connection_effects
                ),
                "feature_count": n_features,
                "is_skipped_layer_connection": target - source > 1,
            })

    graph.metadata["connection_summary"] = connection_summary
    graph.metadata["edge_count"] = len(graph.edges)
    graph.metadata["node_count"] = len(graph.nodes)

    skipped = [
        item for item in connection_summary
        if item["is_skipped_layer_connection"]
    ]

    graph.metadata["skipped_layer_connection_count"] = len(skipped)
    graph.metadata["skipped_layer_connections"] = skipped

    return graph, rows


def save_graph(graph, path):
    payload = {
        "nodes": sorted(graph.nodes),
        "node_metadata": graph.node_metadata,
        "edges": [
            {
                "source": source,
                "target": target,
                "weight": weight,
            }
            for source, target, weight in graph.edges
        ],
        "metadata": graph.metadata,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_csv(rows, path):
    if not rows:
        raise ValueError("No attribution measurements were generated.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    device = torch.device(args.device)
    task = args.task

    original, original_meta = load_original_checkpoint(
        args.checkpoint, device
    )

    seed = original_meta.get("seed")
    if seed is None:
        raise ValueError(
            "Original checkpoint metadata does not contain a seed."
        )

    captured = capture_mlp_io(
        original,
        args.data,
        task,
        "test",
        batch_size=args.batch_size,
        device=device,
    )

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    checkpoints = {
        "plt": args.plt,
        "clt": args.clt,
    }

    for model_type, checkpoint_path in checkpoints.items():
        print(f"\nProcessing {model_type.upper()}...")

        model, metadata = load_replacement_checkpoint(
            checkpoint_path, device
        )

        if int(metadata["n_layers"]) != int(original_meta["layers"]):
            raise ValueError(
                f"{model_type} layer count does not match original model."
            )

        graph, rows = build_attribution_graph(
            model,
            model_type,
            captured["inputs"],
        )

        graph.metadata.update({
            "task": task,
            "base_seed": seed,
            "n_test_examples": captured["n_examples"],
            "n_layers": metadata["n_layers"],
            "n_features_per_layer": metadata["n_features_per_layer"],
            "write_span": metadata.get("write_span", 1),
            "base_checkpoint": str(args.checkpoint),
            "replacement_checkpoint": str(checkpoint_path),
        })

        graph_path = (
            results_dir /
            f"{task}_{model_type}_seed{seed}_attribution_graph.json"
        )

        csv_path = (
            results_dir /
            f"{task}_{model_type}_seed{seed}_feature_effects.csv"
        )

        save_graph(graph, graph_path)
        save_csv(rows, csv_path)

        all_summaries[model_type] = {
            "graph_file": str(graph_path),
            "feature_effects_file": str(csv_path),
            "edge_count": len(graph.edges),
            "node_count": len(graph.nodes),
            "connection_summary": graph.metadata["connection_summary"],
            "skipped_layer_connections": graph.metadata[
                "skipped_layer_connections"
            ],
            "causal_edges": False,
        }

        print(f"Graph saved: {graph_path}")
        print(f"Feature measurements saved: {csv_path}")

        for item in graph.metadata["connection_summary"]:
            print(
                f"L{item['source_layer']} -> L{item['target_layer']}: "
                f"mean effect={item['mean_feature_effect']:.8f}, "
                f"active features={item['nonzero_feature_count']}"
            )

    comparison = {
        "task": task,
        "siit_seed": seed,
        "test_examples": captured["n_examples"],
        "models": all_summaries,
        "interpretation": (
            "Edges measure replacement-feature ablation effects on "
            "destination MLP writes. They are not validated causal edges "
            "in the original Transformer. Graph depth and causal collapse "
            "must not be inferred from these edges alone."
        ),
    }

    comparison_path = (
        results_dir / f"{task}_seed{seed}_graph_comparison.json"
    )

    comparison_path.write_text(json.dumps(comparison, indent=2))

    print(f"\nComparison saved: {comparison_path}")
    print(f"ATTRIBUTION EXPERIMENT COMPLETED: {task}, seed {seed}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plt", required=True)
    parser.add_argument("--clt", required=True)
    parser.add_argument("--task", choices=["2hop", "3hop"], required=True)
    parser.add_argument("--output-dir", default="results/person3/attribution_multiseed")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
