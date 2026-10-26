
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "_person2_src" / "person2_loader"
PACKAGE.mkdir(parents=True, exist_ok=True)
(PACKAGE / "__init__.py").write_text("")

BRANCH = "origin/feature/person2-replacement-models"


def restore_git_file(git_path, destination):
    data = subprocess.check_output(
        ["git", "cat-file", "blob", f"{BRANCH}:{git_path}"],
        cwd=ROOT,
    )
    destination.write_bytes(data)


for name in (
    "model_adapter.py",
    "activation_cache.py",
    "checkpoint_io.py",
    "models.py",
):
    restore_git_file(
        f"src/replacements/{name}",
        PACKAGE / name,
    )

sys.path.insert(0, str(ROOT / "_person2_src"))

from person2_loader.model_adapter import load_original_checkpoint
from person2_loader.activation_cache import capture_mlp_io
from person2_loader.checkpoint_io import load_replacement_checkpoint
from src.metrics.contribution_graph import build_contribution_graph


def main():
    original_path = ROOT / "checkpoints" / "3hop_L2_seed0.pt"
    data_path = ROOT / "data" / "boolean_3hop.csv"

    if not original_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(
            "Expected Transformer checkpoint or dataset is missing."
        )

    original_model, original_meta = load_original_checkpoint(original_path)

    print("Original task:", original_meta["task"])
    print("Original layers:", len(original_model.encoder.layers))

    captured = capture_mlp_io(
        model=original_model,
        data_path=str(data_path),
        task="3hop",
        split="test",
        batch_size=128,
    )

    print("Captured test examples:", captured["n_examples"])
    print("Original model bit accuracy:", round(captured["bit_accuracy"], 4))
    print("MLP input shapes:", captured["activation_shapes"])
    print("MLP write shapes:", captured["mlp_write_shapes"])

    checkpoints = {
        "plt": ROOT.parent / "3hop_plt_seed0_features128_span1.pt",
        "clt": ROOT.parent / "3hop_clt_seed0_features128_span2.pt",
    }

    for replacement_type, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Replacement checkpoint not found: {path}"
            )

        replacement, metadata = load_replacement_checkpoint(path)

        with torch.no_grad():
            reconstructed, features = replacement(captured["inputs"])

        print(f"\n{replacement_type.upper()} checkpoint")
        print("Metadata task:", metadata.get("task"))
        print(
            "Feature tensor shapes:",
            [tuple(t.shape) for t in features],
        )
        print(
            "Reconstructed write shapes:",
            [tuple(t.shape) for t in reconstructed],
        )

        assert len(features) == captured["n_layers"]
        assert len(reconstructed) == captured["n_layers"]

        for layer, (feature, write, target) in enumerate(
            zip(features, reconstructed, captured["writes"])
        ):
            assert feature.shape[:2] == target.shape[:2], (
                f"Feature dimensions mismatch at layer {layer}"
            )
            assert write.shape == target.shape, (
                f"Reconstruction shape mismatch at layer {layer}"
            )
            assert torch.isfinite(feature).all()
            assert torch.isfinite(write).all()

        graph = build_contribution_graph(
            model=replacement,
            features_by_layer=features,
            replacement_type=replacement_type,
            sample_index=0,
            token_index=0,
        )

        print("Graph nodes:", len(graph.nodes))
        print("Graph edges:", len(graph.edges))

        assert graph.metadata["causal_edges"] is False
        assert len(graph.edges) > 0

        for source, target, weight in graph.edges:
            assert source in graph.nodes
            assert target in graph.nodes
            assert torch.isfinite(torch.tensor(weight)).item()
            assert weight >= 0.0

        print("Contribution graph validation PASSED.")

    print("\nReal-activation integration smoke test PASSED.")
    print("Activation shapes and contribution-graph structure validated.")
    print("Edge weights represent contribution magnitudes, not causal effects.")


if __name__ == "__main__":
    main()
