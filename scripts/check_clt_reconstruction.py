
from pathlib import Path
import torch

from _person2_src.person2_loader.model_adapter import load_original_checkpoint
from _person2_src.person2_loader.checkpoint_io import load_replacement_checkpoint
from _person2_src.person2_loader.activation_cache import capture_mlp_io

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cpu"

original_path = ROOT / "checkpoints" / "3hop_L2_seed0.pt"
data_path = ROOT / "data" / "boolean_3hop.csv"
clt_path = ROOT.parent / "3hop_clt_seed0_features128_span2.pt"

original, _ = load_original_checkpoint(original_path, DEVICE)
clt, metadata = load_replacement_checkpoint(clt_path, DEVICE)

cache = capture_mlp_io(
    model=original,
    data_path=str(data_path),
    task="3hop",
    split="test",
    device=DEVICE,
)

with torch.no_grad():
    reconstructed_writes, features = clt(cache["inputs"])

print("CLT metadata:", metadata)
print("Original model test accuracy:", cache["bit_accuracy"])
print()

for layer_idx, (actual, reconstructed) in enumerate(
    zip(cache["writes"], reconstructed_writes)
):
    actual = actual.float()
    reconstructed = reconstructed.float()

    mse = torch.mean((actual - reconstructed) ** 2).item()
    relative_l2 = (
        torch.linalg.vector_norm(actual - reconstructed)
        / torch.linalg.vector_norm(actual).clamp_min(1e-12)
    ).item()

    variance = torch.var(actual, unbiased=False)
    explained_variance = (
        1.0 - torch.mean((actual - reconstructed) ** 2)
        / variance.clamp_min(1e-12)
    ).item()

    print(f"Layer {layer_idx}")
    print(f"  Actual write RMS:       {actual.square().mean().sqrt().item():.6f}")
    print(f"  Reconstructed write RMS:{reconstructed.square().mean().sqrt().item():.6f}")
    print(f"  MSE:                    {mse:.8f}")
    print(f"  Relative L2 error:      {relative_l2:.6f}")
    print(f"  Explained variance:     {explained_variance:.6f}")
    print()

print("Feature activation shapes:")
for idx, z in enumerate(features):
    print(f"  Layer {idx}: {tuple(z.shape)}")
