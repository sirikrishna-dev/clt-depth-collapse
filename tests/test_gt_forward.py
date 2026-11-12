"""The explicit forward pass must reproduce model(x) exactly; otherwise every
ablation in scripts/ground_truth_depth.py would be measuring the wrong model."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from das_interchange import load_model  # noqa: E402
from train_transformer import load_split  # noqa: E402
import gt_forward  # noqa: E402

import re
_PAT = re.compile(r"^[23]hop_L\d+(_siit_k\d+)?_seed\d+\.pt$")
CKPTS = sorted(p for p in (ROOT / "checkpoints").glob("*.pt") if _PAT.match(p.name))


@pytest.mark.parametrize("ckpt", CKPTS, ids=lambda p: p.stem)
def test_forward_matches_model(ckpt):
    model, task = load_model(str(ckpt))
    X, _ = load_split(str(ROOT / "data" / "boolean_combined.csv"), task, "test")
    X = X[:200]
    with torch.no_grad():
        ref = model(X)
    out = gt_forward.forward(model, X)
    assert torch.allclose(ref, out, atol=1e-5), float((ref - out).abs().max())


def test_mlp_intervention_actually_changes_output():
    ckpt = ROOT / "checkpoints" / "2hop_L2_siit_k1_seed0.pt"
    model, task = load_model(str(ckpt))
    X, _ = load_split(str(ROOT / "data" / "boolean_combined.csv"), task, "test")
    X = X[:200]
    clean = gt_forward.forward(model, X)
    zeroed = gt_forward.forward(model, X, {("mlp", 0): lambda m: torch.zeros_like(m)})
    assert not torch.allclose(clean, zeroed)
