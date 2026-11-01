"""Save/load helpers for Person 2 replacement checkpoints."""
from __future__ import annotations
from pathlib import Path
import torch
from .models import CrossLayerTranscoder, PerLayerTranscoder


def load_replacement_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a replacement model from a Person 2 training artifact."""
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not {"state_dict", "metadata", "model_type"}.issubset(saved):
        raise ValueError("Not a Person 2 replacement checkpoint (missing state_dict/metadata/model_type).")
    meta = saved["metadata"]
    model_type = saved["model_type"]
    kwargs = {
        "n_layers": int(meta["n_layers"]),
        "d_model": int(meta.get("d_model", 64)),
        "n_features": int(meta["n_features_per_layer"]),
    }
    if model_type == "plt":
        model = PerLayerTranscoder(**kwargs)
    elif model_type == "clt":
        model = CrossLayerTranscoder(**kwargs, write_span=int(meta["write_span"]))
    else:
        raise ValueError(f"Unknown replacement model_type: {model_type!r}")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    return model, meta
