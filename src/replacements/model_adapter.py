"""Load the team's BitVectorTransformer checkpoints and task datasets.

The fallback model definition exactly mirrors Person 1's currently supplied
src/train_transformer.py. If that module is present in the integrated repo,
its canonical class is imported instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

try:  # Prefer Person 1's canonical implementation after branches are integrated.
    from src.train_transformer import BitVectorTransformer as BitVectorTransformer  # type: ignore
except ModuleNotFoundError as exc:
    if exc.name not in {"src.train_transformer", "train_transformer"}:
        raise

    class BitVectorTransformer(nn.Module):
        """Faithful fallback mirror of Person 1's BitVectorTransformer."""
        def __init__(self, n_inputs: int, layers: int, d_model: int = 64, nhead: int = 4):
            super().__init__()
            self.input_proj = nn.Linear(n_inputs, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, operands, bits]; each bit position is a token.
            x = x.transpose(1, 2)
            h = self.input_proj(x)
            h = self.encoder(h)
            return self.head(h).squeeze(-1)


TASK_ALIASES = {
    "2hop": {"2hop", "boolean_2hop_xor_then_and"},
    "3hop": {"3hop", "boolean_3hop_xor_and_then_xor"},
}


def load_original_checkpoint(checkpoint_path: str | Path, device: torch.device | str = "cpu"):
    """Load a trusted project checkpoint and reconstruct its model.

    Checkpoints use Python/PyTorch serialization; only load files from trusted
    project sources. Returns (model, checkpoint_metadata).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # weights_only=False is required for compatibility with the team's checkpoint
    # dictionary. Only use checkpoints that came from this project/repository.
    saved: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise ValueError("Checkpoint must be a dict containing 'state_dict'.")
    task = saved.get("task")
    layers = saved.get("layers")
    if task not in TASK_ALIASES or not isinstance(layers, int):
        raise ValueError(f"Unrecognized task/layers in checkpoint: task={task!r}, layers={layers!r}")
    n_inputs = 3 if task == "2hop" else 4
    model = BitVectorTransformer(n_inputs=n_inputs, layers=layers, d_model=64, nhead=4)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, saved


def load_task_split(data_path: str | Path, task: str, split: str):
    """Load only model inputs and final targets from a task CSV.

    Intermediate columns u/v are deliberately not selected as model inputs.
    Supports Person 1's original combined CSV and the improved per-task CSVs.
    Returns float32 tensors X [N, operands, 16] and y [N, 16].
    """
    if task not in TASK_ALIASES:
        raise ValueError("task must be '2hop' or '3hop'")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")

    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    frame = pd.read_csv(data_path)
    if "split" not in frame.columns:
        raise ValueError("Dataset must contain a 'split' column.")
    frame = frame.loc[frame["split"].astype(str) == split].copy()
    if "task" in frame.columns:
        aliases = TASK_ALIASES[task]
        task_rows = frame.loc[frame["task"].astype(str).isin(aliases)]
        if len(task_rows):
            frame = task_rows
        elif len(frame):
            unique_tasks = sorted(frame["task"].astype(str).unique().tolist())
            raise ValueError(f"No rows for task {task!r} in {data_path}; found task values {unique_tasks}")

    if frame.empty:
        raise ValueError(f"No rows found for task={task!r}, split={split!r} in {data_path}")

    operands = ["a", "b", "c"] if task == "2hop" else ["a", "b", "c", "d"]
    input_cols = [f"{name}_{bit:02d}" for name in operands for bit in range(16)]
    target_cols = [f"y_{bit:02d}" for bit in range(16)]
    missing = [col for col in input_cols + target_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required input/target columns: {missing[:8]}")

    # Reshape explicitly to [rows, operand_count, bit_count]. No u/v columns
    # are ever included, even if present in the CSV.
    x = np.stack(
        [frame[[f"{name}_{bit:02d}" for bit in range(16)]].to_numpy(dtype=np.float32)
         for name in operands],
        axis=1,
    )
    y = frame[target_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Model inputs/targets contain NaN or infinite values.")
    return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
