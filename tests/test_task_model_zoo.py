import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
METADATA = ROOT / "metadata"


def test_datasets_equations_and_splits():
    for task in ("2hop", "3hop"):
        df = pd.read_csv(DATA / f"boolean_{task}.csv")
        assert len(df) == 7000
        assert df["split"].value_counts().to_dict() == {
            "train": 5000,
            "validation": 1000,
            "test": 1000,
        }

        for i in range(16):
            a = df[f"a_{i:02d}"].astype(int)
            b = df[f"b_{i:02d}"].astype(int)
            c = df[f"c_{i:02d}"].astype(int)
            u = df[f"u_{i:02d}"].astype(int)
            y = df[f"y_{i:02d}"].astype(int)

            assert (u == (a ^ b)).all()

            if task == "2hop":
                assert (y == (u & c)).all()
            else:
                d = df[f"d_{i:02d}"].astype(int)
                v = df[f"v_{i:02d}"].astype(int)
                assert (v == (u & c)).all()
                assert (y == (v ^ d)).all()


def test_model_zoo_and_checkpoints():
    records = [
        json.loads(line)
        for line in (METADATA / "model_zoo.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 6
    assert len({m["model_id"] for m in records}) == 6
    assert {m["task"] for m in records} == {"2hop", "3hop"}

    for model in records:
        assert model["validated"] is False
        assert (ROOT / model["checkpoint"]).is_file()
        assert (METADATA / f'{model["model_id"]}.json').is_file()
        payload = torch.load(
            ROOT / model["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
        assert isinstance(payload, dict)
        assert "state_dict" in payload
