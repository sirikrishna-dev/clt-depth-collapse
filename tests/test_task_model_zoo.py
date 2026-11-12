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
    import numpy as np

    records = [
        json.loads(line)
        for line in (METADATA / "model_zoo.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(records) == 17
    assert len({m["model_id"] for m in records}) == 17

    standard = [
        m for m in records
        if m["source"] == "standard_trained_transformer"
    ]
    siit = [
        m for m in records
        if m["source"] == "siit_trained_transformer"
    ]
    lange = [
        m for m in records
        if m["source"] == "hand_built_circuit"
    ]

    assert len(standard) == 6
    assert len(siit) == 10
    assert len(lange) == 1
    assert {m["task"] for m in standard + siit} == {"2hop", "3hop"}
    assert len(records) == len(standard) + len(siit) + len(lange)

    for model in records:
        assert isinstance(model["validated"], bool)
        assert (ROOT / model["checkpoint"]).is_file()

        metadata_file = METADATA / f'{model["model_id"]}.json'
        assert metadata_file.is_file()

        individual = json.loads(metadata_file.read_text())
        assert individual == model

        payload = torch.load(
            ROOT / model["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )

        assert isinstance(payload, dict)
        assert "state_dict" in payload
        assert payload["layers"] == model["layers"]

        if model["source"] == "hand_built_circuit":
            assert model["model_id"] == "lange_toy_handbuilt"
            assert model["validated"] is True
            assert model["layers"] == 2
            assert model["n_inputs"] == 16
            assert payload["task"] == "lange_toy"

            state = payload["state_dict"]
            assert state["mlp0_in.weight"].shape == (4, 9)
            assert state["mlp0_out.weight"].shape == (9, 4)
            assert state["mlp1_in.weight"].shape == (1, 9)
            assert state["mlp1_out.weight"].shape == (9, 1)

            activations_file = ROOT / model["activations_file"]
            assert activations_file.is_file()

            with np.load(activations_file) as data:
                assert data["inputs"].shape == (16, 4)
                assert data["target"].shape == (16,)
                assert data["mlp0_act"].shape == (16, 4)
                assert data["mlp1_act"].shape == (16, 1)
                assert np.array_equal(data["y"], data["target"])
                assert int(data["target"].sum()) == 4

        else:
            assert payload["task"] == model["task"]
            assert payload["seed"] == model["seed"]

            if model["source"] == "standard_trained_transformer":
                assert model["validated"] is False

            elif model["source"] == "siit_trained_transformer":
                assert model["validated"] is True
                assert "siit_sites" in payload
                assert "training_interchange_check" in model

                if model["task"] == "3hop":
                    assert (
                        model["causal_chain_validation"]["status"]
                        == "passed"
                    )
                    assert model["exclusive_depth_verified"] is False
