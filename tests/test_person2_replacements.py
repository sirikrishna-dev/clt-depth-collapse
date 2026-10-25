import torch

from src.replacements.models import CrossLayerTranscoder, PerLayerTranscoder, SparseLayerTranscoder


def test_sparse_layer_transcoder_shapes():
    model = SparseLayerTranscoder(d_model=64, n_features=24)
    x = torch.randn(3, 16, 64)
    y, features = model(x)
    assert y.shape == x.shape
    assert features.shape == (3, 16, 24)
    assert torch.all(features >= 0)


def test_plt_reconstructs_each_layer_shape():
    model = PerLayerTranscoder(n_layers=2, d_model=64, n_features=32)
    xs = [torch.randn(4, 16, 64), torch.randn(4, 16, 64)]
    ys, zs = model(xs)
    assert len(ys) == len(zs) == 2
    assert all(y.shape == x.shape for x, y in zip(xs, ys))
    assert all(z.shape == (4, 16, 32) for z in zs)


def test_clt_span_one_has_only_local_decoders():
    model = CrossLayerTranscoder(n_layers=2, d_model=64, n_features=32, write_span=1)
    assert set(model.decoders.keys()) == {"0_0", "1_1"}
    xs = [torch.randn(4, 16, 64), torch.randn(4, 16, 64)]
    ys, zs = model(xs)
    assert len(ys) == len(zs) == 2
    assert all(y.shape == x.shape for x, y in zip(xs, ys))


def test_clt_span_two_allows_downstream_write():
    model = CrossLayerTranscoder(n_layers=2, d_model=64, n_features=32, write_span=2)
    assert set(model.decoders.keys()) == {"0_0", "0_1", "1_1"}
    xs = [torch.randn(4, 16, 64), torch.randn(4, 16, 64)]
    ys, _ = model(xs)
    assert all(y.shape == x.shape for x, y in zip(xs, ys))


def test_clt_rejects_invalid_span():
    try:
        CrossLayerTranscoder(n_layers=2, write_span=3)
    except ValueError:
        return
    raise AssertionError("Expected invalid write_span to raise ValueError")


def test_task_loader_uses_only_operands_and_target(tmp_path):
    import pandas as pd
    from src.replacements.model_adapter import load_task_split

    row = {"task": "boolean_2hop_xor_then_and", "split": "train", "hop_depth": 2}
    for operand, value in {"a": 0, "b": 1, "c": 1, "u": 0, "v": 1, "y": 1}.items():
        for bit in range(16):
            row[f"{operand}_{bit:02d}"] = value
    path = tmp_path / "toy.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    x, y = load_task_split(path, "2hop", "train")
    assert tuple(x.shape) == (1, 3, 16)
    assert tuple(y.shape) == (1, 16)
    # The deliberately unusual u/v values are not concatenated into X.
    assert torch.all(x[:, 0, :] == 0)
    assert torch.all(x[:, 1, :] == 1)
    assert torch.all(x[:, 2, :] == 1)
    assert torch.all(y == 1)


def test_replacement_checkpoint_round_trip(tmp_path):
    from src.replacements.checkpoint_io import load_replacement_checkpoint

    model = CrossLayerTranscoder(n_layers=2, d_model=64, n_features=12, write_span=2)
    meta = {"n_layers": 2, "d_model": 64, "n_features_per_layer": 12, "write_span": 2}
    path = tmp_path / "toy_clt.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": meta, "model_type": "clt"}, path)
    restored, restored_meta = load_replacement_checkpoint(path)
    xs = [torch.randn(2, 16, 64), torch.randn(2, 16, 64)]
    before, _ = model.eval()(xs)
    after, _ = restored(xs)
    assert len(before) == len(after) == 2
    assert all(torch.allclose(a, b) for a, b in zip(before, after))
    assert restored_meta["write_span"] == 2


def test_capture_mlp_interface_and_residual_diagnostics(tmp_path):
    import pandas as pd
    from src.replacements.activation_cache import capture_mlp_io
    from src.replacements.model_adapter import BitVectorTransformer

    # Four deterministic examples are enough to test hook shapes and cleanup.
    rows = []
    for idx in range(4):
        row = {"task": "boolean_2hop_xor_then_and", "split": "train", "hop_depth": 2}
        operands = {
            "a": [(idx >> (bit % 2)) & 1 for bit in range(16)],
            "b": [((idx + 1) >> (bit % 2)) & 1 for bit in range(16)],
            "c": [1] * 16,
        }
        u = [operands["a"][bit] ^ operands["b"][bit] for bit in range(16)]
        y = [u[bit] & operands["c"][bit] for bit in range(16)]
        for name, values in operands.items():
            for bit, value in enumerate(values):
                row[f"{name}_{bit:02d}"] = value
        # Include ground-truth columns to ensure model inputs still exclude them.
        for name, values in {"u": u, "v": [None] * 16, "y": y}.items():
            for bit, value in enumerate(values):
                row[f"{name}_{bit:02d}"] = value
        rows.append(row)

    csv_path = tmp_path / "tiny_2hop.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    model = BitVectorTransformer(n_inputs=3, layers=2, d_model=64, nhead=4)
    captured = capture_mlp_io(
        model, str(csv_path), "2hop", "train", batch_size=2,
        device="cpu", capture_block_io=True,
    )
    assert captured["model_input_shape"] == (4, 3, 16)
    assert captured["activation_shapes"] == [(4, 16, 64), (4, 16, 64)]
    assert captured["mlp_write_shapes"] == [(4, 16, 64), (4, 16, 64)]
    assert captured["block_input_shapes"] == [(4, 16, 64), (4, 16, 64)]
    assert captured["block_output_shapes"] == [(4, 16, 64), (4, 16, 64)]
    assert captured["block_io_captured"] is True
    assert captured["bit_accuracy"] >= 0.0
