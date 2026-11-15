import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_d2():
    """A quickly trained D=2 SIIT model + PLT and CLT (shared by several tests)."""
    from cdc.tasks import make_dataset
    from cdc.model import ModelConfig
    from cdc.siit import train_siit
    from cdc.transcoders import make_transcoder, collect_mlp_io, train_transcoder
    data = make_dataset(2, n_train=2000, n_val=200, n_test=300)
    model, _ = train_siit(ModelConfig(D=2, n_layers=2), data, seed=0, steps=600, verbose=False)
    ins, outs = collect_mlp_io(model, data.subset("train").x)
    tcs = {}
    for kind in ("plt", "clt"):
        tc = make_transcoder(kind, 2, 64, 64, 4)
        tcs[kind], _ = train_transcoder(tc, ins, outs, steps=400, batch=512)
    return model, data, tcs
