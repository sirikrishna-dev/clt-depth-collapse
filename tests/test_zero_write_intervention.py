import torch
from torch import nn


def test_zero_write_intervention_preserves_output():
    """Adding a zero delta to a linear2 output must leave model output unchanged."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).eval()
    x = torch.randn(12, 8)

    with torch.no_grad():
        baseline = model(x)

    zero_delta = torch.zeros(12, 4)

    def add_zero_delta(module, inputs, output):
        return output + zero_delta.to(
            device=output.device,
            dtype=output.dtype,
        )

    handle = model[2].register_forward_hook(add_zero_delta)
    try:
        with torch.no_grad():
            intervened = model(x)
    finally:
        handle.remove()

    max_difference = (intervened - baseline).abs().max().item()
    assert max_difference == 0.0
