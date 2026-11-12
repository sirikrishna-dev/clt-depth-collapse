"""Minimal sparse transcoder baselines for the project pilot.

These are explicit research baselines, not a claim to reproduce any specific
external CLT implementation. The team should confirm the CLT read/write
contract with Person 3 before treating generated graphs as final results.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SparseLayerTranscoder(nn.Module):
    """Sparse feature map from one MLP input to that layer's MLP write."""
    def __init__(self, d_model: int = 64, n_features: int = 128):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor):
        features = F.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features


class PerLayerTranscoder(nn.Module):
    """One independent sparse transcoder for each Transformer encoder layer."""
    def __init__(self, n_layers: int, d_model: int = 64, n_features: int = 128):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_features = n_features
        self.transcoders = nn.ModuleList(
            [SparseLayerTranscoder(d_model, n_features) for _ in range(n_layers)]
        )

    def forward(self, layer_inputs: list[torch.Tensor]):
        if len(layer_inputs) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} layer inputs, got {len(layer_inputs)}")
        outputs, features = [], []
        for transcoder, x in zip(self.transcoders, layer_inputs):
            output, z = transcoder(x)
            outputs.append(output)
            features.append(z)
        return outputs, features


class CrossLayerTranscoder(nn.Module):
    """Shared sparse features can write to their own and downstream layers.

    At source layer s, an encoder reads that layer's MLP input. Its feature
    vector can decode into destination layers s .. s+write_span-1 (bounded by
    n_layers). Predictions for a destination are the sum of eligible source
    contributions plus a learned destination bias.

    This is a deliberately minimal, testable CLT-style research baseline. It
    is not asserted to be identical to a particular paper/library's CLT.
    """
    def __init__(
        self,
        n_layers: int,
        d_model: int = 64,
        n_features: int = 128,
        write_span: int = 2,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if not 1 <= write_span <= n_layers:
            raise ValueError(f"write_span must be in [1, {n_layers}]")
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_features = n_features
        self.write_span = write_span
        self.encoders = nn.ModuleList([nn.Linear(d_model, n_features) for _ in range(n_layers)])
        # Each allowed source->destination edge has a separate linear decoder.
        self.decoders = nn.ModuleDict({
            f"{source}_{target}": nn.Linear(n_features, d_model, bias=False)
            for source in range(n_layers)
            for target in range(source, min(n_layers, source + write_span))
        })
        self.destination_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model)) for _ in range(n_layers)]
        )

    def forward(self, layer_inputs: list[torch.Tensor]):
        if len(layer_inputs) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} layer inputs, got {len(layer_inputs)}")
        features = [torch.relu(encoder(x)) for encoder, x in zip(self.encoders, layer_inputs)]
        outputs = []
        for target in range(self.n_layers):
            contributions = []
            for source in range(max(0, target - self.write_span + 1), target + 1):
                key = f"{source}_{target}"
                if key in self.decoders:
                    contributions.append(self.decoders[key](features[source]))
            output = torch.stack(contributions, dim=0).sum(dim=0) + self.destination_bias[target]
            outputs.append(output)
        return outputs, features
