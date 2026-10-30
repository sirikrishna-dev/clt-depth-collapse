"""Reconstruct Lange et al. (2026) 16-input, two-step Boolean residual-MLP circuit.

Described in: https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor/
    cross-layer-transcoders-are-incentivized-to-learn-unfaithful

This is a reconstruction from the published equations, not the authors' code/checkpoint.

Run from the repository root:
    python build_lange_toy.py

Outputs (relative to current working directory):
    checkpoints/lange_toy_handbuilt.npz
    checkpoints/lange_toy_handbuilt.pt  (when torch is installed)
    data/lange_toy_activations.npz
    data/lange_toy_truth_table.csv
    metadata/lange_toy_handbuilt.json

This does not silently modify metadata/model_zoo.jsonl, because the existing
transformer-zoo manifest and loaders may not accept the distinct toy architecture.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

D = 9
MODEL_ID = 'lange_toy_handbuilt'


def build_weights() -> dict[str, np.ndarray]:
    W_in0 = np.zeros((4, D), dtype=np.float32)
    b0 = -np.ones(4, dtype=np.float32)
    W_in0[0, [0, 1]] = [1, -1]
    W_in0[1, [0, 1]] = [-1, 1]
    W_in0[2, [2, 3]] = [1, -1]
    W_in0[3, [2, 3]] = [-1, 1]
    W_out0 = np.zeros((D, 4), dtype=np.float32)
    for k in range(4):
        W_out0[4 + k, k] = 1.0
    W_in1 = np.zeros((1, D), dtype=np.float32)
    W_in1[0, 4:8] = 1.0
    b1 = -np.ones(1, dtype=np.float32)
    W_out1 = np.zeros((D, 1), dtype=np.float32)
    W_out1[8, 0] = 1.0
    return dict(W_in0=W_in0, b0=b0, W_out0=W_out0,
                W_in1=W_in1, b1=b1, W_out1=W_out1)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward(weights: dict[str, np.ndarray], inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Inputs [N,4] with {-1,+1}; return exact replacement training targets."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != 4:
        raise ValueError('Expected inputs with shape [N,4]')
    x0 = np.concatenate([inputs, np.zeros((len(inputs), D - 4), dtype=np.float32)], axis=1)
    pre0 = x0 @ weights['W_in0'].T + weights['b0']
    act0 = relu(pre0)
    out0 = act0 @ weights['W_out0'].T
    x1 = x0 + out0
    pre1 = x1 @ weights['W_in1'].T + weights['b1']
    act1 = relu(pre1)
    out1 = act1 @ weights['W_out1'].T
    x2 = x1 + out1
    return dict(x0=x0, mlp0_in=x0, mlp0_pre=pre0, mlp0_act=act0,
                mlp0_out=out0, mlp1_in=x1, mlp1_pre=pre1,
                mlp1_act=act1, mlp1_out=out1, resid_final=x2, y=x2[:, 8])


def validate_circuit(weights: dict[str, np.ndarray], X: np.ndarray, r: dict[str, np.ndarray]) -> dict:
    a, b, c, d = X.T
    xor_ab = a != b
    xor_cd = c != d
    target = (xor_ab & xor_cd).astype(np.float32)
    assert np.array_equal(r['y'], target), 'Wrong truth table'
    e, f, g, h = r['mlp0_act'].T
    assert np.array_equal(e, ((a == 1) & (b == -1)).astype(np.float32))
    assert np.array_equal(f, ((a == -1) & (b == 1)).astype(np.float32))
    assert np.array_equal(g, ((c == 1) & (d == -1)).astype(np.float32))
    assert np.array_equal(h, ((c == -1) & (d == 1)).astype(np.float32))
    assert np.array_equal(e + f, xor_ab.astype(np.float32))
    assert np.array_equal(g + h, xor_cd.astype(np.float32))
    # Causal edge: neutralize intermediate writes before layer-1 computation.
    x1_ablated = r['mlp1_in'].copy()
    x1_ablated[:, 4:8] = 0
    y_ablated = relu(x1_ablated @ weights['W_in1'].T + weights['b1']).ravel()
    assert np.array_equal(y_ablated, np.zeros_like(target))
    assert np.count_nonzero(y_ablated != target) == 4
    # Cross-input intervention: swap ALL four intermediate dimensions to donor's
    # post-layer-0 activations, keeping the recipient's original input positions.
    # For all 16x16 ordered pairs, downstream output must equal donor output.
    interventions = 0
    for rec in range(len(X)):
        for don in range(len(X)):
            hybrid = r['mlp1_in'][rec].copy()
            hybrid[4:8] = r['mlp1_in'][don, 4:8]
            y_hybrid = float(relu(hybrid @ weights['W_in1'].T + weights['b1'])[0])
            assert y_hybrid == float(target[don])
            interventions += 1
    return {'inputs_tested': len(X), 'true_outputs': int(target.sum()),
            'exact_accuracy': 1.0, 'intermediate_feature_checks_passed': True,
            'intermediate_ablation_changed_outputs': 4,
            'intermediate_ablation_output_all_zero': True,
            'all_intermediate_swap_pairs_tested': interventions,
            'all_intermediate_swap_pairs_passed': True}


def save_torch(weights: dict[str, np.ndarray], X: np.ndarray, target: np.ndarray, path: Path) -> bool:
    try:
        import torch
        from torch import nn
    except ImportError:
        return False

    class ResidualMLPToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp0_in = nn.Linear(D, 4)
            self.mlp0_out = nn.Linear(4, D, bias=False)
            self.mlp1_in = nn.Linear(D, 1)
            self.mlp1_out = nn.Linear(1, D, bias=False)

        def forward(self, inp):
            x = torch.cat((inp, torch.zeros((*inp.shape[:-1], D - 4), dtype=inp.dtype, device=inp.device)), dim=-1)
            x = x + self.mlp0_out(torch.relu(self.mlp0_in(x)))
            x = x + self.mlp1_out(torch.relu(self.mlp1_in(x)))
            return x[..., 8]

    model = ResidualMLPToy()
    with torch.no_grad():
        model.mlp0_in.weight.copy_(torch.from_numpy(weights['W_in0']))
        model.mlp0_in.bias.copy_(torch.from_numpy(weights['b0']))
        model.mlp0_out.weight.copy_(torch.from_numpy(weights['W_out0']))
        model.mlp1_in.weight.copy_(torch.from_numpy(weights['W_in1']))
        model.mlp1_in.bias.copy_(torch.from_numpy(weights['b1']))
        model.mlp1_out.weight.copy_(torch.from_numpy(weights['W_out1']))
        outputs = model(torch.from_numpy(X)).numpy()
        assert np.array_equal(outputs, target)
    torch.save({'state_dict': model.state_dict(), 'task': 'lange_toy',
                'architecture': 'ResidualMLPToy (9-dimensional residual, 2 MLPs, no attention)',
                'layers': 2, 'residual_dim': D, 'hidden_dims': [4, 1]}, path)
    # Ensure serialised weights re-load and match exactly.
    reloaded = torch.load(path, map_location='cpu', weights_only=True)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, reloaded['state_dict'][key])
    return True


def main() -> None:
    root = Path.cwd()
    for dirname in ('checkpoints', 'data', 'metadata'):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    weights = build_weights()
    X = np.array(list(itertools.product((1.0, -1.0), repeat=4)), dtype=np.float32)
    r = forward(weights, X)
    report = validate_circuit(weights, X, r)
    a, b, c, d = X.T
    target = ((a != b) & (c != d)).astype(np.float32)

    np.savez(root / 'checkpoints' / f'{MODEL_ID}.npz', **weights)
    np.savez(root / 'data' / 'lange_toy_activations.npz', inputs=X, target=target,
             **{name: values for name, values in r.items() if name != 'x0'})
    with (root / 'data' / 'lange_toy_truth_table.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['a', 'b', 'c', 'd', 'xor_ab', 'xor_cd', 'e', 'f', 'g', 'h', 'i', 'target'])
        for j, (aa, bb, cc, dd) in enumerate(X):
            writer.writerow([int(aa), int(bb), int(cc), int(dd), int(aa != bb), int(cc != dd),
                             *(int(t) for t in r['mlp0_act'][j]), int(r['y'][j]), int(target[j])])

    torch_saved = save_torch(weights, X, target, root / 'checkpoints' / f'{MODEL_ID}.pt')
    metadata = {
        'model_id': MODEL_ID,
        'source': 'hand_built_circuit',
        'task': '(a XOR b) AND (c XOR d), inputs in {+1,-1}',
        'hop_depth': 2,
        'layers': 2,
        'seed': None,
        'architecture': '2-layer MLP-only residual network, 9-dimensional residual stream (no attention)',
        'residual_dim': D,
        'hidden_dims': [4, 1],
        'checkpoint': f'checkpoints/{MODEL_ID}.pt' if torch_saved else f'checkpoints/{MODEL_ID}.npz',
        'weights_numpy': f'checkpoints/{MODEL_ID}.npz',
        'intermediates': {
            'e,f,g,h': {'definition': 'ReLU(a-b-1), ReLU(b-a-1), ReLU(c-d-1), ReLU(d-c-1)',
                        'location': 'MLP 0 output, residual dimensions 4..7'},
            'i': {'definition': 'ReLU(e+f+g+h-1)', 'location': 'MLP 1 output, residual dimension 8'},
        },
        'n_inputs': 16,
        'validated': True,
        'validation_scope': 'exhaustive truth table and known interventions on a hand-built circuit',
        'validation': report,
        'provenance': 'Weights reconstructed from Lange et al. (2026) written description, not author code/checkpoint',
        'source_url': 'https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor/cross-layer-transcoders-are-incentivized-to-learn-unfaithful',
        'activations_file': 'data/lange_toy_activations.npz',
        'truth_table_file': 'data/lange_toy_truth_table.csv',
        'zoo_manifest_integration': 'pending architecture-specific loader/schema support',
    }
    (root / 'metadata' / f'{MODEL_ID}.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f"Truth table: {report['inputs_tested']}/16 correct; TRUE rows: {report['true_outputs']}/16")
    print(f"Intermediate ablation: {report['intermediate_ablation_changed_outputs']}/16 outputs changed")
    print(f"All intermediate-swap interventions: {report['all_intermediate_swap_pairs_tested']}/256 correct")
    print(f'Torch checkpoint saved: {torch_saved}')
    print('Wrote checkpoints/, data/, metadata/ artifacts; did NOT modify model_zoo.jsonl')


if __name__ == '__main__':
    main()
