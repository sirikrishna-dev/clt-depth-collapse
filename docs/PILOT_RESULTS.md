# Person 2 pilot results — engineering smoke run

Run on the provided `2hop_L2_seed0.pt` checkpoint and `boolean_2hop_improved.csv` test/validation split. These are not final scientific results.

## Original model / activation interface

- Test examples: 1,000
- Original model test bit accuracy: 1.0000
- Original model test exact-vector accuracy: 1.0000
- Task input shape: `[1000, 3, 16]`
- MLP input shapes (layer 1, layer 2): `[1000, 16, 64]` each
- MLP write shapes: `[1000, 16, 64]` each
- Residual-stream block input shapes: `[1000, 16, 64]` each
- Residual-stream block output shapes: `[1000, 16, 64]` each
- Activation finite-value check: passed
- Model inputs exclude the intermediate `u`/`v` columns: yes

## One-epoch replacement smoke runs

Configuration: two layers, 16 features per layer, sparsity coefficient `0.0001`, batch size 256, one epoch, seed 0, CPU.

| Replacement | Validation MSE | Validation relative L2 | Test relative L2 |
|---|---:|---:|---:|
| PLT | 1.474116 | 0.923282 | 0.923334 |
| CLT-style | 1.344767 | 0.881845 | 0.881666 |

Absolute validation relative-L2 gap: `0.041437`.

The comparison is `NOT_MATCHED` under the currently coded provisional tolerance of `0.02`. That tolerance has not been approved as the team's scientific threshold. The models were trained for one epoch only to check the end-to-end code path. Their high reconstruction errors mean these replacements are not ready for attribution-graph interpretation.

The CLT in this pilot is a project-specific cross-layer write baseline; it is not yet validated as identical to a canonical library/paper CLT. Person 3 must validate the graph-generation/attribution interface before graph-based claims.

The generated replacement checkpoint binaries are intentionally omitted from this source bundle. Run `scripts/run_person2_smoke.ps1` to reproduce the smoke run locally.
