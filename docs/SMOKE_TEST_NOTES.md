# Person 2 implementation smoke-test notes

These are engineering smoke tests, not main-grid research results.

## Latest verification in the prepared development environment

- `python -m pytest -q`: **8 passed**.
- The original `2hop_L2_seed0.pt` checkpoint loaded strictly using Person 1's source.
- On `data/boolean_2hop_improved.csv` test split (1,000 rows), the original model achieved 100% bit accuracy and 100% exact-vector accuracy.
- MLP-input and MLP-write hook captures for both layers had shape `[1000, 16, 64]`.
- Supplementary residual-stream block-input and block-output captures for both layers also had shape `[1000, 16, 64]`.
- Captured activations were finite. The input tensor was `[1000, 3, 16]`, and logits were `[1000, 16]`.
- A one-epoch PLT smoke run completed with 16 features/layer. Validation relative-L2 reconstruction error was about `0.923282`.
- A one-epoch CLT-style smoke run completed with 16 features/layer and write-span 2. Validation relative-L2 reconstruction error was about `0.881845`.
- The validation relative-L2 gap between these two one-epoch runs was about `0.041437`. Under the code's provisional tolerance of `0.02`, the comparison is `NOT_MATCHED`.

The one-epoch replacement runs confirm the training, evaluation, metadata, and checkpoint-writing paths execute. Their reconstruction errors are high and their fidelity is not matched, so they are not adequate final replacements and no depth-collapse claim can be drawn from them. The `0.02` tolerance is an engineering placeholder, not a team-approved scientific criterion. The team should agree on a threshold using baseline runs before the main comparisons.

## Agreed interface with Person 3

- **Primary reconstruction target:** MLP input to MLP write, separately for each encoder layer.
- **MLP input:** pre-hook input to `encoder.layers[i].linear1`; for the current `norm_first=False` PyTorch layer this is after `norm2` and the attention residual update.
- **MLP write:** output of `encoder.layers[i].linear2`, captured before `dropout2` and residual addition; the original model runs in eval mode, so dropout is disabled.
- **Supplementary diagnostics:** residual stream entering and leaving each whole Transformer block. These boundary activations are not the primary transcoder targets.

See `PERSON2_PERSON3_INTERFACE.md` for the checkpoint and graph-handoff contract. The CLT is a clearly labelled project-specific experimental baseline, not yet confirmed as identical to a canonical library/paper implementation. Person 3's current branch contains graph utilities and a provisional metric, not a complete graph-generation pipeline from trained replacement models.
