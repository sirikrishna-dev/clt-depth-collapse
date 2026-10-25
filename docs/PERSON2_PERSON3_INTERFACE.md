# Person 2 ↔ Person 3 interface agreement

## Agreed primary replacement target

The primary PLT/CLT training target is **MLP input → MLP write** for each `nn.TransformerEncoderLayer` in Person 1's `BitVectorTransformer`.

For layer `i`:

- **MLP input hook:** pre-forward input to `encoder.layers[i].linear1`. In the current PyTorch `TransformerEncoderLayer` configuration (`norm_first=False`), this is the normalized input after `norm2`, following the attention residual update.
- **MLP write target:** output of `encoder.layers[i].linear2`. This is captured before the layer's `dropout2` and residual addition. The original model is placed in `eval()` mode, so dropout is disabled.
- **Residual-stream diagnostics:** capture the tensor entering `encoder.layers[i]` and the output tensor leaving that block. These are supplementary diagnostics and are **not** the primary transcoder reconstruction targets.

## Tensor shapes in the current model

For a batch of `B` examples:

- original task input: `[B, 3, 16]` for 2-hop; `[B, 4, 16]` for 3-hop
- MLP input and MLP write for each encoder layer: `[B, 16, 64]`
- residual-stream block input and output for each layer: `[B, 16, 64]`
- model logits: `[B, 16]`

The shapes are verified at runtime by `src.replacements.inspect_activations`; don't assume them for changed architectures.

## Replacement checkpoint contract

A Person 2 replacement checkpoint contains:

- `state_dict`
- `model_type` (`plt` or `clt`)
- `metadata`, including task ID, base checkpoint path/seed, replacement type, hook-point names, feature count, sparsity coefficient, write-span (CLT only), training configuration, reconstruction metrics, and limitations.

Person 3 can load it using `src.replacements.checkpoint_io.load_replacement_checkpoint(path)` when both branches are integrated.

## Graph handoff caveats

- Feature identifiers should remain traceable to replacement type, layer/source layer, and feature index.
- Preserve signed values when recording attributed relationships.
- **Do not treat decoder weights as attribution-edge weights by themselves.** The team still needs to specify and validate how attribution relationships are computed from the replacement models.
- The current Person 3 branch supplies the graph data structure and provisional collapse metric, but does not yet implement a complete graph-generation pipeline from trained replacements.
- Ground-truth `u` and `v` are task labels, not proven labels for learned replacement features.

## Scope

The cross-layer implementation in `src.replacements.models.CrossLayerTranscoder` is a project-specific experimental baseline with explicit source-to-destination write span. It is not claimed to be identical to a particular external CLT implementation. Person 3 should validate its graph interpretation before the team draws scientific conclusions from it.
