# InterpBench compatibility checklist

Status at package creation: **PENDING — not yet verified against your team's PLT/CLT code**.

## Inspect and record
- Repository revision and selected model identifier
- Model architecture and number of layers
- Task and reference-circuit format
- Activation/hook points exposed by the model
- Whether the selected model's circuit granularity matches the graph metric
- Whether the model can be loaded in the same environment as the team's transcoder code
- Whether both PLT and CLT can operate on the same base model and activation locations
- Whether reconstruction fidelity can be matched fairly

## Pass criteria
Only mark compatible when a model loads, a forward pass succeeds, required activations can be captured, the transcoder implementation accepts those activations, and the reference-circuit representation is comparable to the planned metric.

TransformerLens is an analysis library, not itself the model zoo. Existing InterpBench models may not implement the custom Boolean tasks in this package. Treat InterpBench and the generated Boolean task family as related but distinct resources unless a specific compatible pairing is demonstrated.
