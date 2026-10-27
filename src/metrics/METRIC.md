

\# Attribution Depth-Collapse Metric



\## Definition



C1 = M1 / (M1 + M2+)



\- M1: Sum of path masses for input-to-output paths with one edge.

\- M2+: Sum of path masses for input-to-output paths with at least two edges.

\- Path mass: Product of absolute edge weights along a path.



\## Graph conventions



\- Nodes represent input, intermediate, or output features.

\- Edges are directed and carry finite, signed attribution weights.

\- Absolute weights are used for path mass; signs are preserved in the graph.

\- Paths are simple paths, so a node cannot repeat within a path.

\- Disconnected nodes do not contribute.

\- A zero total path mass produces NaN.



\## Ground-truth comparison



The true computational depth is recorded separately.

Compare the graph's path-depth distribution with the known computation

and use causal interventions to test whether intermediate variables matter.



\## Limitations



This is an initial metric implementation.

Path products and absolute weights are design choices, not universal

properties of attribution graphs. Validate these conventions against

the graph semantics of the actual PLT/CLT implementation before the

main experiment grid.
## Feature-Level Intervention Experiment

### Setup
Feature-level interventions were evaluated on the 3-hop Boolean task using the original Transformer checkpoint and the trained Cross-Layer Transcoder (CLT). Five selected features per source layer were tested across the edges 0→0, 0→1, and 1→1. The intervention conditions were feature-contribution removal and batch-shuffled contribution subtraction.

### Results
- Number of intervention conditions evaluated: 30.
- Baseline bit accuracy: 1.0000.
- Baseline exact-vector accuracy: 1.0000.
- All tested interventions retained bit accuracy and exact-vector accuracy of 1.0000.
- No test examples changed their predicted output.
- Mean absolute baseline logit: 9.249389.
- Minimum absolute baseline logit: 7.920251.
- Lowest minimum absolute intervened logit across experiments: 7.876228.
- Largest observed absolute logit change: 0.514660.

### Interpretation
The tested interventions changed the original model's logits without changing its predicted outputs. The observed prediction stability is consistent with the model retaining sufficient margin from the binary decision boundary for these interventions.

### Limitations
The experiment tests sensitivity to subtracting CLT-estimated feature contributions from the original model's MLP writes. It does not establish that the selected features are causally irrelevant, recover the original model's true computational graph, or validate computational depth. Only selected features and three source-to-target edges were evaluated. The batch-shuffled condition is a diagnostic control and should not be interpreted as definitive causal evidence.


