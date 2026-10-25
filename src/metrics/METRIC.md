

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



