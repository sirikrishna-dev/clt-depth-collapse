# Graph Semantics Review — Stage 5

## 1. Purpose

This document defines the meaning of the PLT and CLT contribution graphs used in the depth-collapse research project. It separates the quantities directly computed by the implementation from causal or computational interpretations that require further evidence.

## 2. Node Semantics

### Feature Nodes

A feature node represents a feature activation from a particular source layer for a selected sample and token.

### Reconstructed-Write Nodes

A write node represents the destination layer receiving reconstructed contributions from one or more features.

These nodes represent features and reconstructed writes. They do not automatically represent ground-truth computational operations such as XOR or AND.

## 3. Edge Semantics

An edge connects a feature node to a reconstructed-write node.

For feature activation \(a_k\) and decoder column \(W_{:,k}\), the contribution vector is:

\[
c_k = a_k W_{:,k}
\]

The current edge weight is its L2 magnitude:

\[
w_k = \|c_k\|_2
\]

This weight measures the magnitude of the feature's decoder contribution for the selected sample and token.

It is nonnegative and does not preserve the direction or sign of the contribution vector. It is not, by itself, a causal effect, feature importance score, or validated attribution.

## 4. PLT Graph Interpretation

In the current implementation, each PLT feature contributes to the reconstructed write of its own layer.

The graph describes feature-to-write contribution magnitudes within each layer. It does not establish how a feature causally affects features in later layers.

## 5. CLT Graph Interpretation

In the current implementation, a CLT feature can contribute to its own layer's write and to permitted downstream writes according to the configured write span.

The graph records these feature-to-write contribution magnitudes. The current CLT implementation computes features from supplied original layer inputs rather than propagating reconstructed writes through subsequent feature encoders.

Therefore, a feature-to-write edge must not be interpreted as a validated feature-to-feature causal transition.

## 6. Relationship to Computational Depth

The ground-truth 3-hop task contains the following logical sequence:

1. \(u = \operatorname{XOR}(a,b)\)
2. \(v = \operatorname{AND}(u,c)\)
3. \(y = \operatorname{XOR}(v,d)\)

This sequence defines the task's ground-truth computation. It does not prove that the Transformer or its transcoder representation implements the same internal sequence.

The current contribution graph does not contain validated transitions between successive computational features. Consequently, graph path lengths from this graph must not yet be treated as validated measures of computational depth or used to claim depth collapse.

## 7. Limitations

- Edge weights are L2 contribution magnitudes, not signed causal effects.
- Contribution magnitudes depend on the selected sample and token.
- The graph does not currently validate feature-to-feature causal transitions.
- A feature's decodability or activation does not prove that the model causally uses it.
- Replacing an entire layer may transfer computation without isolating a specific feature.
- The graph's structure alone cannot establish whether the model computes the ground-truth intermediate variables.

## 8. Evidence Required Before Making Depth Claims

Before interpreting graph paths as computational depth, the project needs a defensible definition of the transitions being counted and empirical validation of those transitions.

Potential follow-up experiments include targeted interventions, activation patching with appropriate controls, and sensitivity tests. These experiments must distinguish effects attributable to a particular feature from effects caused by changing a full hidden vector or other correlated information.

The intervention design and its results should be documented separately. No causal transition should be added merely because it makes the graph match the expected task depth.

## 9. Current Decision

**Decision:** Retain the current graph as a feature-to-reconstructed-write contribution-magnitude graph. Do not interpret its path lengths as computational depth until the required transitions are defined and independently validated.

This is a limitation of the current evidence, not proof that the graph or model is incorrect.
