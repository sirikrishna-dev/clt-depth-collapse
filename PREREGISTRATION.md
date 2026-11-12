# Pre-registration: CLT vs PLT attribution depth

**Status:** frozen. Committed before any depth metric is computed.
**Branch:** `prereg/depth-collapse` (base: `origin/integration/clean-main`)
**Commit hash:** _post in the group chat after pushing_

Nobody edits this file after the first depth number exists. Any change made afterwards goes in a new section, **Deviations**, with the date, the reason and who made it, and the paper reports it.

---

## 1. Research question

Do cross-layer transcoders (CLTs) make a multi-step computation look shallower in attribution graphs than per-layer transcoders (PLTs), when both are trained under an identical budget and matched fidelity?

## 2. Four quantities kept separate everywhere

| Term | Meaning | How we measure it |
|---|---|---|
| Ground-truth depth (`D_true`) | Sequential MLP steps the original model really uses | Ablation, bypass and chain tests; never by counting layers |
| Attribution depth | Feature steps between input and output in the graph | Depth metrics in Section 5 |
| Decodable | A representation carries information about u or v | Linear probes with shuffled-label controls |
| Causally used | Changing it changes the downstream computation | Interventions and ablations |

Decodable is never reported as causally used.

## 3. Models in scope

| Group | Count | Layers | Seeds |
|---|---|---|---|
| 2-hop baseline | 3 | 2 | 0–2 |
| 3-hop baseline | 3 | 2 | 0–2 |
| 2-hop SIIT | 5 | 2 | 0–4 |
| 3-hop SIIT | 5 | 3 | 0–4 |
| 1-hop control (optional) | ≥3 | 1–2 | 0–2 |

Tasks (element-wise over 16 bit positions):
- 1-hop: `y = a XOR b`
- 2-hop: `u = a XOR b; y = u AND c`
- 3-hop: `u = a XOR b; v = u AND c; y = v XOR d`

Known limitation, stated in advance: each output bit depends on at most 4 input bits at the same position, so one sufficiently wide MLP layer can compute any of these tasks. Multi-step structure in SIIT models is imposed by training. Comparing `D_true` for SIIT and baseline models is how we show the difference between task depth and model depth. The 3-hop baselines have only 2 layers, so `D_true ≤ 2` for them by construction.

## 4. Ground-truth depth (`D_true`)

For each of the original models:

1. Clean bit accuracy and exact-vector accuracy, with the class balance of y reported (in 2-hop about 25% of y bits are 1, so always predicting 0 scores about 75%).
2. Resample-ablate each MLP output and each attention output (3 fixed seeds). Report bit flip rate and balanced-accuracy drop, with denominators.
3. Bypass tests in both directions.
4. Interchange tests for u and v.
   - **v test: donor and recipient must both have c = 1.** With c = 0, v = 0 for both and the swap tests nothing. This replaces the old confounded "decoupled" v protocol.
5. u → v → y chain test (propagation, counterfactual output, restoring v reverses y). Report the filtering rule and denominators.

6. **Early-shortcut test** (all models). For every block before the last necessary MLP, fit a linear probe for y on the residual stream (designated SIIT coordinates excluded), then swap **only** the component along that probe direction from a donor. If the output follows the donor's y, y was already computed at that block and later layers only carry it. A random direction is the control.

**Definition.**
- A **necessary MLP layer** is one whose resample ablation gives a mean flip rate ≥ 5% **or** a balanced-accuracy drop ≥ 5 points (mean over 3 seeds).
- If, at some block L before the last necessary MLP, the early-shortcut score is **≥ 0.95**, then `D_true = L + 1`: y is computed by MLPs 0..L and causally used from there; later MLPs are necessary but carry rather than compute.
- If every early-shortcut score is **≤ 0.05**, `D_true` = the number of necessary MLP layers.
- Otherwise (partial shortcut), `D_true = "unresolved"`, and no depth ratio is computed for that model.
- Linear decodability alone never changes `D_true`. It is reported separately to show the decodable-vs-used contrast.
- For SIIT models, interchange and chain scores ≥ 0.95 are reported as supporting evidence. Bypass recovery (ablate an MLP, restore only the designated coordinate) is reported as "exclusive sites: yes/no". It does not change `D_true`.

## 5. Attribution graph and depth metrics

### Graph
- Nodes: active features (layer, position, index), input embeddings, positions, decoder bias, attention and linear biases, error nodes, and the y logit.
- Edges: computed with attention patterns and LayerNorm scales frozen (list everything frozen in the paper).
- **The PLT has no cross-layer decoder writes, but it does have cross-layer edges through the residual stream** (an L0 feature reaches L1 features and the y logit directly). These edges must be in the PLT graph. The CLT additionally has its cross-layer decoder writes.
- Completeness test: for 32–64 examples on at least one checkpoint per task, in float64, `bias + input contribution + incoming feature contributions + error = pre-activation` with absolute tolerance 1e-4. If it fails, investigate; never hide the residual.

### Metrics (per example, then averaged per model)

| Name | Definition |
|---|---|
| `D_eff` | Expected number of feature nodes visited by a random backward walk from the y logit. Each step picks a source with probability proportional to absolute edge weight; input, bias and error nodes end the walk. |
| `D_max(τ)` | Longest path using only edges that carry ≥ τ of their target's incoming absolute weight. Reported for τ = 0.05 and τ = 0.10. |
| Full-depth path mass | Probability that the backward walk visits at least `D_true` feature nodes. |
| Layer-mass profile | Share of y-logit influence from each source layer. |
| Depth ratio | `D_eff / D_true`. Not computed when `D_true` is unresolved. |
| Error share | Fraction of y-logit influence carried by error nodes. |
| Chain-consistent mass | Share of y-logit influence on paths u-role → v-role → y-role (3-hop) or u-role → y-role (2-hop), versus paths that skip the intermediate role. Roles are assigned by ablation **before** PLT and CLT are compared. |

## 6. PLT and CLT training budget

Both methods share: TopK sparsity (exactly k active features per token), 128 features per layer, k, training steps, batch size, learning rate, initialization seed, data order, and training and evaluation examples. The CLT is the PLT plus cross-layer decoders over the full allowed span. Grid: 10 SIIT checkpoints × {PLT, CLT} × k ∈ {4, 8, 16} (confirmatory). The 6 baselines are added to the grid as a secondary analysis, so depth can be compared across models with `D_true` = 1, 2 and 3.

The CLT has more decoder parameters than the PLT by construction. Parameter counts are reported for every run. A parameter-matched PLT is an optional extra control.

## 7. Frozen thresholds

- **Primary k:** the smallest k in {4, 8, 16} where, for every seed of both tasks, both PLT and CLT reach ≥ 0.99 replacement-model bit agreement with the original. If none does, k = 16 and the result is labelled low-fidelity. All other k values are exploratory and labelled so in every table and figure.
- **Valid pair:** both methods reach ≥ 0.99 bit agreement, and their relative L2 errors differ by at most 10%. Invalid pairs are excluded and counted.
- **Low-fidelity graph:** error nodes carry > 30% of y-logit influence. Excluded from the confirmatory test and reported separately.
- **Completeness tolerance:** float64, absolute 1e-4.

## 8. Confirmatory test and decision rule

Unit of analysis: paired difference per seed, `Δ = depth ratio(CLT) − depth ratio(PLT)`, at the primary k.

Statistics: 95% paired t-interval (primary); 95% bootstrap CI with 10,000 resamples and a fixed seed (supporting); exact sign test (supporting; with 5 seeds the smallest two-sided p is 0.0625, so it cannot reach 0.05 alone). At least 4 valid pairs per task are required.

**Decision rule.** "CLT shows depth collapse" only if **all** hold:
1. Δ is negative in **both** the 2-hop and 3-hop tasks;
2. both the t-interval and the bootstrap CI exclude zero in both tasks;
3. the CLT depth ratio is below 1;
4. ≥ 4 valid seed pairs per task.

Otherwise the result is written as: **"No collapse detected under these conditions."** If fewer than 4 valid pairs exist, the result is **"Inconclusive."**

A non-significant result is not evidence that PLT and CLT are equivalent.

## 9. Real-model check (secondary, not confirmatory)

One documented model with both PLT and CLT transcoders available in circuit-tracer (Gemma-2-2B or Llama-3.2-1B), chosen before running. At least 10 (target 30) matched 2-hop / 1-hop prompt pairs, with expected answers verified and the filtering rule recorded. The same depth metrics as Section 5 are computed, plus an intermediate-feature intervention (e.g. swap "Texas" → "California" and check that the answer becomes "Sacramento"). The real-model transcoders are not budget-matched, so no claim that CLTs are better or worse is made from this check.

## 10. Rules

- Old pilot results live in `results/pilot_v1/`; new matched runs in `results/v2_matched/`. Never mixed in one table.
- Every number in the paper is generated from a CSV by `make_tables.py`.
- Decoder contributions are not called causal edges unless a causal test supports it.
- Feature L0 F22 is not used for any new ablation.

## Deviations

1. **2026-11-13, Person 1: `D_true` rule revised before commit, after the exploratory ground-truth run** (`scripts/ground_truth_depth.py`, results in `results/v2_matched/person1/`).
   - *Original rule:* count the necessary MLP layers, and mark a model "unresolved" if y was linearly decodable before the last necessary layer.
   - *What happened:* y was linearly decodable (balanced accuracy ≥ 0.999) after block 0 in **all 16 models**, so the original rule would have marked every model unresolved and made the confirmatory test impossible.
   - *Revision:* decodability was replaced by a causal early-shortcut test (Section 4, item 6), so only an early y signal that is actually used changes `D_true`.
   - *Scope:* no PLT or CLT depth metric had been computed when this change was made. Sections 5–8 (the CLT vs PLT confirmatory test) are unchanged.
