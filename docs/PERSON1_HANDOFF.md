# Person 1 Handoff: Task Design and Model Zoo

## Deliverables
Boolean task definitions, generated datasets, six pilot Transformer
checkpoints, metadata, linear-probe results, activation-patching results,
and a preliminary directional-intervention analysis.

## Task definitions
All operations are bitwise over 16-bit operands.

- 2-hop: u = XOR(a, b); y = AND(u, c).
- 3-hop: u = XOR(a, b); v = AND(u, c); y = XOR(v, d).

The tasks differ in both named computational depth and final operation.
Depth comparisons must account for this difference.

## Dataset
Each task contains 7,000 rows: 5,000 train, 1,000 validation, and
1,000 test examples. These are random tuple splits, not held-out
compositional-generalization splits. Intermediate variables and targets
are dataset labels, not model input features.

## Model zoo
Six two-layer Transformer checkpoints are included: seeds 0, 1, and 2
for each task. The architecture uses hidden dimension 64, four attention
heads, and a per-bit binary output head.

The reported pilot evaluations achieved 100% test bit accuracy and
100% exact-vector accuracy for all six models. Automated checks verify
checkpoint presence, metadata references, and dataset equations.
The metadata remains `validated: false`: predictive accuracy alone does
not establish the model's internal algorithm.

## Linear probing
Across three 2-hop model seeds, the intermediate XOR value u was
almost perfectly linearly decodable from both encoder layers on the
random test split. This demonstrates linear accessibility, not causal use.

## Activation patching and checkpoint status
The current learned-model validation reruns activation patching for all six
checkpoints and writes:

- `results/learned_model_patching_all_checkpoints.csv` (row-level interventions)
- `results/learned_model_patching_summary.csv` (rates by checkpoint, tested
  intermediate, layer, and intervention group)
- `results/patching_filter_checkpoint_status.csv` (exploratory screen only)
- `results/layer_hook_diagnostics.csv` (hook calls, activation shapes, and
  layer-to-layer activation differences)

The summary uses `EXPLORATORY_RESULT_NOT_CAUSAL_VALIDATION`; checkpoint metadata
uses `patching_status: exploratory_screen_pass` when the post-hoc screen passes,
`causal_validation_status: incomplete`, and `validated: false`. These fields
are intentionally not a contradiction: `PASS_EXPLORATORY` means only that the
specified exploratory thresholds were met, not that the model is causally
validated.

The current rerun found that each layer hook fired once during the clean pass
for every checkpoint, both layers had shape `(1000, 16, 64)`, and their clean
activation tensors were not equal (mean absolute differences were nonzero and
all elements differed beyond the numerical tolerance). Thus the hooks are
attached to distinct layer outputs. However, the patched output bit was exactly
the same for corresponding Layer 1 and Layer 2 intervention rows. This remains
an observed result requiring interpretation; it is not proof that the layers
are interchangeable. The patching intervention can cause the same binary
output at both locations even though the hidden activations differ.

The task-specific pair construction now tests:

- 2-hop `u`: recipients and donors are matched with `c=1`; treatment changes
  `u`, while controls hold `u` fixed.
- 3-hop `u`: `c=1` and `d` are matched; treatment changes `u`, while controls
  hold `u` fixed.
- 3-hop `v`: `d` is matched; treatment changes `v`, while controls hold `v`
  fixed.

All six checkpoints currently pass the post-hoc exploratory thresholds, with
100% treatment prediction flips and 0% control flips in the measured groups.
The criterion was not preregistered. Critically, the script replaces the full
64-dimensional hidden vector at one bit-position token, not an isolated `u` or
`v` feature. The donor/recipient matching improves the behavioral contrast but
does not make the internal intervention selective. Therefore these results are
exploratory evidence, not selective causal proof; metadata correctly keeps
`validated: false` and `causal_validation_status: incomplete`.

Whole-layer replacement can transfer a donor's computation wholesale and is
not evidence that a specific intermediate feature is represented. Linear
probing shows accessibility, not causal use.

## Directional intervention
The recorded probe-direction experiment produced no prediction flips.
Its logit effects did not clearly outperform the random-direction control.
The causal role of the probe direction remains inconclusive.

## InterpBench and CLT/PLT compatibility
Compatibility remains pending. This package has not demonstrated that
the custom Boolean tasks are natively supported by InterpBench or that
these checkpoints integrate with the team's PLT/CLT implementation.

Before marking compatibility verified, demonstrate model loading,
forward passes, activation capture at agreed hook points, transcoder
compatibility, and comparable reconstruction fidelity.

## Reproduction
From the repository root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pytest -q

Regenerate datasets:

    python src/generate_dataset.py

Example training commands:

    python src/train_transformer.py --task 2hop --layers 2 --seed 0 --epochs 20
    python src/train_transformer.py --task 3hop --layers 2 --seed 0 --epochs 20

Review script `--help` options before rerunning analyses. Existing CSVs
record the experiments performed for this handoff.

## Next steps for the team
1. Independently rerun test evaluation from each checkpoint.
2. Perform controlled causal interventions with matched controls.
3. Add held-out composition splits if compositional generalization is needed.
4. Test integration with the team's PLT/CLT attribution pipeline.
5. Compare PLT and CLT depth metrics with matched models and reconstruction
   quality before drawing conclusions about depth collapse.
