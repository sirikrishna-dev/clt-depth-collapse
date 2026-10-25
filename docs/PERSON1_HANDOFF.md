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

## Activation patching
Whole-layer replacement can transfer a donor's computation wholesale and
is not evidence that a specific intermediate feature is represented.

Targeted token-level patching changed recipient predictions when donor
and recipient u differed, while the recorded same-u control did not.
This is preliminary: replacing a full token hidden vector is not a
selective intervention on u alone.

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
