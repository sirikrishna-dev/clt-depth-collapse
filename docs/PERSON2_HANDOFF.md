# Person 2: PLT/CLT Replacement-Model Pilot

## What this implementation does

This folder adds an initial, executable research baseline for Person 2:

1. Loads a trusted `BitVectorTransformer` checkpoint from Person 1.
2. Loads the improved per-task CSV and explicitly selects only operands as model inputs (`a,b,c` for 2-hop; `a,b,c,d` for 3-hop). It never feeds `u` or `v` to the model.
3. Captures each Transformer's encoder-layer MLP input (the input to `linear1`, after `norm2` for the default `norm_first=False` layer) and MLP write (the output of `linear2`, before `dropout2` and residual addition; dropout is disabled in evaluation mode).
4. Trains per-layer sparse transcoders (PLT baseline) and a minimal shared-feature cross-layer write baseline (CLT-style baseline).
5. Reports validation and test reconstruction MSE, relative L2 error, explained variance, feature activity, and checkpoints.

## Important architecture caveat

`CrossLayerTranscoder` is a deliberately minimal CLT-style experimental implementation. Each source layer encodes its MLP input into sparse features; those features can decode to the source layer and a configurable number of downstream layer writes. Contributions to the same destination layer are summed. This is explicit and tested, but it has **not** been established as identical to the exact CLT implementation intended by a particular paper or library. Agree the read/write contract with Person 3 before using these outputs for final graph claims.

The transcoders reconstruct MLP writes, not all residual-stream activations or final task outputs. Prediction accuracy of the original Transformer and transcoder reconstruction fidelity are separate measurements.

## Dependencies and required project files

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The run requires the original model checkpoint, e.g. `checkpoints/2hop_L2_seed0.pt`, and a task CSV, e.g. `data/boolean_2hop_improved.csv`. This Person 2 branch ZIP originally did not include Person 1's checkpoints or source; make those available in the repository (merge the branch or obtain the files through the team's agreed process) before running the command.

The model loader imports `src.train_transformer.BitVectorTransformer` if Person 1's source is present. Otherwise it uses a local fallback that mirrors the supplied source; strict state-dict loading is used to detect architectural mismatch.

## Commands

Run tests:

```bash
python -m pytest -q
```

Train a PLT baseline for the 2-hop seed-0 checkpoint:

```bash
python -m src.replacements.train --task 2hop --model-type plt --checkpoint checkpoints/2hop_L2_seed0.pt --data data/boolean_2hop_improved.csv --n-features 128 --sparsity 0.0001 --epochs 20
```

Train the CLT-style baseline on the same original model and data:

```bash
python -m src.replacements.train --task 2hop --model-type clt --checkpoint checkpoints/2hop_L2_seed0.pt --data data/boolean_2hop_improved.csv --n-features 128 --sparsity 0.0001 --write-span 2 --epochs 20
```

Compare validation reconstruction fidelity after both runs:

```bash
python -m src.replacements.compare_fidelity --table results/person2/fidelity_table.csv --tolerance 0.02
```

The 0.02 absolute relative-L2 gap is only a provisional engineering threshold. It is not yet team-approved and should be decided before final experiment comparisons. Use validation metrics for tuning; reserve test metrics for final reporting.

## Outputs

The training command writes to `results/person2/` by default:

- one replacement `.pt` checkpoint per run;
- a JSON metadata record;
- per-epoch training history;
- an append/update `fidelity_table.csv`.

Do not commit large generated activation caches or checkpoints unless the repository's artifact policy allows it. The code currently captures activations in memory to keep the pilot simple.

## What remains before the main grid

- Integrate the Person 1 source/checkpoints into the working branch.
- Person 3 agreed to the MLP-input-to-MLP-write interface as the primary replacement target, with residual-stream block input/output captures as supplementary diagnostics. The minimal CLT-style implementation is still a project-specific baseline; it is not yet established as identical to a canonical external CLT implementation.
- Finalize the downstream graph-builder interface. The current Person 3 branch supplies graph utilities and a collapse metric, not a complete model-derived attribution-graph builder. Decoder weights must not be treated as attribution edges without an agreed attribution method.
- Agree a reconstruction-fidelity metric and matching tolerance before comparing graph-collapse results.
- Revisit Person 1's exploratory patching validation before treating the base checkpoints as fully validated scientific models.
- Only then scale beyond a one-checkpoint pilot.

Inspect the original Transformer and captured activation shapes before training replacements:

```bash
python -m src.replacements.inspect_activations --task 2hop --checkpoint checkpoints/2hop_L2_seed0.pt --data data/boolean_2hop_improved.csv --split test
```

Person 3 or another downstream consumer can reload a replacement model with `src.replacements.checkpoint_io.load_replacement_checkpoint(path)`; the saved metadata must be retained with the state dictionary.

## Current agreed interface implementation

Person 3 agreed to use the MLP input-to-MLP-write interface as the primary PLT/CLT target, with residual-stream block input/output captures as supplementary diagnostics. The updated `activation_cache.py` supports both. `inspect_activations.py` captures both interfaces; training captures only the primary MLP target to control memory use.

A local smoke-test bundle contains a copy of the original 2-hop seed-0 checkpoint so the user can run the first pilot. The checkpoint is ignored by Git in this Person 2 bundle to avoid accidentally duplicating shared base-model artifacts. Check the team branch/asset policy before merging checkpoints or generated replacement binaries.
