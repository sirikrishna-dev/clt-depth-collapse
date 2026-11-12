# Person 1: ground-truth depth (v2)

## Run it (Windows, from the repo root)

```cmd
pip install -r requirements.txt
python -m pytest -q tests/test_gt_forward.py tests/test_stats_utils.py
python scripts/ground_truth_depth.py
python scripts/make_person1_tables.py > results/v2_matched/person1/TABLES.md
```

The full run takes about a minute on a laptop CPU. Outputs go to `results/v2_matched/person1/`:

| File | Contents |
|---|---|
| `gt_sanity.csv` | Bit, exact-vector and balanced accuracy; class balance |
| `gt_ablation.csv` | MLP and attention ablations (resample ×3 seeds, mean, zero) and block skip |
| `gt_decodability.csv` | Probes for u, v, y after every block |
| `gt_y_shortcut.csv` | Is the early y signal **used**? (probe-direction swap + random control) |
| `gt_siit_interchange_chain_bypass.csv` | SIIT u/v interchange (corrected c = 1 protocol and the legacy one), chain, bypass |
| `gt_dtrue.csv` | `D_true` per model with the reason and evidence |
| `gt_config.json` | Thresholds, seeds, git commit, versions |
| `TABLES.md` | Paper tables generated from the CSVs |

## New code

- `src/gt_forward.py`: an explicit forward pass with intervention points. `nn.TransformerEncoderLayer` can run a fused fast path in eval mode, where hooks on its submodules never fire. `tests/test_gt_forward.py` checks this forward pass equals `model(x)` on all 16 checkpoints.
- `src/stats_utils.py`: Wilson CI, paired t-interval, bootstrap, sign test, and the frozen decision rule (`decide`), with tests.

## Findings (numbers: see `TABLES.md`)

1. **Necessary is not the same as computing.** In every model, every MLP layer is necessary (resample ablation flips ≥ 19% of bits). But in the 2-hop baselines, swapping only the y direction after block 0 makes the output follow the donor on 100% of changed bits. Those models compute y in one step, and block 1 only carries it. So `D_true = 1`, although the network has 2 layers and both are "necessary".
2. **Decodable is not the same as used.** y is linearly decodable after block 0 in all 16 models (balanced accuracy ≥ 0.999). In SIIT models, the same swap moves the output 0% of the time: the signal is there but unused.
3. **Ground-truth depth now spans 1, 2 and 3:** 2-hop baselines 1; 2-hop SIIT 2; 3-hop baselines 2 (seed 1 unresolved: partial shortcut 0.46); 3-hop SIIT 3.
4. **The old v-test problem is fixed.** With the corrected protocol (donor and recipient both c = 1), v interchange is 1.000 on all 5 seeds (n = 1,895 each). The legacy protocol reproduces the old 0.903 and 0.624.
5. **The designated coordinates are not the only channel.** In 3-hop SIIT, ablating MLP1 and restoring only the v coordinate undoes 18–40% of the damage. The intermediate information also travels in other dimensions. So `exclusive_depth_verified` stays false, and Person 3 must assign u/v roles to features by ablation, not by coordinate.

## Consequences for the team

- **Person 2:** add the 6 baselines to the PLT/CLT grid (cheap, secondary analysis). Then depth can be compared across models with `D_true` 1, 2 and 3, a stronger trend than 2 vs 3 hops.
- **Person 3:** chain-consistent mass must use roles found by ablation (finding 5).
- **Paper:** finding 1 is a methodological contribution in its own right: counting necessary layers overestimates depth.

## Likely examiner questions

- *How do you know the 3-hop SIIT models use 3 steps?* All three MLPs are necessary. No early y signal is used (0% at blocks 0 and 1). The u → v → y chain holds at 1.000 with Wilson lower bound 0.999.
- *One MLP can compute these functions; isn't depth artificial?* Yes, and the 2-hop baselines show it: they do it in one step. SIIT imposes depth, and we measure it rather than assume it.
- *Why resample ablation?* Zero ablation is off-distribution. All three variants are reported in `gt_ablation.csv`.
- *Couldn't the probe swap fail simply because the probe direction is wrong?* The same probe direction succeeds at 100% in the 2-hop baselines, and the random-direction control is 0 everywhere. So the test can detect a shortcut, and when it says "not used" that is informative.
- *What is the denominator of "1.000"?* Only bits whose target changes. Every CSV row has `k`, `n` and a 95% Wilson interval.
