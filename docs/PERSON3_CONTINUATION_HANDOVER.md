# Person 3 continuation – handover (13 Nov 2026)

Branch: `person3-multiseed-results` (built on top of `integration/clean-main`).
Research question: do Cross-Layer Transcoders (CLTs) make multi-step computation look
shallower in attribution graphs than Per-Layer Transcoders (PLTs) ("depth collapse")?

**Short status:** the original single-seed pilot was recovered and extended to all five
SIIT seeds of both tasks (≈860 single-feature interventions). 2-hop shows no PLT/CLT
difference. 3-hop shows a small, seed-consistent difference for Layer-1 `v` features that
is *consistent with* CLTs routing influence through cross-layer shortcuts. Depth collapse
has **not** yet been measured directly: the shortcut test is ready to run, and graph
construction/depth measurement has not been started.

---

## 1. What happened before this branch

* The earlier handover report (Word doc, "CLT_PLT_Research_Handover_Report") described a
  seed-0, 2-hop pilot. Its code and outputs were never pushed and were lost with a laptop.
* `evaluate_individual_features.py`, the `--write-scale` version of
  `run_person3_causal_intervention.py`, and `tests/test_zero_write_intervention.py` were
  **reconstructed** (not byte-identical to the originals).
* Verification: the reconstructed scripts reproduce every seed-0 number in the report
  exactly (PLT L0 F115 and CLT L0 F20; scales 1.0 / 0.776 / 0.5; selected features and
  decodability counts). The full SHA-256 of `checkpoints/2hop_L2_siit_k1_seed0.pt`
  (`DCA9E0A4…C14770`) matches the report, so the checkpoint caveat is closed.
* Exact magnitude-matching scale for the pilot = 0.0196005 / 0.0252558 = **0.77608**
  (the report's 0.776 was effectively exact).
* Known weakness: `tests/test_zero_write_intervention.py` tests a toy `nn.Sequential`,
  not the real `BitVectorTransformer`. It should be rewritten to hook the real model.

## 2. Files added or changed on this branch

| Path | Purpose |
|---|---|
| `src/replacements/evaluate_individual_features.py` | Reconstructed single-feature decodability screen. Added `--variable {u,v}`. Ties broken with `rank(method="first")`. |
| `src/metrics/run_person3_causal_intervention.py` | Reconstructed intervention script with `--write-scale`. Output names use `0p776` instead of `0.776`. |
| `tests/test_zero_write_intervention.py` | Reconstructed zero-write test (toy model – see above). |
| `scripts/run_feature_batch.py` | Runs the intervention for every selected feature of a seed and writes a summary CSV. Options: `--source-layer`, `--min-val-acc`, `--max-per-model`, `--target-layers` (CLT shortcuts), `--models`. Records every layer's change and the logit change. Reuses finished runs. |
| `scripts/combine_batches.py` | Combines seeds; per-seed Welch tests, seed-level paired test (main), pooled tests, prediction check, and **section 5: push-size-adjusted comparison** (OLS: metric ~ CLT + log(write size) [+ seed intercepts]). Option `--source-layer`. |
| `scripts/shortcut_compare.py` | Shortcut test analysis (ready, not yet run on real data – see §5). |
| `results/person3/…` | All decodability CSVs and intervention outputs (see §4). |

All scripts are run from the repository root, e.g. `python scripts/combine_batches.py --task 2hop`.

## 3. Method in one paragraph

For each base SIIT model and each replacement (PLT/CLT), every feature is screened for how
well it alone predicts an intermediate variable (`u` or `v`): one scalar logistic probe
per bit position, fit on train, ranked on validation. Selected features are then ablated
in the replacement model; the resulting decoder-write difference is injected into the
original Transformer's `linear2` output at the source layer only (`--target-layers` =
source, i.e. CLT cross-layer decoders **off**), scale 1.0. Downstream change is measured
as the mean absolute change of replacement features at later layers (read at each
layer's MLP input), divided by the mean absolute injected write ("change per unit write").
Because this ratio falls as write size grows (strong negative correlation, e.g. r = −0.42
for 3-hop `v`), the main comparison controls for log(write size).

## 4. Results

### 4.1 Decodability (screening only – not causal)

* **2-hop `u`:** seed 0 has one perfect Layer-0 feature per model (PLT F115, CLT F20);
  seeds 1–4 have many (7–16). Decodability is saturated and cannot separate PLT and CLT.
* **3-hop `u`:** in seeds 0, 1, 2, 4 no single Layer-0 feature exceeds ≈0.75–0.87; 0.75 is
  exactly what an OR- or AND-like feature scores on XOR, so Layer-0 features look like
  *components* of the XOR computation. Many Layer-1 features decode `u` perfectly.
  Seed 3 is the exception (one perfect Layer-0 feature in each model).
* **3-hop `v`** (majority baseline ≈0.755): Layer 0 best 0.84–0.97 (hints only); Layers
  1 and 2 have many perfect features.
* PLT and CLT show the same layer-by-layer pattern in every seed; no sign of the CLT
  representing `u` or `v` earlier than the PLT.

### 4.2 Interventions (outputs in `results/person3/causal_batch/`)

| Experiment | Runs | Raw seed-level paired test | **Push-size-adjusted** (seed-level) | Verdict |
|---|---|---|---|---|
| 2-hop, L0 `u` features → L1 | 117 | PLT−CLT −0.0027, p = 0.84, PLT higher 2/5 | −0.0005, p = 0.97, 2/5 | No difference |
| 3-hop, L0 features → L1 | 150 | +0.019, p = 0.18, 4/5 | +0.022, p = 0.24, 3/5 | Inconclusive |
| 3-hop, L0 features → L2 | 150 | +0.014, p = 0.27, 4/5 | +0.013, p = 0.27, 4/5 (seeds 0,1 drive it) | Inconclusive |
| 3-hop, L1 `v` features → L2 | 447 | +0.013, p = 0.09, 4/5 | **+0.018 (~11%), p = 0.025, PLT higher 5/5**, Wilcoxon p = 0.062 | Consistent PLT > CLT |

* No intervention changed any prediction (0 of ~860 runs; accuracy stayed 1.0).
* The pooled OLS with seed intercepts gives smaller p-values (e.g. p < 0.0001 for 3-hop
  `v`), but it treats features as independent; use the seed-level test as the headline.
* Four comparisons were made; with a Bonferroni threshold (0.0125) the 3-hop `v` result
  is suggestive, not conclusive.
* The 3-hop L0 runs used `--min-val-acc 0.75 --max-per-model 15` (no perfect Layer-0 `u`
  features exist in most seeds), so the selected features are OR/AND-like pieces; in
  seed 3 they are genuine `u` features. Keep this in mind when interpreting.

### 4.3 Interpretation

With CLT cross-layer decoders switched off, CLT `v` features propagate ~11% less effect
along the layer-by-layer route than PLT features of equal push size, in all five 3-hop
seeds, and not at all in 2-hop. This is consistent with (not proof of) the CLT moving
part of each feature's downstream influence into its cross-layer shortcuts – the
mechanism by which an attribution graph could look shallower.

## 5. Next steps (in order)

1. **Shortcut test (scripts ready, not yet run).** Re-run the 3-hop Layer-0 CLT features
   with the 0→1 shortcut switched on and compare with the normal runs:
   ```
   # once per seed s = 0..4 – rebuild the normal summaries so they include the logit column (reuses saved runs)
   python scripts/run_feature_batch.py --task 3hop --seed s --source-layer 0 --analysis results/person3/3hop_u_features_seed{s}.csv --min-val-acc 0.75 --max-per-model 15
   # CLT only, shortcut 0->1 on
   python scripts/run_feature_batch.py --task 3hop --seed s --source-layer 0 --analysis results/person3/3hop_u_features_seed{s}.csv --min-val-acc 0.75 --max-per-model 15 --models clt --target-layers 0,1
   # after all seeds
   python scripts/shortcut_compare.py --task 3hop --source-layer 0 --targets 0,1
   ```
   Measurement caveat: features at layer k are read from layer k's MLP *input*, so a
   shortcut written into layer k's MLP *output* is invisible at layer k but visible at
   later layers and in the logits. The 0→1 shortcut is visible at Layer 2 and the logits;
   a 1→2 shortcut (for `v` features: `--source-layer 1 --targets 1,2`) is visible only in
   the logits. Prediction: if the CLT's missing Layer-2/logit effect returns when the
   shortcut is on, that supports routing through shortcuts.
2. **Define and validate feature-to-feature edges.** Decide the edge definition (e.g.
   intervention-based effect of source feature on target feature, with a threshold and
   controls such as random/shuffled features), validate on the known SIIT locations of
   `u` (L0) and `v` (L1).
3. **Build graphs and measure depth** for PLT and CLT (with CLT cross-layer edges
   included), using `src/metrics/graph_depth_analysis.py` only after step 2. Compare path
   length from inputs to output across seeds.
4. **Realistic-data/model validation** (mentor's central concern: the Boolean task may be
   too simple).
5. Housekeeping: rewrite the zero-write test against the real model; consider a matched-
   scale check (rerun selected CLT features with `--write-scale` set so their write size
   equals the PLT mean) to confirm the push-size adjustment.

## 6. Reproduction notes

* Windows, Python 3.14, PyTorch 2.13 (CPU), run from the repo root with `python -m …` /
  `python scripts/…`.
* 3-hop replacement checkpoints are under
  `results/person2/siit/3hop_L3_siit_k1_seed{s}/3hop_{plt|clt}_seed0_features128_span{1|3}.pt`
  (the "seed0" in the file name is the transcoder's own seed; the folder gives the base seed).
* Decodability screens take ~10–20 min per run on CPU; batch interventions take a few
  seconds per feature.
