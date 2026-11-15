# Do Cross-Layer Transcoders Collapse Computational Depth?

**A certified depth sweep with Strict-IIT transformers, and a validation on GPT-2**

UE24AM342AA1 Advanced Foundations in Machine Learning, PES University. Team 20 (The Invariants):
Siri Krishna C R, D R Rushmitha, Shreya N Hegde, Shruti Choudhary.

---

## The question

Interpretability tools explain a model by drawing an **attribution graph**: which internal features pass
information to which, from the input to the output. Two tools are compared:

- **PLT (per-layer transcoder):** each feature writes only to the next layer.
- **CLT (cross-layer transcoder):** a feature can also write directly to any later layer ("skip" edges).

**Depth collapse** means the graph shows *fewer* computation steps than the model really uses. We ask whether
it grows with the true depth of the computation, whether it is specific to the CLT under a fair comparison, and
whether it appears on a real language model.

On a real model nobody knows the true steps, so we build models whose depth is **known and certified**, and check
the graphs against that ground truth.

## Main results

**Synthetic models (certified ground truth; transcoders at k = 8; 95% bootstrap CI over certified seeds)**

| True depth D | Certified models | DCR, PLT | DCR, CLT | DCR, param-matched PLT | Skip mass PLT / CLT |
|---|---|---|---|---|---|
| 1 | 5 / 5 | 1.000 | 1.000 | 1.000 | 0.000 / 0.000 |
| 2 | 5 / 5 | 0.728 [0.697, 0.758] | 0.750 [0.713, 0.788] | 0.714 | 0.145 / 0.188 |
| 3 | 4 / 5 | 0.657 [0.643, 0.672] | 0.643 [0.629, 0.657] | 0.646 | 0.274 / 0.382 |
| 4 | 1 / 5 | 0.503 | 0.509 | 0.501 | 0.327 / 0.480 |
| 5 | 0 / 5 | — | — | — | — |

DCR = D_feat / D, where D_feat is the influence-weighted number of feature steps on input-to-output paths in the
attribution graph. DCR = 1 means every step is shown.

1. **Graphs lose depth, and lose more as true depth grows:** DCR 1.00 → 0.73 → 0.66 → 0.50.
2. **PLT and CLT lose depth equally** (paired CLT − PLT: +0.045 at D = 2, −0.043 at D = 3; both p = 0.125).
3. **The CLT routes much more influence through layer-skipping edges**, in every certified seed. The
   parameter-matched PLT behaves like the PLT, so this comes from the cross-layer design, not extra parameters.
4. **These shortcuts are created by the replacements, not the model.** Logit-level bypass ratio at D = 3:
   original model 0.042, PLT 0.148, CLT 0.321.
5. **Decodable is not causally used:** 26 probe-decodable cases had no causal effect.

**GPT-2 small, two-hop factual recall** (10 two-hop and 12 one-hop prompts)

| Test | PLT | CLT |
|---|---|---|
| V1: D_feat(2-hop) − D_feat(1-hop), paired by state | **+0.277** [0.170, 0.384], p = 0.031 | +0.087 [−0.061, 0.221], p = 0.344 |
| V2: answer-logit drop, state features − random features | +0.83 [−0.85, 2.93], p = 0.59 | **+4.33** [2.27, 6.52], p = 0.016 |
| V3: Spearman(graph-claimed mediation, ablation drop) | 0.571, p = 0.18 | 0.643, p = 0.12 |
| V4: CLT − PLT, 2-hop D_feat / skip mass | | −0.160 [−0.345, −0.020], p = 0.059 / +0.073, p = 0.002 |

PLT graphs register the extra reasoning step; CLT graphs do not, and the CLT again uses more skip mass.
See **Limitations** before drawing conclusions.

All numbers above are generated from the CSV files in `results/` (see `paper/generated/`).

## Repository layout

```
configs/         run configurations (part_A_D1-3, part_B_D4, part_C_D5 = reported run; quick; full)
data/            exported synthetic datasets (D = 1..5) and the GPT-2 fact list
notebooks/       Colab notebooks: 01 synthetic pipeline, 02 GPT-2 validation
paper/           LaTeX source; paper/generated/ and paper/figures/ are produced by scripts
results/
  part_A/        raw outputs for D = 1, 2, 3   (certificates, fidelity, graph metrics, validation)
  part_B/        raw outputs for D = 4
  part_C/        raw outputs for D = 5 (certificates only: no model was certified)
  merged/        parts A + B + C combined, plus summary_*.csv
  gpt2_twohop/   GPT-2 prompts, transcoder fidelity, per-prompt graph metrics, validation tests
scripts/         run_pipeline.py, logit_bypass.py, merge_results.py, make_tables.py, make_figures.py, make_lm_tables.py
src/cdc/         the library
  tasks.py           Boolean multi-hop tasks with known intermediate values
  model.py           attention transformer with designated hop sites
  siit.py            Strict Interchange Intervention Training + depth certificate
  transcoders.py     PLT, CLT and parameter-matched PLT with TopK sparsity
  graphs.py          feature-to-feature attribution graphs with error nodes
  depth_metrics.py   D_feat, DCR, skip mass, hop profiles
  validation.py      logit bypass, graph-vs-ablation, probes vs interventions
  stats.py           bootstrap CIs, paired sign-flip tests, Holm correction
  pipeline.py        the stages: zoo → replacements → graphs → validate → probes
  realdata/          GPT-2 engine, two-hop validation (lm_twohop.py), InterpBench loader (not run)
tests/           23 automated tests (metric sanity, certificates, graph conservation, GPT-2 conversion, ...)
```

## How to run

**Requirements:** Python 3.10+, `pip install -r requirements.txt`. A GPU is needed for the full runs; a CPU is
enough for the tests and for regenerating every table and figure from the saved results.

**1. Check the installation (CPU, ~2 min)**
```bash
pip install -r requirements.txt
python -m pytest -q tests
```

**2. Regenerate all tables and figures from the saved results (CPU, seconds)**
```bash
python scripts/merge_results.py results/part_A results/part_B results/part_C --out results/merged
python scripts/make_tables.py  --results results/merged --paper paper/generated
python scripts/make_figures.py --results results/merged --out paper/figures
python scripts/make_lm_tables.py --results results/gpt2_twohop --paper paper/generated
```

**3. Re-run the experiments (GPU; we used free Colab T4s)**

Open `notebooks/01_synthetic_pipeline_colab.ipynb` and run it once per part (`PART = 'A'`, `'B'`, `'C'`),
then `notebooks/02_gpt2_twohop_colab.ipynb`. Or from a shell:

```bash
python scripts/run_pipeline.py --config configs/part_A_D1-3.yaml --stage all
python scripts/logit_bypass.py --results results/part_A --ckpt checkpoints/part_A
# same for part_B_D4 and part_C_D5, then merge (step 2)
PYTHONPATH=src python -m cdc.realdata.lm_twohop --model gpt2 --out results/gpt2_twohop
```
Approximate time on one T4: part A 1–2 h, part B 1–2 h, part C 1–2 h, GPT-2 1–1.5 h.
Running the `zoo` stage again retrains models that failed certification, with 2× and then 4× steps
(at most two retries).

`make test`, `make merge`, `make tables`, `make figures` and `make lm` wrap the same commands.

## Method in brief

1. **Tasks.** 16-bit inputs x0..xD; each hop combines the previous result with the next input, element-wise
   (D = 2: AND then XOR). The last hop is always XOR, because a final AND is linearly separable and can be
   computed by the readout without an MLP step. 5,000 / 1,000 / 1,000 train / val / test examples per depth.
2. **SIIT training.** Behaviour loss + interchange loss at each hop site + strictness loss (resampling
   non-circuit components must not change the output) + a bypass constraint; reserved site coordinates.
3. **Certification.** A model is analysed only if it passes all of: clean accuracy ≥ 0.99, interchange accuracy
   on flipped bits ≥ 0.98 at every hop, strictness ≥ 0.98, bypass ≥ 0.98 (also held out), circuit-only accuracy
   ≥ 0.99, every hop's MLP necessary, site separation ≥ 3.
4. **Replacements at an identical budget.** PLT and CLT with the same 128 features per layer, the same TopK k
   (exact number of active features), steps, data and seed; plus a PLT with extra features to match the CLT's
   parameter count.
5. **Graphs and metrics.** Feature-to-feature attribution graphs with error nodes, attention and LayerNorm
   frozen (maximum conservation error about 7e-6); D_feat, DCR, skip mass.
6. **Validation.** Logit-level bypass, graph-predicted vs actual ablation (Pearson r: PLT 0.750, CLT 0.539,
   param-matched PLT 0.621), and probes vs interventions.
7. **GPT-2.** The same graph engine on GPT-2 small (verified against Hugging Face's forward pass by a test),
   PLT and CLT with 2,048 features per layer and k = 32 trained on WikiText-103 (held-out NMSE PLT 0.314,
   CLT 0.239), and the tests V1–V4 above.

## Data

- **Synthetic:** generated by `src/cdc/tasks.py` with a fixed seed; exported to `data/boolean_D1.csv` …
  `boolean_D5.csv` (inputs, every intermediate value, output, split).
- **GPT-2 small:** https://huggingface.co/openai-community/gpt2
- **WikiText-103** (`wikitext-103-raw-v1`): https://huggingface.co/datasets/Salesforce/wikitext
- **Facts:** `data/gpt2_city_state_capital.csv`, 51 city → state → capital triples over 20 US states.

## Limitations

- Certification gets harder with depth: 1 of 5 models at D = 4 and 0 of 5 at D = 5. The second retry was not
  run for these, so D = 4 results rest on a single model and D = 5 is excluded from the depth analysis.
- With 4–5 seeds per depth, a paired sign-flip test cannot reach p < 0.05 (smallest possible p is 0.0625).
- Reduced grid to fit the compute budget: L = D layers only, k ∈ {4, 8, 16}, 100 graph examples per model
  (`configs/full.yaml` holds the larger grid).
- GPT-2: GPT-2 small answers few capital questions, so prompts were kept if the answer ranked in the top 50
  (with top 10, no two-hop prompt survives); only 10 two-hop prompts; the unsigned depth measure does not
  reproduce V1; the transcoders are small (NMSE 0.24–0.31).
- InterpBench models were not evaluated (`src/cdc/realdata/interpbench_eval.py` is a starting point).

## References

1. Ameisen, Lindsey, et al. *Circuit Tracing: Revealing Computational Graphs in Language Models.* Anthropic, 2025.
2. *Cross-Layer Transcoders are Incentivized to Learn Unfaithful Circuits.* ICML 2026.
3. Gupta, Arcuschin, Kwa, Garriga-Alonso. *InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques.* NeurIPS 2024.
4. Dunefsky, Chlenski, Nanda. *Transcoders Find Interpretable LLM Feature Circuits.* NeurIPS 2024.
5. Geiger, Wu, Lu, et al. *Inducing Causal Structure for Interpretable Neural Networks.* ICML 2022.
6. Radford, Wu, Child, et al. *Language Models are Unsupervised Multitask Learners.* 2019.
7. Merity, Xiong, Bradbury, Socher. *Pointer Sentinel Mixture Models.* ICLR 2017.
