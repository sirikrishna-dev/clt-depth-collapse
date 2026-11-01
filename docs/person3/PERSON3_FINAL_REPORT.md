\# Person 3 – Experimental Evaluation and Causal Attribution Analysis



\## 1. Introduction



This work focuses on evaluating the internal representations and feature interactions of PLT and CLT replacement models trained on Boolean computation tasks.



The experiments use 2-hop and 3-hop Transformer models across five independent random seeds. The main objective is to study how intermediate variables are represented across different layers, understand the decoder-derived feature connections, and investigate how feature-level write interventions affect downstream representations.



The evaluation combines linear probing, shuffled-label controls, attribution graph analysis, and controlled write interventions. These experiments provide evidence about representation availability and sensitivity to feature interventions.



\## 2. Objectives and Scope



The main objectives of this work are:



1\. Evaluate whether intermediate Boolean variables can be linearly decoded from different model representations.

2\. Use shuffled-label controls to check whether probe performance depends on meaningful representation information.

3\. Construct and compare PLT and CLT attribution graphs.

4\. Examine same-layer and cross-layer decoder contribution patterns.

5\. Perform feature-level decoder-write interventions and measure downstream representation changes.

6\. Compare PLT and CLT intervention effects across five independent seeds.

7\. Document the experimental findings, limitations, and reproducibility details.



The scope is limited to the supplied Boolean tasks, trained checkpoints, replacement models, and the implemented evaluation procedures.



\## 3. Experimental Setup



\### 3.1 Models and Checkpoints



The evaluation uses ten original SIIT Transformer checkpoints:



\- Five 2-hop models: seeds 0–4.

\- Five 3-hop models: seeds 0–4.



Each original checkpoint has corresponding PLT and CLT replacement checkpoints.



The 2-hop Transformer contains two layers, while the 3-hop Transformer contains three layers. The models use a hidden dimension of 64 and four attention heads.



The replacement models use 128 features. PLT uses a write span of 1, while CLT uses a span of 2 for the 2-hop task and 3 for the 3-hop task.



\### 3.2 Datasets and Tasks



Two Boolean computation tasks were evaluated.



\*\*2-hop task:\*\*



\- First operation: u = XOR(a, b)

\- Second operation: y = AND(u, c)



\*\*3-hop task:\*\*



\- First operation: u = XOR(a, b)

\- Second operation: v = AND(u, c)

\- Third operation: y = XOR(v, d)



Each dataset contains 7,000 examples:



\- 5,000 training examples

\- 1,000 validation examples

\- 1,000 test examples



The experiments use the test split for evaluation.



\### 3.3 PLT and CLT Configurations



PLT and CLT replacement checkpoints were evaluated against their corresponding original Transformer checkpoints.



PLT uses same-layer decoder writes, whereas CLT supports decoder writes to multiple destination layers within its configured span.



The evaluation examines intermediate representations, decoder contribution effects, and the changes produced by controlled feature-level interventions.



All five seeds are treated as independent model runs. Results are aggregated across seeds to examine consistency and variation.





\## 4. SIIT Intermediate Variable Evaluation



\### 4.1 Probe Evaluation



Linear probing was performed to evaluate whether the intermediate Boolean variables could be decoded from different model representations.



The evaluation covered the original SIIT models and their corresponding PLT and CLT representations across five independent seeds.



For the 2-hop task, the intermediate variable u was highly linearly decodable across the evaluated representations. Its test accuracy was approximately 100% across the five seeds.



For the 3-hop task, both intermediate variables u and v were evaluated. The MLP input at layer L0 showed some variation across seeds.



The following table presents the 3-hop MLP input L0 probe accuracies:



| Seed | Variable u | Variable v |

|---|---:|---:|

| 0 | 63.02% | 99.33% |

| 1 | 99.79% | 99.81% |

| 2 | 99.92% | 99.86% |

| 3 | 100.00% | 100.00% |

| 4 | 98.08% | 99.31% |



The results indicate that the intermediate variables are generally highly decodable from the evaluated representations. However, seed 0 shows lower decodability for u at the 3-hop L0 MLP input.



These results demonstrate the availability of information in the representations. High probe accuracy alone does not establish that a particular layer actually performs the corresponding Boolean operation.



\### 4.2 Shuffled-Label Controls



Shuffled-label controls were performed to check whether the probe results depended on meaningful information in the representations.



Twenty shuffled-label runs were performed for each evaluated probe configuration. The original probe performance was compared with the resulting shuffled-label accuracy distribution.



The shuffled-label control accuracies were generally around 50–52% across the evaluated configurations.



The empirical p-value was calculated using:



p = (1 + number of shuffled scores greater than or equal to the original score) / (number of shuffles + 1)



With twenty shuffles, the smallest possible empirical p-value is approximately 0.0476.



Therefore, these controls provide an exploratory comparison against shuffled labels, but the limited number of shuffles does not support strong statistical significance claims.



\### 4.3 Results Across Five Seeds



The multi-seed evaluation showed that intermediate-variable information was generally available across different model representations.



The main observations were:



1\. The 2-hop intermediate variable u achieved approximately 100% probe accuracy across the evaluated representations.

2\. The 3-hop intermediate variables u and v were also highly decodable in most evaluated representations.

3\. The 3-hop L0 MLP input showed seed-dependent variation, particularly for u.

4\. Shuffled-label controls generally produced accuracy values around 50–52%.

5\. Probe performance should be interpreted as evidence of linear information availability rather than direct evidence of computation at a specific layer.



Overall, the probing experiments establish that intermediate Boolean information is accessible in many of the evaluated representations. Further causal evidence is needed to determine how this information participates in the model's computation.





\## 5. Attribution Graph Analysis



\### 5.1 Methodology



Attribution graphs were generated for the PLT and CLT replacement models across all ten original Transformer checkpoints.



The analysis covered five seeds for each task, resulting in 20 attribution graphs: ten PLT graphs and ten CLT graphs.



The graph analysis measures decoder contribution effects by examining the change in destination writes when source features are ablated.



For each source-to-destination connection, the mean decoder contribution effect was calculated across the evaluated examples and aggregated across five seeds.



The graph edges represent decoder-derived contribution effects. They are not treated as verified causal edges.



\### 5.2 PLT Results



PLT showed same-layer connections in the evaluated models.



For the 2-hop task, the observed connections were:



\- L0 → L0: mean effect 0.02816

\- L1 → L1: mean effect 0.02876



For the 3-hop task, the observed connections were:



\- L0 → L0: mean effect 0.03060

\- L1 → L1: mean effect 0.03205

\- L2 → L2: mean effect 0.03081



No cross-layer decoder connections were present in the evaluated PLT configurations.



\### 5.3 CLT Results



CLT showed both same-layer and cross-layer decoder contribution connections.



For the 2-hop task:



| Connection | Mean effect | Standard deviation |

|---|---:|---:|

| L0 → L0 | 0.02737 | 0.00148 |

| L0 → L1 | 0.02294 | 0.00134 |

| L1 → L1 | 0.02268 | 0.00131 |



For the 3-hop task:



| Connection | Mean effect | Standard deviation |

|---|---:|---:|

| L0 → L0 | 0.03118 | 0.00339 |

| L0 → L1 | 0.02242 | 0.00203 |

| L0 → L2 | 0.02146 | 0.00168 |

| L1 → L1 | 0.02808 | 0.00221 |

| L1 → L2 | 0.02511 | 0.00091 |

| L2 → L2 | 0.02378 | 0.00210 |



\### 5.4 Across-Seed Comparison



The attribution analysis showed a consistent structural difference between the two replacement configurations.



PLT contained same-layer decoder contribution connections, while CLT additionally contained cross-layer connections.



The additional CLT connections were:



\- 2-hop: L0 → L1

\- 3-hop: L0 → L1, L0 → L2, and L1 → L2



These connection patterns were observed across all five seeds.



The relatively small standard deviations for most connections indicate that the measured contribution patterns were reasonably consistent across the five independently trained replacement models.



The results demonstrate that CLT supports additional cross-layer decoder contributions compared with PLT in the evaluated configurations. However, decoder contribution effects alone do not establish causal bypass, computational depth reduction, or depth collapse.



\*\*Figure 3.\*\* PLT and CLT attribution graph contribution effects across five seeds. Error bars represent standard deviation across seeds.

\## 6. Causal Write Intervention Experiments



\### 6.1 Intervention Method



Controlled feature-level decoder-write interventions were performed to examine how removing a selected replacement feature affects downstream representations of the original Transformer.



Feature L0 F22 was selected for the multi-seed experiments.



The intervention procedure was:



1\. Load the original Transformer checkpoint and its corresponding PLT or CLT replacement checkpoint.

2\. Capture baseline MLP inputs from the original Transformer using the test dataset.

3\. Calculate replacement feature activations from the captured MLP inputs.

4\. Set the selected source feature activation to zero.

5\. Calculate the difference between the original and modified decoder writes.

6\. Inject the resulting write difference into the corresponding original Transformer MLP output.

7\. Capture modified downstream MLP inputs.

8\. Measure changes in downstream replacement feature activations, model logits, and predictions.



The experiment was conducted on the test split.



\### 6.2 Initial Multi-Seed Results



The initial intervention experiments covered all ten original Transformer checkpoints, with both PLT and CLT replacements.



This resulted in 20 experiments:



\- Five 2-hop PLT runs.

\- Five 2-hop CLT runs.

\- Five 3-hop PLT runs.

\- Five 3-hop CLT runs.



For the initial CLT experiments, all permitted destination layers were selected. For PLT, only the corresponding same-layer destination was available.



The intervention results showed measurable downstream feature changes, with variation across seeds.



For the 2-hop task, the mean downstream feature changes were:



| Model | Destination | Mean feature change |

|---|---|---:|

| CLT | L1 | 0.005277 |

| PLT | L1 | 0.004306 |



For the 3-hop task:



| Model | Destination | Mean feature change |

|---|---|---:|

| CLT | L1 | 0.006968 |

| CLT | L2 | 0.006731 |

| PLT | L1 | 0.002555 |

| PLT | L2 | 0.001920 |



All 20 experiments retained 100% baseline and intervened bit accuracy, with zero prediction changes.



\### 6.3 Matched PLT and CLT Comparison



To make the comparison more controlled, additional CLT experiments were performed with the destination restricted to L0.



This allowed the CLT intervention to be compared with PLT using the same source layer, selected feature, and destination layer.



The matched experiments covered five seeds for each task, resulting in ten additional CLT runs.



The matched comparison produced the following mean downstream feature changes:



| Task | Destination | CLT | PLT |

|---|---|---:|---:|

| 2-hop | L1 | 0.005277 | 0.004306 |

| 3-hop | L1 | 0.006968 | 0.002555 |

| 3-hop | L2 | 0.003611 | 0.001920 |



Across all three comparisons, CLT showed a larger mean downstream feature change than PLT.



The changed-feature fractions also differed between the models. In the 2-hop task, they were similar, while in the 3-hop task, CLT showed a larger changed-feature fraction.



All matched experiments retained 100% intervened accuracy and zero prediction changes.



These results indicate greater downstream representation sensitivity under the tested CLT intervention. They do not establish that CLT generally has greater causal influence.



\### 6.4 Interpretation and Experimental Limitations



The intervention results show that decoder-write changes derived from the selected replacement feature can affect downstream representations in the original Transformer.



However, the intervention is a controlled transcoder-derived write intervention. It is not equivalent to removing an intrinsic feature from the original Transformer and allowing the entire replacement model to evolve naturally.



The selected feature F22 is held fixed across seeds for replication, but feature identities are not guaranteed to have equivalent learned meanings across independently trained replacement models.



The absence of final prediction changes indicates that the measured internal representation changes did not alter the model's thresholded predictions on the evaluated test examples.



Therefore, the results support downstream representation sensitivity but do not independently prove causal depth collapse or computational bypass.





\## 7. Visualizations



Three visualizations were generated to summarize the experimental findings.



\### Figure 1: PLT vs CLT Downstream Feature Changes



File: `results/person3/final\_figures/01\_plt\_vs\_clt\_feature\_change.png`



This figure compares the mean downstream feature changes produced by PLT and CLT interventions across the evaluated tasks and destination layers.



\### Figure 2: Seed-wise Variation



File: `results/person3/final\_figures/02\_seed\_wise\_variation.png`



This figure shows the variation in downstream feature changes across five independent random seeds. It highlights differences in intervention effects between independently trained replacement models.



\### Figure 3: Attribution Graph Comparison



File: `results/person3/final\_figures/03\_attribution\_graph\_comparison.png`



This figure compares decoder contribution effects for PLT and CLT across the 2-hop and 3-hop tasks. Error bars represent standard deviation across five seeds. Missing bars indicate structurally unavailable connections, not measured zero effects.



\## 8. Key Findings



The experiments produced the following findings:



1\. Intermediate Boolean variables were generally highly linearly decodable across the evaluated model representations.

2\. Shuffled-label controls generally produced accuracy values around 50–52%, supporting the presence of meaningful information in the evaluated representations.

3\. PLT attribution graphs contained same-layer decoder contribution connections.

4\. CLT attribution graphs contained additional cross-layer decoder contribution connections across all five seeds.

5\. Feature-level decoder-write interventions produced measurable downstream representation changes.

6\. In the matched L0-only comparison, CLT showed larger mean downstream feature changes than PLT in all three task-layer comparisons.

7\. None of the 20 initial interventions or 10 additional matched CLT interventions changed final predictions. Intervened accuracy remained 100%.



Together, these results show that intermediate information is accessible in model representations and that decoder-derived write interventions can affect downstream representations without changing the final predictions in the evaluated test examples.



\## 9. Limitations



The experiments have several limitations:



1\. Linear probe accuracy indicates information availability but does not prove that a specific layer performs the corresponding computation.

2\. Shuffled-label controls used 20 repetitions, limiting the resolution of empirical p-values.

3\. Attribution graph edges represent decoder-derived contribution effects, not verified causal relationships.

4\. The causal intervention is a controlled transcoder-derived write intervention rather than an intrinsic feature ablation in the original Transformer.

5\. Feature F22 was selected consistently across seeds, but feature identities may not have equivalent meanings across independently trained models.

6\. The intervention experiments produced no final prediction changes, so their effects are demonstrated through internal representation and logit changes rather than changes in task accuracy.

7\. The results are limited to the evaluated Boolean tasks, checkpoints, and experimental configurations.



Therefore, the findings should not be interpreted as definitive proof of computational depth collapse or causal bypass.



\## 10. Conclusion



This work evaluated intermediate-variable representations, decoder contribution patterns, and feature-level intervention effects in PLT and CLT replacement models.



The probing experiments showed that intermediate Boolean variables were generally accessible through linear probes. Shuffled-label controls provided an exploratory check that the results depended on information in the representations.



Attribution graph analysis showed that CLT supported additional cross-layer decoder contributions compared with PLT in the evaluated configurations.



Controlled feature-level interventions demonstrated measurable downstream representation changes. In the matched comparison, CLT produced larger mean downstream feature changes than PLT across the three evaluated task-layer comparisons. However, final predictions remained unchanged in all experiments.



Overall, the experiments provide evidence about representation availability, decoder contribution structure, and downstream representation sensitivity. Further causal validation is required before making stronger claims about computational depth collapse.



\## 11. Reproducibility and Output Files



The main experimental outputs are organized under `results/person3/`.



\### SIIT Evaluation



\- `results/person3/siit\_evaluation/`

\- `results/person3/siit\_evaluation\_multishuffle/`

\- `results/person3/multiseed\_probe\_summary.csv`



\### Attribution Graph Analysis



\- `results/person3/attribution\_multiseed/`

\- `results/person3/attribution\_multiseed/all\_seed\_graph\_summary.csv`

\- `results/person3/attribution\_multiseed/mean\_std\_across\_seeds.csv`



\### Initial Causal Interventions



\- `results/person3/causal\_multiseed/`

\- `results/person3/causal\_multiseed/causal\_all\_seed\_results.csv`

\- `results/person3/causal\_multiseed/causal\_mean\_std\_summary.csv`



\### Matched Causal Interventions



\- `results/person3/causal\_matched/`

\- `results/person3/causal\_matched/matched\_clt\_all\_results.csv`

\- `results/person3/causal\_matched/matched\_clt\_summary.csv`

\- `results/person3/causal\_matched/plt\_clt\_matched\_all\_seeds.csv`

\- `results/person3/causal\_matched/plt\_clt\_matched\_summary.csv`



\### Final Visualizations



\- `results/person3/final\_figures/01\_plt\_vs\_clt\_feature\_change.png`

\- `results/person3/final\_figures/02\_seed\_wise\_variation.png`

\- `results/person3/final\_figures/03\_attribution\_graph\_comparison.png`



\### Scripts



\- `src/replacements/evaluate\_siit\_intermediates.py`

\- `src/metrics/run\_person3\_attribution.py`

\- `src/metrics/run\_person3\_causal\_intervention.py`

\- `scripts/aggregate\_causal\_results.py`

\- `scripts/aggregate\_matched\_causal.py`

\- `scripts/compare\_matched\_plt\_clt.py`

\- `scripts/create\_person3\_visualizations.py`
