## Table 4.1 Sanity and class balance (test split)

| group          |   models |   bit_acc_min |   balanced_acc_min |   frac_y1 |   majority_baseline |
|:---------------|---------:|--------------:|-------------------:|----------:|--------------------:|
| 2-hop SIIT     |        5 |             1 |                  1 |     0.25  |               0.75  |
| 2-hop baseline |        3 |             1 |                  1 |     0.25  |               0.75  |
| 3-hop SIIT     |        5 |             1 |                  1 |     0.493 |               0.507 |
| 3-hop baseline |        3 |             1 |                  1 |     0.493 |               0.507 | 

## Table 4.2 MLP resample ablation (mean over 3 seeds, then range across models)

|                       |   flip_min |   flip_max |   ba_drop_min |   ba_drop_max |
|:----------------------|-----------:|-----------:|--------------:|--------------:|
| ('2-hop SIIT', 0)     |      0.191 |      0.251 |         0.29  |         0.336 |
| ('2-hop SIIT', 1)     |      0.377 |      0.377 |         0.502 |         0.502 |
| ('2-hop baseline', 0) |      0.293 |      0.345 |         0.446 |         0.481 |
| ('2-hop baseline', 1) |      0.377 |      0.377 |         0.502 |         0.502 |
| ('3-hop SIIT', 0)     |      0.252 |      0.296 |         0.252 |         0.296 |
| ('3-hop SIIT', 1)     |      0.259 |      0.497 |         0.259 |         0.497 |
| ('3-hop SIIT', 2)     |      0.493 |      0.493 |         0.493 |         0.493 |
| ('3-hop baseline', 0) |      0.38  |      0.425 |         0.38  |         0.426 |
| ('3-hop baseline', 1) |      0.493 |      0.493 |         0.493 |         0.493 | 

## Table 4.3 Decodable vs used: y after block 0

| group          | y probe balanced acc (min-max)   | output follows donor y (min-max)   |   random-direction control (max) |
|:---------------|:---------------------------------|:-----------------------------------|---------------------------------:|
| 2-hop SIIT     | 1.0-1.0                          | 0.0-0.0                            |                                0 |
| 2-hop baseline | 1.0-1.0                          | 1.0-1.0                            |                                0 |
| 3-hop SIIT     | 0.999-1.0                        | 0.0-0.0                            |                                0 |
| 3-hop baseline | 1.0-1.0                          | 0.015-0.46                         |                                0 | 

## Table 4.4 SIIT designated-site tests (range across seeds; score [95% Wilson CI] (n))

| group      | test          | protocol                               | worst_seed                    | best_seed                     |
|:-----------|:--------------|:---------------------------------------|:------------------------------|:------------------------------|
| 2-hop SIIT | bypass        | ablate_mlp0_restore_u_site             | 0.930 [0.921, 0.937] (n=3634) | 1.000 [0.999, 1.000] (n=3951) |
| 2-hop SIIT | interchange_u | control_coord32_changed_bits           | 0.000 [0.000, 0.001] (n=4113) | 0.000 [0.000, 0.001] (n=4113) |
| 2-hop SIIT | interchange_u | random_pairs_changed_bits              | 1.000 [0.999, 1.000] (n=4113) | 1.000 [0.999, 1.000] (n=4113) |
| 3-hop SIIT | bypass        | ablate_mlp0_restore_u_site             | 0.441 [0.426, 0.456] (n=4113) | 0.967 [0.961, 0.972] (n=4039) |
| 3-hop SIIT | bypass        | ablate_mlp0_restore_v_site             | 0.479 [0.463, 0.494] (n=4113) | 0.966 [0.960, 0.971] (n=4039) |
| 3-hop SIIT | bypass        | ablate_mlp1_restore_v_site             | 0.179 [0.167, 0.191] (n=4126) | 0.401 [0.390, 0.412] (n=7875) |
| 3-hop SIIT | chain         | v_propagation                          | 1.000 [0.999, 1.000] (n=3843) | 1.000 [0.999, 1.000] (n=3843) |
| 3-hop SIIT | chain         | v_restore_reverses_y                   | 1.000 [0.999, 1.000] (n=3843) | 1.000 [0.999, 1.000] (n=3843) |
| 3-hop SIIT | chain         | y_counterfactual                       | 1.000 [0.999, 1.000] (n=3843) | 1.000 [0.999, 1.000] (n=3843) |
| 3-hop SIIT | interchange_u | control_coord32_changed_bits           | 0.000 [0.000, 0.001] (n=3843) | 0.000 [0.000, 0.001] (n=3843) |
| 3-hop SIIT | interchange_u | random_pairs_changed_bits              | 1.000 [0.999, 1.000] (n=3843) | 1.000 [0.999, 1.000] (n=3843) |
| 3-hop SIIT | interchange_v | control_coord32_both_c1                | 0.000 [0.000, 0.002] (n=1895) | 0.000 [0.000, 0.002] (n=1895) |
| 3-hop SIIT | interchange_v | corrected_both_c1_changed_bits         | 1.000 [0.998, 1.000] (n=1895) | 1.000 [0.998, 1.000] (n=1895) |
| 3-hop SIIT | interchange_v | legacy_decoupled_recipient_c1_donor_c0 | 0.624 [0.613, 0.634] (n=7917) | 1.000 [1.000, 1.000] (n=7917) |
| 3-hop SIIT | interchange_v | random_pairs_changed_bits              | 1.000 [0.999, 1.000] (n=5808) | 1.000 [0.999, 1.000] (n=5808) | 

## Table 4.5 Ground-truth depth per model

| group          | model_id              | necessary_mlp_layers   | y_shortcut_scores   | D_true     | evidence                                 | exclusive_sites                                                                                       |
|:---------------|:----------------------|:-----------------------|:--------------------|:-----------|:-----------------------------------------|:------------------------------------------------------------------------------------------------------|
| 2-hop baseline | 2hop_L2_seed0         | [0, 1]                 | {0: 1.0}            | 1          | ablation+shortcut test                   | nan                                                                                                   |
| 2-hop baseline | 2hop_L2_seed1         | [0, 1]                 | {0: 1.0}            | 1          | ablation+shortcut test                   | nan                                                                                                   |
| 2-hop baseline | 2hop_L2_seed2         | [0, 1]                 | {0: 1.0}            | 1          | ablation+shortcut test                   | nan                                                                                                   |
| 2-hop SIIT     | 2hop_L2_siit_k1_seed0 | [0, 1]                 | {0: 0.0}            | 2          | ablation+shortcut test+interchange/chain | yes                                                                                                   |
| 2-hop SIIT     | 2hop_L2_siit_k1_seed1 | [0, 1]                 | {0: 0.0}            | 2          | ablation+shortcut test+interchange/chain | no: ablate_mlp0_restore_u_site=0.93                                                                   |
| 2-hop SIIT     | 2hop_L2_siit_k1_seed2 | [0, 1]                 | {0: 0.0}            | 2          | ablation+shortcut test+interchange/chain | yes                                                                                                   |
| 2-hop SIIT     | 2hop_L2_siit_k1_seed3 | [0, 1]                 | {0: 0.0}            | 2          | ablation+shortcut test+interchange/chain | yes                                                                                                   |
| 2-hop SIIT     | 2hop_L2_siit_k1_seed4 | [0, 1]                 | {0: 0.0}            | 2          | ablation+shortcut test+interchange/chain | yes                                                                                                   |
| 3-hop baseline | 3hop_L2_seed0         | [0, 1]                 | {0: 0.0286}         | 2          | ablation+shortcut test                   | nan                                                                                                   |
| 3-hop baseline | 3hop_L2_seed1         | [0, 1]                 | {0: 0.4602}         | unresolved | nan                                      | nan                                                                                                   |
| 3-hop baseline | 3hop_L2_seed2         | [0, 1]                 | {0: 0.0146}         | 2          | ablation+shortcut test                   | nan                                                                                                   |
| 3-hop SIIT     | 3hop_L3_siit_k1_seed0 | [0, 1, 2]              | {0: 0.0, 1: 0.0}    | 3          | ablation+shortcut test+interchange/chain | no: ablate_mlp0_restore_u_site=0.65; ablate_mlp0_restore_v_site=0.67; ablate_mlp1_restore_v_site=0.32 |
| 3-hop SIIT     | 3hop_L3_siit_k1_seed1 | [0, 1, 2]              | {0: 0.0, 1: 0.0}    | 3          | ablation+shortcut test+interchange/chain | no: ablate_mlp0_restore_u_site=0.76; ablate_mlp0_restore_v_site=0.77; ablate_mlp1_restore_v_site=0.28 |
| 3-hop SIIT     | 3hop_L3_siit_k1_seed2 | [0, 1, 2]              | {0: 0.0, 1: 0.0}    | 3          | ablation+shortcut test+interchange/chain | no: ablate_mlp0_restore_u_site=0.82; ablate_mlp0_restore_v_site=0.81; ablate_mlp1_restore_v_site=0.40 |
| 3-hop SIIT     | 3hop_L3_siit_k1_seed3 | [0, 1, 2]              | {0: 0.0, 1: 0.0}    | 3          | ablation+shortcut test+interchange/chain | no: ablate_mlp0_restore_u_site=0.44; ablate_mlp0_restore_v_site=0.48; ablate_mlp1_restore_v_site=0.18 |
| 3-hop SIIT     | 3hop_L3_siit_k1_seed4 | [0, 1, 2]              | {0: 0.0, 1: 0.0}    | 3          | ablation+shortcut test+interchange/chain | no: ablate_mlp0_restore_u_site=0.97; ablate_mlp0_restore_v_site=0.97; ablate_mlp1_restore_v_site=0.31 | 

