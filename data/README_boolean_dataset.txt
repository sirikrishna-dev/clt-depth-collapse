CLT Depth-Collapse Project — Improved Boolean Dataset (v2.0)

FILES
- boolean_2hop_improved.csv: 7,000 rows
- boolean_3hop_improved.csv: 7,000 rows
- boolean_dataset_improved_combined.csv: 14,000 rows
- boolean_dataset_split_summary.csv: row counts by task and split
- boolean_dataset_improved_metadata.json: equations, dimensions, seed, split details

SIZE
- Bits per input vector: 16
- Rows per task: 7,000 (train 5,000, validation 1,000, test 1,000)
- Combined rows: 14,000
- Rows are bit-vector examples; each vector is stored in separate bit columns.

TASK DEFINITIONS
2-hop:
  u = bitwise XOR(a, b)
  y = bitwise AND(u, c)

3-hop:
  u = bitwise XOR(a, b)
  v = bitwise AND(u, c)
  y = bitwise XOR(v, d)

Each vector is 16 bits long. The same equations apply independently at each bit position.
The input tuples are unique across train/validation/test within each task.

IMPORTANT RESEARCH CAVEAT
This dataset provides exact equation-level labels and separate splits, but it does not guarantee a model will implement the intended intermediate circuit. The model could learn an equivalent computation. Verify intermediate dependence using activation patching/causal interventions. Also ensure the architecture and tokenization/representation are suitable for the model being trained.

This is a generated task dataset, not an existing InterpBench dataset. InterpBench model compatibility must be checked separately.
