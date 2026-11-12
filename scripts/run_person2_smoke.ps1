$ErrorActionPreference = "Stop"

# Run from the repository root in an activated Python environment.
python -m pytest -q

python -m src.replacements.inspect_activations `
  --task 2hop `
  --checkpoint checkpoints/2hop_L2_seed0.pt `
  --data data/boolean_2hop_improved.csv `
  --split test `
  --batch-size 256

python -m src.replacements.train `
  --task 2hop `
  --model-type plt `
  --checkpoint checkpoints/2hop_L2_seed0.pt `
  --data data/boolean_2hop_improved.csv `
  --n-features 16 `
  --sparsity 0.0001 `
  --epochs 1 `
  --batch-size 256 `
  --capture-batch-size 512 `
  --out-dir results/person2/pilot `
  --device cpu

python -m src.replacements.train `
  --task 2hop `
  --model-type clt `
  --checkpoint checkpoints/2hop_L2_seed0.pt `
  --data data/boolean_2hop_improved.csv `
  --n-features 16 `
  --sparsity 0.0001 `
  --write-span 2 `
  --epochs 1 `
  --batch-size 256 `
  --capture-batch-size 512 `
  --out-dir results/person2/pilot `
  --device cpu

python -m src.replacements.compare_fidelity `
  --table results/person2/pilot/fidelity_table.csv `
  --tolerance 0.02
