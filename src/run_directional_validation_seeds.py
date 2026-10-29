import subprocess
import sys
from pathlib import Path

source = Path("src/directional_intervention_validation.py")
text = source.read_text()

for seed in (1, 2):
    updated = text.replace(
        'SEED = 0',
        f'SEED = {seed}',
    ).replace(
        'CHECKPOINT = Path("checkpoints/2hop_L2_seed0.pt")',
        f'CHECKPOINT = Path("checkpoints/2hop_L2_seed{seed}.pt")',
    ).replace(
        'OUT = Path("results/directional_intervention_validation_seed0_layer1.csv")',
        f'OUT = Path("results/directional_intervention_validation_seed{seed}_layer1.csv")',
    )

    if updated == text:
        raise RuntimeError(f"Could not configure seed {seed}")

    temp_script = Path(f"src/_directional_validation_seed{seed}.py")
    temp_script.write_text(updated)

    try:
        print(f"\n===== Running seed {seed} =====", flush=True)
        result = subprocess.run([sys.executable, str(temp_script)])
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        temp_script.unlink(missing_ok=True)

print("\nBoth additional seed runs completed.")
