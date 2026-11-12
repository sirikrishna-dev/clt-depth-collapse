from pathlib import Path


EXPECTED_CHECKPOINTS = [
    f"{task}_L2_seed{seed}.pt"
    for task in ("2hop", "3hop")
    for seed in (0, 1, 2)
]


def verify_checkpoints():
    """Check whether the expected original model checkpoints exist."""

    root = Path(__file__).resolve().parents[2]
    checkpoint_dir = root / "checkpoints"

    print("Checking expected original model checkpoints...")
    print(f"Checkpoint directory: {checkpoint_dir}\n")

    if not checkpoint_dir.is_dir():
        print("ERROR: Checkpoint directory does not exist.")
        return

    found = []
    missing = []

    for filename in EXPECTED_CHECKPOINTS:
        path = checkpoint_dir / filename

        if path.is_file() and path.stat().st_size > 0:
            found.append(filename)
            print(f"FOUND:   {filename}")
        else:
            missing.append(filename)
            print(f"MISSING: {filename}")

    print(f"\nFound: {len(found)} of {len(EXPECTED_CHECKPOINTS)}")
    print(f"Missing: {len(missing)} of {len(EXPECTED_CHECKPOINTS)}")

    print(
        "\nNote: File existence and size do not prove that a "
        "checkpoint loads correctly or that its model is scientifically valid."
    )


if __name__ == "__main__":
    verify_checkpoints()