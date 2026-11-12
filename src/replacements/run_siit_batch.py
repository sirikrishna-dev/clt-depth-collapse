"""Train PLT and CLT-style replacements on every SIIT checkpoint, then evaluate them.

Each checkpoint gets its own folder under results/person2/siit/<model_id>/ because
train.py names runs without the base model's identity and would overwrite otherwise.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from .siit_utils import (
    build_matching_report,
    discover_siit_checkpoints,
    flatten_recovery,
    parse_model_id,
)


def clt_span_for(layers: int, spec: str) -> int:
    span = layers if spec == "full" else int(spec)
    return max(1, min(span, layers))


def train_pair(entry: dict, args, out_dir: Path, data_path: Path) -> None:
    from .train import train  # lazy import so --dry-run works without torch

    span = clt_span_for(entry["layers"], args.clt_span)
    for model_type in ("plt", "clt"):
        run_id = f"{entry['task']}_{model_type}_seed{args.seed}_features{args.n_features}_span{1 if model_type == 'plt' else span}"
        if args.skip_existing and (out_dir / f"{run_id}.pt").exists():
            print(f"  [skip] {run_id} already trained")
            continue
        ns = argparse.Namespace(
            root=".", task=entry["task"], model_type=model_type,
            checkpoint=str(entry["path"].resolve()), data=str(data_path.resolve()),
            out_dir=str(out_dir.resolve()), n_features=args.n_features, sparsity=args.sparsity,
            write_span=span, epochs=args.epochs, batch_size=args.batch_size,
            capture_batch_size=args.capture_batch_size, lr=args.lr, seed=args.seed, device=args.device,
        )
        print(f"  training {run_id}")
        train(ns)


def evaluate_pair(entry: dict, args, out_dir: Path, data_path: Path) -> None:
    from .evaluate_siit_intermediates import evaluate

    plt_path = next(out_dir.glob(f"{entry['task']}_plt_*.pt"))
    clt_path = next(out_dir.glob(f"{entry['task']}_clt_*.pt"))
    result = evaluate(entry["path"], data_path, plt_path, clt_path, device="cpu")
    result["model_id"] = entry["model_id"]
    (out_dir / "intermediate_recovery.json").write_text(json.dumps(result, indent=2))


def _is_model_dir(path: Path) -> bool:
    try:
        parse_model_id(path.name)
        return True
    except ValueError:
        return False


def aggregate(out_root: Path, tolerance: float) -> None:
    fid_frames, rec_rows = [], []
    for folder in sorted(p for p in out_root.iterdir() if p.is_dir() and _is_model_dir(p)):
        table = folder / "fidelity_table.csv"
        if table.exists():
            frame = pd.read_csv(table)
            frame.insert(0, "model_id", folder.name)
            frame.insert(1, "siit_seed", int(folder.name.rsplit("seed", 1)[1]))
            fid_frames.append(frame)
        recovery = folder / "intermediate_recovery.json"
        if recovery.exists():
            data = json.loads(recovery.read_text())
            seed = int(folder.name.rsplit("seed", 1)[1])
            rec_rows.extend(flatten_recovery(folder.name, data["task"], seed, data["results"]))

    if fid_frames:
        fidelity = pd.concat(fid_frames, ignore_index=True)
        fidelity.to_csv(out_root / "siit_fidelity_table.csv", index=False)
        report = build_matching_report(fidelity, tolerance)
        report.to_csv(out_root / "siit_fidelity_matching_report.csv", index=False)
        print("\n=== PLT vs CLT fidelity (validation relative L2) ===")
        cols = [c for c in ["model_id", "plt_validation_relative_l2", "clt_validation_relative_l2",
                            "absolute_gap", "relative_gap_percent", "status"] if c in report.columns]
        print(report[cols].to_string(index=False))
    if rec_rows:
        recovery = pd.DataFrame(rec_rows)
        recovery.to_csv(out_root / "siit_intermediate_recovery_summary.csv", index=False)
        write_markdown_summary(out_root, recovery)



def write_markdown_summary(out_root: Path, recovery: pd.DataFrame) -> None:
    grouped = (
        recovery.groupby(
            ["task", "variable", "representation", "kind", "layer", "designated_layer"],
            as_index=False
        )
        .agg(
            seeds=("siit_seed", "nunique"),
            test_acc_mean=("mean_test_accuracy", "mean"),
            test_acc_min=("mean_test_accuracy", "min"),
            majority=("majority_baseline_test_accuracy", "mean"),
            shuffled_acc=("shuffled_test_accuracy", "mean"),
            shuffled_p=("shuffled_empirical_p_value", "mean"),
        )
        .sort_values(["task", "variable", "layer", "kind"])
    )

    lines = [
        "# SIIT recovery summary (auto-generated)",
        "",
        "Mean per-bit linear-probe test accuracy across SIIT seeds.",
        "`majority` is the majority-class baseline.",
        "`shuffled` is the mean test accuracy after training with shuffled labels.",
        "`shuffled p` is the mean empirical p-value across SIIT seeds.",
        "",
    ]

    for (task, var), g in grouped.groupby(["task", "variable"]):
        lines += [
            f"## {task} - variable `{var}`",
            "",
            "| representation | kind | designated | seeds | test acc | majority | shuffled acc | shuffled p |",
            "|---|---|:-:|--:|--:|--:|--:|--:|",
        ]

        for _, r in g.iterrows():
            lines.append(
                f"| {r.representation} | {r.kind} "
                f"| {'yes' if r.designated_layer else ''} "
                f"| {r.seeds} "
                f"| {r.test_acc_mean:.4f} "
                f"| {r.majority:.4f} "
                f"| {r.shuffled_acc:.4f} "
                f"| {r.shuffled_p:.4f} |"
            )

        lines.append("")

    (out_root / "SIIT_SUMMARY.md").write_text("\n".join(lines))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-root", default="results/person2/siit")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=1e-4)
    parser.add_argument("--clt-span", default="full",
                        help="'full' = n_layers (2 for 2hop, 3 for 3hop), or an integer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--capture-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--tolerance", type=float, default=0.02, help="PROVISIONAL; not team-approved")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = discover_siit_checkpoints(args.checkpoint_dir)
    if args.only:
        entries = [e for e in entries if e["model_id"] in set(args.only)]
    print(f"Found {len(entries)} SIIT checkpoint(s):")
    for e in entries:
        print(f"  {e['model_id']:<28} task={e['task']} layers={e['layers']} clt_span={clt_span_for(e['layers'], args.clt_span)}")
    if not entries:
        raise SystemExit("No SIIT checkpoints found - check --checkpoint-dir.")
    if args.dry_run:
        return

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for n, entry in enumerate(entries, 1):
        data_path = Path(args.data_dir) / f"boolean_{entry['task']}_improved.csv"
        out_dir = out_root / entry["model_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{n}/{len(entries)}] {entry['model_id']}")
        start = time.time()
        try:
            train_pair(entry, args, out_dir, data_path)
            if not args.skip_probes:
                evaluate_pair(entry, args, out_dir, data_path)
        except Exception:
            traceback.print_exc()
            failures.append(entry["model_id"])
        print(f"  done in {time.time() - start:.0f}s")

    aggregate(out_root, args.tolerance)
    if failures:
        print("\nFAILED:", ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()