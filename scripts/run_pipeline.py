#!/usr/bin/env python
"""One entry point for every stage.  Each stage is resumable (skips finished work).

    python scripts/run_pipeline.py --config configs/quick.yaml --stage zoo
    python scripts/run_pipeline.py --config configs/quick.yaml --stage all
Stages: zoo, replacements, graphs, validate, probes, all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402
import torch  # noqa: E402

from cdc import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["zoo", "replacements", "graphs", "validate", "probes", "all"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(open(args.config))
    stages = ["zoo", "replacements", "graphs", "validate", "probes"] if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"===== stage {s} =====", flush=True)
        getattr(pipeline, f"stage_{s}")(cfg, device=args.device)


if __name__ == "__main__":
    main()
