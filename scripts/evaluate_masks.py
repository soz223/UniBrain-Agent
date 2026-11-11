"""Evaluate aligned discrete NPY masks without inventing physical spacing."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unibrain_agent.research_metrics import region_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--spacing", type=float, nargs=3, required=True)
    parser.add_argument("--units", choices=("mm", "voxel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    result = region_metrics(np.load(args.prediction, allow_pickle=False), np.load(args.reference, allow_pickle=False), args.spacing, args.units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
