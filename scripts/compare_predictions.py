"""Estimate paired subject confidence intervals from complete trainer reports."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unibrain_agent.paired_analysis import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    pairs = json.loads(args.manifest.read_text(encoding="utf-8"))["pairs"]
    pairs = [{key: args.manifest.resolve().parent / row[key] for key in ("a", "b")} for row in pairs]
    report = compare(pairs, args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
