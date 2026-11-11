"""Run an isolated, bounded full architecture replacement experiment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.full_replacement import run_loop
from unibrain_agent.paths import ORIGINAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ORIGINAL)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--provider", choices=("codex", "claude", "openai", "anthropic", "compatible"), default="codex")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--inject-hard-mask", action="store_true")
    args = parser.parse_args()
    record = run_loop(args.output, args.source, args.data, args.python, args.provider,
                      args.gpu, args.inject_hard_mask, args.attempts, args.steps)
    print(json.dumps({"status": record["status"], "report": str(args.output / "report.json")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
