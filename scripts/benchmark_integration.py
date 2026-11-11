"""Run integration trials or CPU checker self-tests."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.integration_benchmark import fault_selftest, preflight, read_config, run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "faults"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode != "faults" and not args.config:
        parser.error("--config is required")
    if args.mode != "preflight" and not args.output:
        parser.error("--output is required")
    if args.mode == "faults":
        result = fault_selftest(args.output)
    elif args.mode == "preflight":
        result = preflight(read_config(args.config))
    else:
        result = run_benchmark(read_config(args.config), args.output)
    print(json.dumps(result if args.mode == "preflight" else {"output": str(args.output), "status": result.get("status", result.get("passed"))}))
    return 1 if result.get("status") == "blocked" or result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
