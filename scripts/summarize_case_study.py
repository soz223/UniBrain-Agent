"""Export a sanitized audit of an existing full adapter repair run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.case_study import write_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_trace(args.run, args.output)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}))
