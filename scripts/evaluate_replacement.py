"""Evaluate a trained replacement pipeline without further optimization."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.replacement_evaluation import evaluate_checkpoint, parser


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        result = evaluate_checkpoint(args)
        print(json.dumps({"status": result["status"], "method": result["method"],
                          "optimizer_updates": 0, "report": str(args.output / "report.json") if not args.preflight else None}))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        raise SystemExit(1)
