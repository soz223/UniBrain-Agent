"""Train full architecture comparators from a pinned source and fold directory."""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.replacement_training import parser, train


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        result = train(args)
        print(json.dumps({"status": result["status"], "steps": result.get("steps"), "report": str(args.output / "report.json")}))
    except Exception as error:
        traceback.print_exc()
        print(json.dumps({"status": "failed", "error": str(error)}))
        raise SystemExit(1)
