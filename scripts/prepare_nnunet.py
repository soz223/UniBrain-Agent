"""Export one training fold and run the installed official nnU-Net planner."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.nnunet_prepare import export_training, parser, run_planner


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        report = export_training(args) if args.command == "export" else run_planner(args)
        print(json.dumps({"status": report["status"], "output": str(args.output), "reason": report.get("reason")}))
        raise SystemExit(0 if report["status"] in ("exported", "ready") else 2)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        raise SystemExit(1)
