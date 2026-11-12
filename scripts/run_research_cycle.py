"""Run the candidate selection, generated integration and training cycle."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.research_cycle import parser, run_cycle


if __name__ == "__main__":
    args = parser().parse_args()
    report = run_cycle(args)
    print(json.dumps({"status": report["status"], "phase": report.get("phase"), "actual_updates": report["actual_updates"],
                      "report": str(args.output / "report.json")}))
    raise SystemExit(0 if report["status"] in ("complete", "selected", "ready") else 1)
