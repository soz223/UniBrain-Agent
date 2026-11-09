"""Check every selectable image pipeline adapter on the public sample."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unibrain_agent.builds import create_build, prepare
from unibrain_agent.catalog import DEFAULTS, STAGES
from unibrain_agent.runner import write_json


def main():
    cases = [("default", "default", dict(DEFAULTS))]
    for stage, _, _, options in STAGES:
        for method in options[1:]:
            if method == "M15" or (stage == "registration" and method == "M03"):
                continue
            selection = {**DEFAULTS, stage: method}
            if method == "M03":
                selection.update(extraction="M03", registration="M03")
            cases.append((stage, method, selection))
    results = []
    target = ROOT / "builds/adapter_checks.json"
    for stage, method, selection in cases:
        record = create_build(selection, "rules")
        print(f"Checking {stage}: {method}", flush=True)
        result = prepare(record["id"])
        item = {"stage": stage, "method": method, "build_id": result["id"], "status": result["status"], "error": result["error"]}
        results.append(item)
        write_json(target, results)
        print(json.dumps(item), flush=True)
    return int(any(result["status"] != "ready" for result in results))


if __name__ == "__main__":
    sys.exit(main())
