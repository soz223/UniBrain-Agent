"""Run the unchanged original model on exported smoke subsets, one fold at a time."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    results = []
    for cohort in ("abide", "adhd"):
        for fold in range(1, 6):
            folder = workspace / "runs" / cohort / f"round{fold}"
            if folder.exists():
                raise FileExistsError(folder)
            folder.mkdir(parents=True)
            data = workspace / "data" / cohort / f"round{fold}"
            manifest = json.loads((data / "cohort.json").read_text())
            if manifest["scope"] != "smoke_subset":
                raise ValueError("Expected an explicitly labeled smoke subset")
            command = [sys.executable, str(project / "scripts/run_upstream.py"), str(args.source.resolve()),
                       "--epochs", "2", "--data-dir", str(data), "--output-dir", str(folder),
                       "--threads", "4", "--telemetry"]
            print(f"START {cohort} fold {fold}", flush=True)
            with (folder / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=project, env={**os.environ, "MPLBACKEND": "Agg"},
                                        stdout=log, stderr=subprocess.STDOUT)
            path = folder / "verification.json"
            verification = json.loads(path.read_text()) if path.exists() else {}
            passed = result.returncode == 0 and verification.get("audit", {}).get("passed", False)
            results.append({"cohort": cohort, "fold": fold, "passed": passed, "exit_code": result.returncode,
                            "updates": len(verification.get("updates", [])),
                            "validation_epochs": verification.get("audit", {}).get("validation_epochs", 0),
                            "seconds": verification.get("elapsed_seconds"), "run": str(folder)})
            (workspace / "summary.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(results[-1]), flush=True)
            if not passed:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
