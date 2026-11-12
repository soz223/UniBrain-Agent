"""Create and execute explicitly selected paper experiment jobs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unibrain_agent.experiment_suite import AGENT_VARIANTS, example_config, import_integration_index, load_config, plan, preflight, run_job, save, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("path", type=Path)
    imported = sub.add_parser("import-integration-index")
    imported.add_argument("--config", type=Path, required=True)
    imported.add_argument("--index", type=Path, required=True)
    imported.add_argument("--variants", nargs="+", choices=AGENT_VARIANTS, required=True)
    imported.add_argument("--apply", action="store_true")
    for name in ("plan", "run", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "plan":
            command.add_argument("--output", type=Path, required=True)
        elif name == "run":
            command.add_argument("--job", required=True)
            command.add_argument("--smoke-steps", type=int, choices=(1, 2))
            command.add_argument("--timeout", type=int, default=86400)
            command.add_argument("--execute", action="store_true")
        else:
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "import-integration-index":
        result = import_integration_index(args.config, args.index, args.variants, args.apply)
        print(json.dumps(result, indent=2))
        return 0
    if args.action == "init":
        if args.path.exists():
            parser.error("Configuration already exists")
        save(args.path, example_config())
        return 0
    config = load_config(args.config)
    if args.action == "plan":
        if args.output.exists():
            parser.error("Plan output already exists")
        jobs = [{**job, "preflight": preflight(config, job)} for job in plan(config)]
        save(args.output, {"config_hash": config["_hash"], "jobs": jobs})
        print(json.dumps({"planned": len(jobs), "configured": sum(job["preflight"]["ready"] for job in jobs)}))
        return 0
    if args.action == "summarize":
        result = write_summary(config, args.output)
        print(json.dumps({"planned": result["planned"], "complete": result["complete"]}))
        return 0
    job = next((job for job in plan(config) if job["id"] == args.job), None)
    if job is None:
        parser.error("Unknown job ID")
    if not args.execute:
        print(json.dumps(preflight(config, job), indent=2))
        return 0
    result = run_job(config, job, args.smoke_steps, args.timeout)
    print(json.dumps({"status": result["status"], "job_id": job["id"]}))
    return 0 if result["status"] in {"complete", "smoke_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
