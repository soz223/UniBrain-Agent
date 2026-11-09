import argparse


def main():
    parser = argparse.ArgumentParser(prog="unibrain_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a measured baseline and agent candidate comparison")
    run.add_argument("--backend", choices=["codex", "rules"], default="codex")
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--data-dir")
    run.add_argument("--output-dir")
    run.add_argument("--build-id")
    prepare = sub.add_parser("prepare", help="Generate and validate a selected pipeline")
    prepare.add_argument("selection")
    prepare.add_argument("--backend", choices=["codex", "rules"], default="codex")
    serve = sub.add_parser("serve", help="Open the local prototype dashboard")
    serve.add_argument("--port", type=int, default=8769)
    serve.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        from .runner import run as execute
        from .builds import build_path
        result = execute(args.backend, args.steps, args.seed, args.data_dir, args.output_dir,
                         build_dir=build_path(args.build_id) if args.build_id else None)
        print(result["run_dir"] + "/index.html")
    elif args.command == "prepare":
        import json
        from pathlib import Path
        from .builds import create_build, prepare as execute
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        record = create_build(selection, args.backend)
        result = execute(record["id"])
        print(json.dumps({"build_id": result["id"], "status": result["status"], "error": result["error"]}))
        if result["status"] != "ready":
            raise SystemExit(1)
    else:
        from .server import serve as start
        start(args.port, args.open)


if __name__ == "__main__":
    main()
