import argparse


def main():
    parser = argparse.ArgumentParser(prog="unibrain_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a measured baseline and agent candidate comparison")
    run.add_argument("--backend", choices=["configured", "codex", "rules"], default="configured")
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--data-dir")
    run.add_argument("--output-dir")
    run.add_argument("--build-id")
    run.add_argument("--device", default=None)
    run.add_argument("--single", action="store_true")
    run.add_argument("--coordination", choices=["single", "roles"], default="single")
    original = sub.add_parser("original", help="Run the full original UniBrain CUDA workflow")
    original.add_argument("--epochs", type=int, default=2)
    original.add_argument("--output-dir")
    prepare = sub.add_parser("prepare", help="Generate and validate a selected pipeline")
    prepare.add_argument("selection")
    prepare.add_argument("--backend", choices=["configured", "codex", "rules"], default="configured")
    serve = sub.add_parser("serve", help="Open the local prototype dashboard")
    serve.add_argument("--port", type=int, default=8775)
    serve.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        from .runner import run as execute
        from .builds import build_path
        from .builds import verify_build
        folder = build_path(args.build_id) if args.build_id else None
        build = verify_build(folder) if folder else None
        if build and build.get("profile") == "original":
            if args.coordination == "roles":
                parser.error("Original reproduction keeps its optimizer; use a custom build for role coordination")
            from .native import run_original
            result = run_original(args.steps, build=build, output_dir=args.output_dir)
        else:
            result = execute(args.backend, args.steps, args.seed, args.data_dir, args.output_dir,
                             build_dir=folder, comparison=not args.single, device=args.device, coordination=args.coordination)
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
    elif args.command == "original":
        from .native import run_original
        print(run_original(args.epochs, output_dir=args.output_dir)["run_dir"])
    else:
        from .server import serve as start
        start(args.port, args.open)


if __name__ == "__main__":
    main()
