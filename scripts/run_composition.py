"""Train or inspect a full two-stage composition using the matched trainer."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.full_composition import build_composed_model
from unibrain_agent.replacement_training import parser, preflight, train


def main():
    cli = parser()
    cli.add_argument("--build-only", action="store_true")
    args = cli.parse_args()
    if not args.secondary_extraction or args.method not in {"voxelmorph", "lapirn"}:
        cli.error("Composition requires --secondary-extraction and a dense registration method")
    if args.build_only:
        audit = preflight(args)
        model, config, architecture = build_composed_model(
            args.source, args.method, args.secondary_extraction, mode=args.composition_mode,
            registration_weights=args.weights, extraction_weights=args.extraction_weights,
            previous_checkpoint=args.previous_checkpoint, previous_report=args.previous_report,
            previous_sha256=args.previous_sha256, host_checkpoint=args.host_checkpoint, dataset=args.dataset, fold=args.fold)
        args.output.mkdir(parents=True, exist_ok=False)
        result = {"status": "built", "scope": "CPU construction and strict transfer only; no training or GPU forward",
                  "architecture": architecture, "config": config, "audit": audit}
        (args.output / "build_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        del model
    else:
        result = train(args)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
