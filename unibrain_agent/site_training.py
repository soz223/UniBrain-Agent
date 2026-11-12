"""Validate native site experiments before invoking the full trainer."""

from pathlib import Path


def validate_site_options(args):
    split_id = getattr(args, "split_id", None)
    manifest = Path(args.data) / "site_split.json"
    if split_id is None:
        if manifest.is_file():
            raise ValueError("Site data require an explicit split ID; they are not an original cross-validation fold")
        return
    if not split_id.strip() or args.dataset not in {"abide", "adhd"} or args.fold is not None:
        raise ValueError("Site experiments require a dataset and split ID, with no numeric fold")
    if not manifest.is_file():
        raise ValueError("Site experiments require a prepared site_split.json manifest")
    allowed = {("original", "original"), ("synthstrip", "joint"), ("synthstrip", "independent_finetuned")}
    if (args.method, args.setting) not in allowed:
        raise ValueError("Native site experiments support original and SynthStrip joint or independent_finetuned")
    if args.initialization == "training_fold" or args.host_checkpoint or args.secondary_extraction:
        raise ValueError("Site experiments cannot import numeric-fold checkpoints or composition state")
    if (args.trainable != "all" or args.cut_extraction or args.cut_boundary != "none"
            or args.undo_repair != "none" or args.repair_strategy != "soft"):
        raise ValueError("The site protocol trains all intended components with the declared connected interface")


def main(argv=None):
    import json
    from .replacement_training import parser, train

    cli = parser()
    args = cli.parse_args(argv)
    try:
        if not getattr(args, "split_id", None):
            raise ValueError("The site runner requires --split-id")
        validate_site_options(args)
        result = train(args)
        print(json.dumps({"status": result["status"], "split_id": args.split_id,
                          "steps": result.get("steps"), "report": str(args.output / "report.json")}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
