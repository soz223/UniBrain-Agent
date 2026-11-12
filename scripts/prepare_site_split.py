"""Export a declared site holdout from the complete UniBrain master array."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unibrain_agent.site_split import export_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("master", "metadata", "templates", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--held-out-site", required=True)
    parser.add_argument("--dataset", choices=("abide", "adhd"), default="abide")
    parser.add_argument("--template-origin", required=True)
    parser.add_argument("--template-scope", choices=("external_public", "source_training_only"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    report = export_site(args.master, args.metadata, args.templates, args.output, args.held_out_site,
                         args.template_origin, args.template_scope, args.seed, args.validation_fraction, args.dataset)
    print(json.dumps({"status": "complete", "counts": {k: v["count"] for k, v in report["splits"].items()}}))


if __name__ == "__main__":
    main()
