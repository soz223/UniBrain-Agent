"""Export a complete original fold using the version 2 provenance protocol."""

import argparse
import json
from pathlib import Path

if __package__:
    from .export_acceptance_subset import export
else:
    from export_acceptance_subset import export


def export_fold(cohort, fold, destination, full=False, dataset=None, existing=None):
    return export(cohort, destination, fold=fold, dataset=dataset, full=full, existing=existing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dataset", choices=("abide", "adhd"), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--full", action="store_true", help="Retain every original subject in the original split order")
    parser.add_argument("--existing", type=Path, help="Verify and hard-link an existing fold on the same filesystem")
    args = parser.parse_args()
    report = export_fold(args.cohort, args.fold, args.destination, args.full, args.dataset, args.existing)
    print(json.dumps({"status": report["status"], "scope": report["scope"], "dataset": report["dataset"],
                      "fold": report["fold"], "counts": {key: value["count"] for key, value in report["splits"].items()}}))
