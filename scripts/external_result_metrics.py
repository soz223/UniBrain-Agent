"""Read aggregate external evaluation metrics for the existing W&B monitor."""

import json
import sys
from pathlib import Path

from wandb_metrics import METRICS, scalars


def read_result(folder, cursor):
    event = {}
    validation = folder / "validation.json"
    conversion = folder / "result.json"
    if validation.is_file():
        record = json.loads(validation.read_text())
        event = scalars(record.get("validation", record), METRICS, "evaluation/")
        event.update(scalars(record, ("n_train", "n_val"), "subjects/"))
    elif conversion.is_file():
        record = json.loads(conversion.read_text())
        event = {"bids/validation_errors": record["validation_errors"],
                 "bids/warnings": len(record["warnings"]), "bids/outputs": len(record["outputs"])}
    return {"ready": bool(event), "more": False, "status": "complete" if event else "waiting",
            "cursor": [int(bool(event)), 0, 0], "events": [event] if event and not cursor else []}


if __name__ == "__main__":
    print(json.dumps(read_result(Path(sys.argv[1]), int(sys.argv[2]))))
