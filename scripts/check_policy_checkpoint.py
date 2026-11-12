"""Check actual actor and critic parameter changes using CPU only."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from unibrain_agent.policy_checkpoint import digest, inspect_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(2)
    try:
        result = inspect_policy(args.checkpoint, args.report)
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        message = str(error)
        for name in ("checkpoint", "report", "output"):
            message = message.replace(str(getattr(args, name).resolve()), "<" + name + ">")
        result = {"status": "failed", "device": "cpu", "gpu_used": False, "error": message,
                  "checkpoint_sha256": digest(args.checkpoint), "report_sha256": digest(args.report)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "policy_updates": result.get("policy_updates")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
