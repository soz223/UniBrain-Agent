"""Run the pinned upstream entry point with an explicit epoch budget."""

import argparse
import hashlib
import inspect
import json
import math
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

COMMIT = "f9f8d906b1725046d532f862a28876482aca2cfa"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_results(source, record):
    import numpy as np
    import torch

    counts = {split: len(np.load(source / "dataset" / f"adhd_{split}_sample.npy", allow_pickle=True))
              for split in ("train", "val", "test")}
    validation_lines = [line for path in source.glob("result/loss_log/*_val.txt")
                        for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("Epoch[")]
    checkpoints = []
    for path in source.glob("result/model/*.pth"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        finite = all(bool(torch.isfinite(value).all()) for value in state.values() if value.is_floating_point())
        checkpoints.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
                            "state_tensors": len(state), "finite": finite})
        del state
    expected = counts["train"] * record["epochs"]
    active_modules = {"ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred"}
    checks = {"source_unchanged": record["source_unchanged"], "training_completed": record["status"] == "completed",
              "all_training_updates": len(record["updates"]) == expected,
              "finite_gradient_norms": bool(record["updates"]) and all(step["finite_gradient_norms"] for step in record["updates"]),
              "active_modules_updated": bool(record["updates"]) and all(active_modules.issubset(step["modules_with_observed_updates"]) for step in record["updates"]),
              "all_validation_epochs": len(validation_lines) == record["epochs"],
              "checkpoint_saved": bool(checkpoints), "checkpoint_finite": bool(checkpoints) and all(item["finite"] for item in checkpoints)}
    return {"passed": all(checks.values()), "checks": checks, "sample_counts": counts,
            "expected_updates": expected, "validation_epochs": len(validation_lines), "checkpoints": checkpoints,
            "dataset_sha256": {path.name: digest(path) for path in sorted((source / "dataset").glob("*.npy"))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.epochs < 2:
        parser.error("Use at least two epochs to exercise upstream checkpoint saving")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != COMMIT or subprocess.check_output(["git", "diff", "HEAD", "--", "*.py"], cwd=source):
        raise RuntimeError("Expected the clean pinned UniBrain source")
    os.chdir(source)
    sys.path.insert(0, str(source))
    import torch
    import torchvision
    import torch_geometric
    import train

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Upstream uses CUDA tensors directly; use a CUDA environment")
    record = {"commit": commit, "status": "running", "epochs": args.epochs,
              "runtime": {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                          "torch_geometric": torch_geometric.__version__, "gpu": torch.cuda.get_device_name()},
              "source_sha256": {name: digest(source / name) for name in ("main.py", "model.py", "train.py", "utils.py")},
              "changes": ["Epoch budget only", "Read-only optimizer observations"], "updates": []}
    output = source.parent / "verification.json"

    def save():
        output.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")

    original_train = train.train_UniBrain
    original_step = torch.optim.Adam.step
    model = None
    started = time.perf_counter()

    def observe_step(optimizer, *values, **options):
        before = {name: parameter.detach().flatten()[::max(1, parameter.numel() // 32)][:32].clone()
                  for name, parameter in model.named_parameters() if parameter.grad is not None}
        norms = [float(parameter.grad.norm()) for parameter in model.parameters() if parameter.grad is not None]
        result = original_step(optimizer, *values, **options)
        changed = [name for name, parameter in model.named_parameters() if name in before and
                   not torch.equal(before[name], parameter.detach().flatten()[::max(1, parameter.numel() // 32)][:32])]
        update = {"step": len(record["updates"]) + 1, "seconds": round(time.perf_counter() - started, 3),
                  "gradient_tensors": len(norms), "finite_gradient_norms": all(math.isfinite(value) for value in norms),
                  "modules_with_observed_updates": sorted({name.split(".")[0] for name in changed}),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        record["updates"].append(update)
        save()
        print(json.dumps(update), flush=True)
        return result

    def bounded_train(*values, **options):
        nonlocal model
        bound = inspect.signature(original_train).bind(*values, **options)
        record["original_epochs"] = bound.arguments["num_epochs"]
        bound.arguments["num_epochs"] = args.epochs
        model = bound.arguments["model_name"]
        record["training_arguments"] = {key: value for key, value in bound.arguments.items() if key != "model_name"}
        record["parameters"] = sum(p.numel() for p in model.parameters())
        record["parameters_by_module"] = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
        (source.parent / "model.txt").write_text(str(model), encoding="utf-8")
        save()
        print(json.dumps({"parameters": record["parameters"], "epochs": args.epochs, "gpu": record["runtime"]["gpu"]}), flush=True)
        return original_train(*bound.args, **bound.kwargs)

    train.train_UniBrain = bounded_train
    torch.optim.Adam.step = observe_step
    save()
    try:
        runpy.run_path(str(source / "main.py"), run_name="__main__")
        record["status"] = "completed"
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        traceback.print_exc()
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["source_unchanged"] = all(digest(source / name) == value for name, value in record["source_sha256"].items())
        record["checkpoints"] = [str(path.relative_to(source)) for path in source.glob("result/model/*.pth")]
        if record["status"] == "completed":
            try:
                record["audit"] = audit_results(source, record)
            except Exception as error:
                record["audit"] = {"passed": False, "error": str(error)}
        save()
    if record["status"] != "completed" or not record.get("audit", {}).get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
