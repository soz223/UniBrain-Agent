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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def training_values():
    frame = inspect.currentframe().f_back
    try:
        while frame and "loss_train" not in frame.f_locals:
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("Original training scalars were not found in the optimizer call stack")
        losses = {key: float(frame.f_locals[name].detach()) for key, name in
                  {"extraction": "ext_loss", "registration": "sim_loss_train", "segmentation": "seg_loss_train",
                   "classification": "classifier_loss_train"}.items()}
        return float(frame.f_locals["loss_train"].detach()), losses
    finally:
        del frame


def training_frame():
    frame = inspect.currentframe().f_back
    while frame and "loss_train" not in frame.f_locals:
        frame = frame.f_back
    return frame


def audit_results(source, record, data_dir=None, work_dir=None):
    import numpy as np
    import torch

    data_dir, work_dir = data_dir or source / "dataset", work_dir or source
    counts = {split: len(np.load(data_dir / f"adhd_{split}_sample.npy", mmap_mode="r", allow_pickle=False))
              for split in ("train", "val", "test")}
    validation_lines = [line for path in work_dir.glob("result/loss_log/*_val.txt")
                        for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("Epoch[")]
    checkpoints = []
    for path in work_dir.glob("result/model/*.pth"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        finite = all(bool(torch.isfinite(value).all()) for value in state.values() if value.is_floating_point())
        checkpoints.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
                            "state_tensors": len(state), "finite": finite})
        del state
    expected = counts["train"] * record["epochs"]
    checks_so_far = record.get("step_checks", {})
    updates = record["updates"]
    update_count = record.get("update_count", len(updates))
    active_modules = {"ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred"}
    checks = {"source_unchanged": record["source_unchanged"], "training_completed": record["status"] == "completed",
              "all_training_updates": update_count == expected,
              "finite_gradient_norms": bool(update_count) and checks_so_far.get("finite_gradient_norms", all(step["finite_gradient_norms"] for step in updates)),
              "active_modules_updated": bool(update_count) and checks_so_far.get("active_modules_updated", all(active_modules.issubset(step["modules_with_observed_updates"]) for step in updates)),
              "all_validation_epochs": len(validation_lines) == record["epochs"],
              "checkpoint_saved": bool(checkpoints), "checkpoint_finite": bool(checkpoints) and all(item["finite"] for item in checkpoints)}
    if record.get("gradient_policy"):
        checks["policy_applied_every_step"] = bool(update_count) and checks_so_far.get("policy_applied_every_step", all("gradient_policy" in step for step in updates))
        checks["policy_checkpoint_saved"] = (work_dir / record.get("policy_checkpoint", "missing")).is_file()
        checks["policy_learned"] = expected < 5 or record.get("policy_updates", 0) > 0
    return {"passed": all(checks.values()), "checks": checks, "sample_counts": counts,
            "expected_updates": expected, "validation_epochs": len(validation_lines), "checkpoints": checkpoints,
            "dataset_sha256": {path.name: digest(path) for path in sorted(data_dir.glob("*.npy"))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gradient-policy", action="store_true")
    parser.add_argument("--taskforce-profile", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--coordination-batch-size", type=int, default=64)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.epochs < 2:
        parser.error("Use at least two epochs to exercise upstream checkpoint saving")
    manifest = json.loads((ROOT / "catalog/sources.json").read_text(encoding="utf-8"))
    pinned = next(item for item in manifest["repositories"] if item["directory"] == "reg_unibrain")
    for item in pinned["files"]:
        if item["path"] in {"main.py", "model.py", "train.py", "utils.py"}:
            actual = hashlib.sha256((source / item["path"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if actual != item["sha256"]:
                raise RuntimeError("Pinned UniBrain source changed: " + item["path"])
    data_dir = (args.data_dir or source / "dataset").resolve()
    work_dir = args.output_dir.resolve() if args.output_dir else source
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    sys.path.insert(0, str(source))
    import torch
    import torchvision
    import torch_geometric
    import train
    import numpy as np
    import psutil

    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("Upstream uses CUDA tensors directly; use a CUDA environment")
    record = {"commit": COMMIT, "status": "running", "epochs": args.epochs,
              "runtime": {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                          "torch_geometric": torch_geometric.__version__, "gpu": torch.cuda.get_device_name()},
              "source_sha256": {name: digest(source / name) for name in ("main.py", "model.py", "train.py", "utils.py")},
              "changes": ["Epoch budget only", "Read-only optimizer observations"], "updates": [],
              "update_count": 0, "step_checks": {"finite_gradient_norms": True, "active_modules_updated": True,
                                                  "policy_applied_every_step": True},
              "gradient_policy": "taskwise_actor_critic" if args.gradient_policy else None}
    if args.gradient_policy:
        record["changes"] = ["Epoch budget", "Task-wise actor-critic gradient aggregation", "Optimizer observations"]
    output = (work_dir if args.output_dir else source.parent) / "verification.json"
    process = psutil.Process()
    process.cpu_percent()
    from unibrain_agent.data import OriginalDataset
    counts = {split: len(OriginalDataset(data_dir / f"adhd_{split}_sample.npy")) for split in ("train", "val", "test")}
    if args.data_dir or args.output_dir:
        from types import SimpleNamespace
        train.load_data_no_fix_v2 = lambda name, size, ifshuffle=True: torch.utils.data.DataLoader(
            OriginalDataset(data_dir / name), batch_size=size, shuffle=ifshuffle)
        train.np = SimpleNamespace(**{name: getattr(np, name) for name in dir(np)})
        train.np.load = lambda path, *a, **k: np.load(data_dir / Path(path).name, *a, **k)
        record["changes"].append("External data and output paths with memory mapped input; original training equations unchanged")

    def save():
        output.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")

    original_train = train.train_UniBrain
    original_step = torch.optim.Adam.step
    original_backward = torch.Tensor.backward
    model = None
    model_parameter_ids = set()
    active_modules = {"ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred"}
    team = None
    policy_parameters = []
    objective_step = None
    policy_step = None
    started = time.perf_counter()
    current_phase = None
    validation_count = 0
    handles = []

    def emit(kind, message, **extra):
        if not args.telemetry:
            return
        resources = {"cpu_percent": round(process.cpu_percent() / (psutil.cpu_count() or 1), 1),
                     "rss_mb": round(process.memory_info().rss / 1024 ** 2, 1), "device": "CUDA",
                     "threads": args.threads, "gpu_name": torch.cuda.get_device_name(),
                     "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                     "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
                     "scope": "Original UniBrain worker process"}
        event = {"kind": kind, "message": message, "seconds": round(time.perf_counter() - started, 3), **extra,
                 "resources": resources}
        print("UNIBRAIN_EVENT " + json.dumps(event, allow_nan=False), flush=True)

    def before_forward(module, values, options):
        nonlocal current_phase
        phase = "training" if torch.is_grad_enabled() else "validation"
        current_phase = phase
        emit("phase", "Original UniBrain model is running" if phase == "training" else "Validating original UniBrain", phase=phase)

    def after_forward(module, values, output):
        nonlocal validation_count
        if current_phase == "validation":
            validation_count += 1
            emit("validation_progress", f"Validation sample {validation_count}/{counts['val'] * args.epochs}",
                 completed=validation_count, total=counts["val"] * args.epochs)
            if args.telemetry and validation_count % counts["val"] == 0:
                np.savez_compressed(work_dir / "candidate_preview.npz", input=values[1].detach().cpu().numpy(),
                                    mask=output[1][-1].detach().cpu().numpy(), extracted=output[0][-1].detach().cpu().numpy(),
                                    warped=output[2][-1].detach().cpu().numpy(), edges=output[9].detach().cpu().numpy())

    def attach(name, module):
        tick = [0.0]
        def before(module, values):
            tick[0] = time.perf_counter()
            emit("module", name + " is running", module=name, label=name, action="started")
        def after(module, values, output):
            emit("module", name + " finished", module=name, label=name, action="completed", duration=time.perf_counter() - tick[0])
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

    def observe_step(optimizer, *values, **options):
        nonlocal policy_step
        if not any(id(parameter) in model_parameter_ids for group in optimizer.param_groups for parameter in group["params"]):
            return original_step(optimizer, *values, **options)
        total_loss, losses = training_values()
        if policy_step is not None:
            total_loss = policy_step["weighted_loss"]
        before = {name: parameter.detach().flatten()[::max(1, parameter.numel() // 32)][:32].clone()
                  for name, parameter in model.named_parameters() if parameter.grad is not None}
        norms = [float(parameter.grad.norm()) for parameter in model.parameters() if parameter.grad is not None]
        result = original_step(optimizer, *values, **options)
        if policy_step is not None and hasattr(team, "after_model_step"):
            policy_step.update(team.after_model_step())
        changed = [name for name, parameter in model.named_parameters() if name in before and
                   not torch.equal(before[name], parameter.detach().flatten()[::max(1, parameter.numel() // 32)][:32])]
        update = {"step": record["update_count"] + 1, "seconds": round(time.perf_counter() - started, 3),
                  "objective": objective_step,
                  "gradient_tensors": len(norms), "finite_gradient_norms": all(math.isfinite(value) for value in norms),
                  "modules_with_observed_updates": sorted({name.split(".")[0] for name in changed}),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        record["update_count"] = update["step"]
        record["updates"].append(update)
        record["updates"] = record["updates"][-500:]
        record["step_checks"]["finite_gradient_norms"] &= update["finite_gradient_norms"]
        record["step_checks"]["active_modules_updated"] &= active_modules.issubset(update["modules_with_observed_updates"])
        if total_loss is not None:
            update.update(weighted_loss=total_loss, losses=losses,
                          weights=policy_step["weights"] if policy_step else
                          {term['id']: term['weight'] for term in objective_step['terms']} if objective_step else {})
            if policy_step:
                update["gradient_policy"] = policy_step
            record["step_checks"]["policy_applied_every_step"] &= "gradient_policy" in update if args.gradient_policy else True
            emit("training", f"Original UniBrain: update {update['step']}/{counts['train'] * args.epochs}", **update)
        policy_step = None
        with (work_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(update, allow_nan=False) + "\n")
        if update["step"] == 1 or update["step"] % 25 == 0:
            save()
        print(json.dumps(update), flush=True)
        return result

    def bounded_train(*values, **options):
        nonlocal model, team, model_parameter_ids, policy_parameters
        bound = inspect.signature(original_train).bind(*values, **options)
        record["original_epochs"] = bound.arguments["num_epochs"]
        bound.arguments["num_epochs"] = args.epochs
        model = bound.arguments["model_name"]
        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        if args.gradient_policy:
            from unibrain_agent.taskforce_paper import make_team, policy_parameters as select_parameters
            team = make_team((bound.arguments["lambda_ext"], bound.arguments["lambda_reg"],
                              bound.arguments["lambda_seg"], bound.arguments["lambda_cls"]),
                             profile=args.taskforce_profile, batch_size=args.coordination_batch_size)
            policy_parameters, probe_names = select_parameters(model, args.taskforce_profile)
            record["taskforce_profile"] = args.taskforce_profile
            record["gradient_probes"] = {"parameter_tensors": len(policy_parameters),
                                         "parameter_values": sum(parameter.numel() for parameter in policy_parameters),
                                         "modules": sorted({name.split(".")[0] for name in probe_names})}
        record["training_arguments"] = {key: value for key, value in bound.arguments.items() if key != "model_name"}
        record["parameters"] = sum(p.numel() for p in model.parameters())
        record["parameters_by_module"] = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
        output.with_name("model.txt").write_text(str(model), encoding="utf-8")
        if args.telemetry:
            handles.extend([model.register_forward_pre_hook(before_forward, with_kwargs=True), model.register_forward_hook(after_forward)])
            for name, module in model.named_children():
                attach(name, module)
        emit("budget", "Original UniBrain loaded: full widths, five registration stages", steps=counts["train"] * args.epochs)
        save()
        print(json.dumps({"parameters": record["parameters"], "epochs": args.epochs, "gpu": record["runtime"]["gpu"]}), flush=True)
        return original_train(*bound.args, **bound.kwargs)

    train.train_UniBrain = bounded_train
    torch.optim.Adam.step = observe_step
    def coordinated_backward(tensor, *values, **options):
        nonlocal policy_step, objective_step
        frame = training_frame()
        if frame is None or frame.f_locals["loss_train"] is not tensor:
            return original_backward(tensor, *values, **options)
        terms = {key: frame.f_locals[name] for key, name in
                 {"extraction": "ext_loss", "registration": "sim_loss_train",
                  "segmentation": "seg_loss_train", "classification": "classifier_loss_train"}.items()}
        weights = {task: record['training_arguments'][key] for task, key in
                   [('extraction', 'lambda_ext'), ('registration', 'lambda_reg'), ('segmentation', 'lambda_seg'), ('classification', 'lambda_cls')]}
        total = tensor
        if team is not None:
            total, decision = team.combine(terms, policy_parameters)
            weights = decision['weights']
            policy_step = {**decision, "weighted_loss": float(total.detach())}
            emit("policy", "Task-wise gradient policy updated the training mixture",
                 step=record["update_count"] + 1, weights=weights,
                 reward=decision["reward"], policy_updates=decision["policy_updates"])
        from unibrain_agent.objective import observe_objective
        from unibrain_agent.catalog import DEFAULTS
        objective_step = observe_objective(model, terms, weights, total, record['update_count'] + 1)
        objective_step.update(selection=dict(DEFAULTS), coordination='taskforce' if team else 'single')
        return original_backward(total, *values, **options)
    torch.Tensor.backward = coordinated_backward
    save()
    try:
        runpy.run_path(str(source / "main.py"), run_name="__main__")
        record["status"] = "completed"
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        traceback.print_exc()
    finally:
        torch.Tensor.backward = original_backward
        torch.optim.Adam.step = original_step
        if team is not None:
            team.save(work_dir / "gradient_policy.pt")
            record["policy_updates"] = team.updates
            record["policy_checkpoint"] = "gradient_policy.pt"
        record["elapsed_seconds"] = time.perf_counter() - started
        record["source_unchanged"] = all(digest(source / name) == value for name, value in record["source_sha256"].items())
        record["checkpoints"] = [str(path.relative_to(work_dir)) for path in work_dir.glob("result/model/*.pth")]
        for handle in handles:
            handle.remove()
        if record["status"] == "completed":
            try:
                emit("phase", "Verifying original checkpoints and parameter updates", phase="saving")
                record["audit"] = audit_results(source, record, data_dir, work_dir)
            except Exception as error:
                record["audit"] = {"passed": False, "error": str(error)}
        save()
    if record["status"] != "completed" or not record.get("audit", {}).get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
