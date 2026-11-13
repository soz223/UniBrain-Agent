"""Monitor one owned trainer and resume bounded, recoverable interruptions."""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def retry_reason(code, report, checkpoint, stalled=False):
    if not checkpoint.is_file():
        return None
    error = str(report.get("error", ""))
    if error or report.get("status") == "failed":
        return None
    if stalled and report.get("stop_reason") == "termination_signal":
        return "stalled_worker_checkpointed"
    if code == -9:
        return "killed_worker_checkpoint_rollback"
    return None


def option(command, name, value):
    command = list(command)
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend((name, str(value)))
    return command


def worker_command(command, seconds):
    if seconds is not None:
        return option(command, "--wall-seconds", seconds)
    result = list(command)
    if "--wall-seconds" in result:
        index = result.index("--wall-seconds")
        del result[index:index + 2]
    return result


def diagnose(folder, report, targets, code=None, stalled=False):
    gradient = report.get("gradient_verification", {}).get("last_step") or {}
    paths = {**gradient.get("classification_to_candidates", {}),
             **{name + ":total": value for name, value in gradient.get("total_to_candidates", {}).items()}}
    faults = {name: {key: value.get(key) for key in ("finite", "nonzero", "observed_tensors", "tensor_count", "l1")}
              for name, value in paths.items() if not value.get("finite") or not value.get("nonzero")}
    history = report.get("history", [])
    latest = history[-1] if history else {}
    metrics = latest.get("validation", {})
    train_auc, val_auc = latest.get("training_probe", {}).get("auc"), metrics.get("auc")
    warnings = []
    if train_auc is not None and val_auc is not None and train_auc - val_auc > .1:
        warnings.append("training_validation_gap")
    if len(history) >= 5 and all(row.get("validation", {}).get("auc", 1) <= .55 for row in history[-5:]):
        warnings.append("five_validation_epochs_near_chance")
    error = str(report.get("error", ""))
    kind = ("gradient_path_failure" if "gradient" in error.lower() else "out_of_memory" if "out of memory" in error.lower()
            else "worker_stall" if stalled else "worker_killed" if code == -9 else "worker_error" if error else "validation_review")
    result = {"updated_at": datetime.now(timezone.utc).isoformat(), "kind": kind, "step": report.get("steps"),
              "epoch": latest.get("epoch"), "warnings": warnings, "gradient_evidence": faults,
              "targets": {name: {"target": target, "observed": metrics.get(name),
                                  "met": metrics[name] >= target if metrics.get(name) is not None else None}
                          for name, target in targets.items()},
              "suggested_checks": ["supervision_and_label_provenance", "spatial_alignment", "metric_contract", "native_gradient_and_parameter_updates"],
              "checkpoint_available": (folder / "last.pt").is_file(), "automatic_hyperparameter_changes": False,
              "scope": "Validation monitoring; target comparison is not five-fold test reproduction"}
    atomic_json(folder / "training_diagnosis.json", result)
    return result


def gpu_available(environment):
    device = environment.get("CUDA_VISIBLE_DEVICES")
    if not device or not device.startswith("GPU-"):
        return
    result = subprocess.run(["nvidia-smi", "--id=" + device,
                             "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    memory, utilization = (int(value.strip()) for value in result.stdout.strip().split(","))
    if memory > 256 or utilization > 10:
        raise RuntimeError("Selected GPU is occupied; no worker launched")


def supervise(spec):
    folder = Path(spec["output"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    command = spec["command"]
    if ("--verify-gradients" not in command or "--output" not in command
            or Path(command[command.index("--output") + 1]).resolve() != folder):
        raise ValueError("Supervisor requires gradient verification and matching output directory")
    seconds = float(spec["wall_seconds"]) if spec.get("wall_seconds") is not None else None
    retries = int(spec.get("max_restarts", 2))
    stall_seconds = float(spec.get("stall_seconds", 900))
    if (seconds is not None and not 0 < seconds < float("inf")) or not 0 <= retries <= 5 or stall_seconds < 60:
        raise ValueError("Invalid supervision limits")
    lock = folder / "supervisor.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stopped = []
    handlers = {sig: signal.signal(sig, lambda number, frame: stopped.append(number))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    state = {"status": "starting", "restarts": 0, "consecutive_restarts": 0, "pid": os.getpid(), "max_restarts": retries,
             "wall_seconds": seconds, "automatic_actions": ["scaled_gradient_diagnosis", "checkpoint_resume"],
             "validation_used_for_repair": False}
    child = None
    reviewed_epoch = None

    def publish(status, **values):
        state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(folder / "supervisor.json", state)

    def event(action, **values):
        with (folder / "supervisor_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "action": action, **values}) + "\n")

    try:
        while not stopped:
            remaining = seconds - (time.monotonic() - started) if seconds is not None else None
            if remaining is not None and remaining <= 0:
                publish("budget_exhausted")
                break
            try:
                gpu_available(spec.get("environment", {}))
            except RuntimeError:
                publish("waiting_for_gpu")
                time.sleep(5)
                continue
            launch = worker_command(command, remaining)
            attempt_step = read_json(folder / "report.json").get("steps", 0)
            if state["restarts"]:
                launch = option(launch, "--resume", folder / "last.pt")
            with (folder / "supervised_worker.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(launch, cwd=spec["cwd"], env={**os.environ, **spec.get("environment", {})},
                                         stdout=log, stderr=subprocess.STDOUT)
                event("launch", pid=child.pid, restart=state["restarts"])
                publish("running", worker_pid=child.pid)
                last_change = time.monotonic()
                signature = None
                stalled = False
                was_stalled = False
                requested_stop = None
                while child.poll() is None:
                    files = [folder / name for name in ("updates.jsonl", "report.json", "health.json", "last.pt")]
                    current = [(p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files]
                    if current != signature:
                        signature, last_change = current, time.monotonic()
                    elapsed = time.monotonic() - started
                    stalled = time.monotonic() - last_change > stall_seconds
                    if requested_stop is None and (stopped or (seconds is not None and elapsed >= seconds) or stalled):
                        was_stalled = stalled
                        child.terminate()
                        requested_stop = time.monotonic()
                        publish("checkpointing", reason="user_stop" if stopped else "stall" if stalled else "budget")
                    if requested_stop is not None and time.monotonic() - requested_stop > 180:
                        child.kill()
                        child.wait()
                        publish("needs_attention", reason="worker_did_not_checkpoint")
                        return state
                    if requested_stop is None:
                        publish("running", heartbeat_age_seconds=round(time.monotonic() - last_change, 1))
                    report = read_json(folder / "report.json")
                    history = report.get("history", [])
                    epoch = history[-1].get("epoch") if history else None
                    if report.get("status") == "running" and epoch is not None and epoch != reviewed_epoch:
                        diagnosis = diagnose(folder, report, spec.get("targets", {}))
                        reviewed_epoch = epoch
                        event("validation_review", epoch=epoch, warnings=diagnosis["warnings"])
                    time.sleep(5)
                report = read_json(folder / "report.json")
                reason = retry_reason(child.returncode, report, folder / "last.pt", was_stalled)
                diagnosis = diagnose(folder, report, spec.get("targets", {}), child.returncode, was_stalled)
                if report.get("steps", 0) >= attempt_step + 100:
                    state["consecutive_restarts"] = 0
                failure_key = [report.get("steps"), report.get("error")]
                if (reason is None and diagnosis["kind"] == "gradient_path_failure"
                        and diagnosis["checkpoint_available"] and state["consecutive_restarts"] == 0
                        and state.get("last_gradient_failure") != failure_key):
                    reason = "checkpoint_replay_to_verify_gradient_failure"
                if diagnosis["kind"] == "gradient_path_failure":
                    state["last_gradient_failure"] = failure_key
                event("worker_exit", code=child.returncode, reason=reason, report_status=report.get("status"))
                if stopped or (seconds is not None and time.monotonic() - started >= seconds):
                    publish("stopped" if stopped else "budget_exhausted")
                    break
                if reason and state["consecutive_restarts"] < retries:
                    state["restarts"] += 1
                    state["consecutive_restarts"] += 1
                    publish("recovering", reason=reason)
                    continue
                publish("complete" if child.returncode == 0 and report.get("status") == "complete"
                        else "budget_exhausted" if child.returncode == 0 and report.get("status") == "budget_exhausted"
                        else "needs_attention", error=report.get("error"), returncode=child.returncode)
                break
    except Exception as error:
        publish("needs_attention", error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        lock.unlink(missing_ok=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = supervise(read_json(args.manifest))
    if result["status"] == "needs_attention":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
