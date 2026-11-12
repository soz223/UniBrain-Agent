"""Authenticated dashboard bridge for bounded full-model research cycles."""

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .paths import CONFIG, PROJECT, configured_path
from .providers import PROVIDERS
from .owned_process import OwnedProcessCleanupError, run_logged


STAGES = ("denoising", "bias_correction", "extraction", "extraction_registration", "normalization", "harmonization", "registration",
          "segmentation", "parcellation", "network_generation", "classification")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def tail_updates(path):
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - 262144)
        stream.seek(start)
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    rows = []
    for line in lines[1:] if start else lines:
        try:
            value = json.loads(line)
            rows.append({key: value[key] for key in ("step", "epoch", "phase", "total_loss", "losses", "gradient_policy", "gradient_verification") if key in value})
        except (ValueError, TypeError):
            continue
    return rows[-100:]


class CycleWorkspace:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else configured_path("research_cycles", PROJECT / "builds" / "research_cycles")
        self.lock = threading.RLock()
        self.latest = read_json(self.folder / "latest.json").get("id")
        self.active = None

    def path(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", identifier):
            raise ValueError("Invalid research cycle ID")
        result = (self.folder / identifier).resolve()
        if result.parent != self.folder.resolve():
            raise ValueError("Research cycle is outside its configured directory")
        return result

    def defaults(self, runtime, selected=None):
        device = runtime.get("device", "auto")
        gpu = int(device.split(":")[-1]) if re.fullmatch(r"cuda:\d+", device) else 0
        return {"source": runtime["original_source"], "data": (selected or {}).get("path", runtime["data"]),
                "gpu": gpu, "stage": "classification", "stages": list(STAGES),
                "acceptance_steps": 1, "train_steps": 2, "coordination": "none",
                "scope": "Full architecture, bounded execution check; not a cohort performance experiment"}

    def create(self, body, runtime, selected, provider):
        allowed = {"task", "stage", "source", "data", "plans", "dataset", "fold", "gpu", "seed",
                   "acceptance_steps", "train_steps", "coordination", "mode"}
        if set(body) - allowed:
            raise ValueError("Unknown research cycle option")
        config = self.defaults(runtime, selected)
        config.update({key: value for key, value in body.items() if value != ""})
        task = config.get("task", "")
        if not isinstance(task, str) or not task.strip() or len(task) > 4000:
            raise ValueError("Describe the task in 1 to 4000 characters")
        if config.get("stage") not in STAGES and config.get("stage") != "all":
            raise ValueError("Unknown research stage")
        if config["coordination"] not in ("none", "taskforce"):
            raise ValueError("Unknown training coordination")
        if config.get("mode", "run") not in ("run", "preflight"):
            raise ValueError("Choose a prerequisite check or a research cycle")
        for name, default, low, high in (("gpu", 0, 0, 1024), ("seed", 17, 0, 2147483647),
                                       ("acceptance_steps", 1, 1, 2), ("train_steps", 2, 1, 1000)):
            value = config.get(name, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid " + name)
            config[name] = value
        if config.get("dataset") not in (None, "abide", "adhd"):
            raise ValueError("Choose ABIDE or ADHD, or leave the dataset unspecified")
        if config.get("fold") is not None and (type(config["fold"]) is not int or not 1 <= config["fold"] <= 5):
            raise ValueError("Fold must be an integer from 1 to 5")
        for name in ("source", "data", "plans"):
            if name not in config:
                continue
            value = config[name]
            if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
                raise ValueError("Invalid backend " + name + " path")
            path = Path(os.path.expandvars(value)).expanduser()
            path = (path if path.is_absolute() else CONFIG.parent / path).resolve()
            if not (path.is_file() if name == "plans" else path.is_dir()):
                raise ValueError("Backend path does not exist: " + name)
            config[name] = str(path)
        with self.lock:
            if self.active:
                raise ValueError("A full research cycle is already running")
            identifier = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            root = self.path(identifier)
            root.mkdir(parents=True)
            command = [runtime["original_python"], str(PROJECT / "scripts/run_research_cycle.py"),
                       "--source", config["source"], "--data", config["data"], "--output", str(root / "cycle"),
                       "--task", task.strip(), "--provider", provider["provider"], "--coordination", config["coordination"],
                       "--checkpoint-mode", "weights"]
            for key in ("gpu", "seed", "acceptance_steps", "train_steps", "plans", "dataset", "fold"):
                if config.get(key) is not None:
                    command.extend(("--" + key.replace("_", "-"), str(config[key])))
            if config["stage"] != "all":
                command.extend(("--stage", config["stage"]))
            if provider.get("model"):
                command.extend(("--model", provider["model"]))
            if config.get("mode") == "preflight":
                command.append("--preflight")
            record = {"id": identifier, "status": "running", "started_at": time.time(),
                      "provider": provider["provider"], "config": config, "command": command}
            save_json(root / "job.json", record)
            save_json(self.folder / "latest.json", {"id": identifier})
            self.active = self.latest = identifier
            return {"id": identifier, "started": True}

    def execute(self, identifier, provider, ai_path):
        root = self.path(identifier)
        record = read_json(root / "job.json")
        environment = dict(os.environ)
        environment["UNIBRAIN_AI_SETTINGS"] = str(Path(ai_path).resolve())
        environment["PYTHONIOENCODING"] = "utf-8"
        key = PROVIDERS[provider["provider"]]["env"]
        if key and provider.get("api_key"):
            environment[key] = provider["api_key"]
        try:
            exit_code = run_logged(record["command"], root / "process.log", None, cwd=PROJECT, env=environment)
            report = read_json(root / "cycle/report.json")
            status = report.get("status", "failed")
            if status == "running" or exit_code and status in {"ready", "complete"}:
                status = "failed"
            record.update(status=status, exit_code=exit_code,
                          error=None if report else "Worker produced no report; inspect the private process log")
        except OwnedProcessCleanupError:
            record.update(status="blocked", cleanup_failed=True,
                          error="Process cleanup could not be confirmed. Inspect the backend before another run.")
        except Exception:
            record.update(status="failed", error="Research worker could not start; inspect the backend runtime configuration")
        finally:
            record["finished_at"] = time.time()
            with self.lock:
                save_json(root / "job.json", record)
                self.active = None

    def read(self, identifier):
        root = self.path(identifier)
        job = read_json(root / "job.json")
        if not job:
            raise FileNotFoundError("Research cycle not found")
        report = read_json(root / "cycle/report.json")
        integration = read_json(root / "cycle/integration/result.json") or report.get("integration", {})
        training = read_json(root / "cycle/training/report.json") or report.get("training", {})
        updates = tail_updates(root / "cycle/training/updates.jsonl")
        running = job["status"] == "running" and self.active == identifier
        status = "running" if running else "interrupted" if job["status"] == "running" else job["status"]
        checks = {}
        calls = []
        for item in integration.get("calls", []):
            calls.append({key: item[key] for key in ("role", "usage", "adapter_sha256") if key in item})
            if item.get("evaluation"):
                checks = item["evaluation"].get("checks", {})
        proposals = sorted((root / "cycle/integration").glob("proposal_*"))
        if proposals:
            observed = read_json(proposals[-1] / "evaluation.json")
            checks = observed.get("checks", checks)
        choice = report.get("selection", {}).get("choice", {})
        blockers = report.get("blockers", [])
        coordination = dict(training.get("coordination", {}))
        gradients = dict(training.get("gradient_verification", {}))
        if updates and updates[-1].get("gradient_policy"):
            latest_policy = updates[-1]["gradient_policy"]
            coordination.update(mode="taskforce", policy_updates=latest_policy.get("policy_updates", 0),
                                last_decision=latest_policy, observed_step=updates[-1]["step"])
        if updates and updates[-1].get("gradient_verification"):
            gradients.update(last_step=updates[-1]["gradient_verification"], observed_step=updates[-1]["step"])
        return {"id": identifier, "status": status, "phase": report.get("phase", "starting"),
                "elapsed_seconds": round(job.get("finished_at", time.time()) - job["started_at"], 1),
                "provider": report.get("provider", {"provider": job["provider"]}), "selection": choice,
                "method": report.get("method", choice.get("method")), "config": job["config"],
                "checks": checks, "calls": calls, "proposal_count": len(proposals),
                "selection_usage": report.get("selection", {}).get("usage"),
                "actual_updates": {"acceptance": integration.get("optimizer_updates", report.get("actual_updates", {}).get("acceptance", 0)),
                                   "training": updates[-1]["step"] if updates else training.get("steps", 0)},
                "updates": updates, "coordination": coordination,
                "gradients": gradients,
                "accepted_adapter": report.get("accepted_adapter"),
                "adapter_unchanged": report.get("accepted_adapter_unchanged_after_training"),
                "error": job.get("error") or report.get("error"), "blockers": blockers,
                "cleanup_failed": job.get("cleanup_failed", False),
                "scope": "Bounded full-model execution check; accuracy and Agent effectiveness require cohort experiments"}

    def code(self, identifier):
        root = self.path(identifier)
        report = read_json(root / "cycle/report.json")
        expected = (report.get("accepted_adapter") or {}).get("sha256")
        path = root / "cycle/accepted_adapter.py"
        if not expected or not path.is_file():
            raise ValueError("This cycle has no accepted adapter yet")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("The accepted adapter changed after validation")
        return {"name": "accepted_adapter.py", "sha256": actual, "code": path.read_text(encoding="utf-8")}
