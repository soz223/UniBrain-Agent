import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .catalog import DEFAULTS
from .paths import CATALOG, PROJECT, RUNS
from .runner import write_json
from .runtime import settings


def verify_original(source):
    source = Path(source)
    manifest = json.loads((CATALOG / "sources.json").read_text(encoding="utf-8"))
    spec = next(item for item in manifest["repositories"] if item["directory"] == "reg_unibrain")
    files = [item for item in spec["files"] if item["path"] in {"main.py", "train.py", "model.py", "utils.py"}]
    if len(files) != 4:
        raise ValueError("Original source manifest is incomplete")
    for item in files:
        path = source / item["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != item["sha256"]:
            raise ValueError("Missing or modified original source: " + item["path"] + ". Run scripts/bootstrap.py --source reg_unibrain --samples.")
    return [{"repository": spec["repository"], "commit": spec["commit"], "verified_files": 4, "verified_clean": True}]


def original_environment(config):
    env = os.environ.copy()
    device = config["device"]
    if device == "cpu":
        raise ValueError("Original UniBrain uses CUDA directly. Select auto or cuda:N in the runtime configuration.")
    if device not in {"auto", "cuda"} and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("Original training supports auto, cuda or cuda:N")
    if device.startswith("cuda:"):
        available = env.get("CUDA_VISIBLE_DEVICES")
        index = int(device.split(":")[1])
        if available is not None and (not available or index >= len(available.split(","))):
            raise ValueError("Requested CUDA index is outside the visible GPU allocation")
        env["CUDA_VISIBLE_DEVICES"] = available.split(",")[index] if available else str(index)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    return env


def check_original_runtime(config):
    verify_original(config["original_source"])
    probe = "import json, torch, torchvision, torch_geometric, SimpleITK, nibabel, sklearn, psutil; assert torch.cuda.is_available(), 'CUDA unavailable'; print(json.dumps({'torch': torch.__version__, 'device': torch.cuda.get_device_name(0)}))"
    result = subprocess.run([config["original_python"], "-c", probe], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=original_environment(config), timeout=90,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise ValueError("Original runtime check failed. Install requirements-upstream.txt and requirements.txt in a CUDA environment. " + result.stderr[-600:])
    for split in ("train", "val", "test"):
        if not (Path(config["data"]) / f"adhd_{split}_sample.npy").exists():
            raise ValueError("Missing data split: " + split)
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_original(epochs=2, build=None, on_event=None, output_dir=None):
    config = build.get("runtime", settings()) if build else settings()
    check_original_runtime(config)
    folder = Path(output_dir) if output_dir else RUNS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "profile": "original", "mode": "single", "created": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(folder.resolve()), "upstream_commit": "f9f8d906b1725046d532f862a28876482aca2cfa",
              "backend": "original", "epochs": epochs, "events": [], "seed": None,
              "build": {"id": build["id"], "selection": build["selection"]} if build else None}

    def emit(event):
        event.setdefault("seconds", round(time.perf_counter() - started, 3))
        report["events"].append(event)
        with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        if event["kind"] != "module":
            write_json(folder / "report.json", report)
        if on_event:
            on_event(event, folder)

    try:
        emit({"kind": "phase", "phase": "preparing", "message": "Loading the original UniBrain training entry point"})
        command = [config["original_python"], str(PROJECT / "scripts/run_upstream.py"), config["original_source"],
                   "--epochs", str(epochs), "--data-dir", config["data"], "--output-dir", str(folder),
                   "--threads", str(config["threads"]), "--telemetry"]
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", env=original_environment(config),
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("UNIBRAIN_EVENT "):
                    emit(json.loads(line.removeprefix("UNIBRAIN_EVENT ")))
            code = process.wait()
        if code:
            tail = (folder / "run.log").read_text(encoding="utf-8")[-1800:]
            raise RuntimeError("Original UniBrain stopped. " + tail)
        verification = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
        if not verification.get("audit", {}).get("passed"):
            raise RuntimeError("Original UniBrain verification did not pass")
        validation = {}
        lines = [line for path in folder.glob("result/loss_log/*_val.txt") for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("Epoch[")]
        for key, pattern in {"extraction_dice": r"Ext Dice:\s*([\d.]+)", "tissue_dice": r"Seg pred Dice:\s*([\d.]+)", "accuracy": r"acc_val:\s*([\d.]+)"}.items():
            match = re.search(pattern, lines[-1]) if lines else None
            validation[key] = float(match.group(1)) if match else None
        counts = verification["audit"]["sample_counts"]
        report.update(status="complete", native=verification, elapsed_seconds=time.perf_counter() - started,
                      data={"splits": {key: {"count": value} for key, value in counts.items()}},
                      environment={"device": "cuda", "threads": config["threads"]},
                      candidate={"steps": len(verification["updates"]), "history": verification["updates"],
                                 "training_seconds": verification["elapsed_seconds"], "coordination": None,
                                 "validation": {**validation, "count": counts["val"]},
                                 "checkpoint": verification["checkpoints"][-1], "model": {"config": {"stages": build["selection"] if build else DEFAULTS}}})
        emit({"kind": "complete", "message": "Original UniBrain finished. Source hashes, updates and checkpoint verified."})
        write_json(folder / "report.json", report)
        from .report import render
        render(report, folder)
        write_json(RUNS / "latest.json", {"run_dir": str(folder.resolve())})
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        emit({"kind": "failed", "message": str(error)})
        write_json(folder / "report.json", report)
        raise
