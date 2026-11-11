"""Run a budgeted original UniBrain baseline through the deployed backend."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unibrain_agent.catalog import DEFAULTS


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def api(port, token, route, body=None):
    headers = {"Authorization": "Bearer " + token}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"http://127.0.0.1:{port}/api/{route}",
                      data=json.dumps(body).encode() if body is not None else None, headers=headers)
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def wait_for(read, done, interval=10):
    while True:
        result = read()
        if result.get("status") == "failed" or result.get("error"):
            raise RuntimeError(str(result.get("error") or result))
        if done(result):
            return result
        time.sleep(interval)


def gpu_snapshot(index):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    devices = {int(parts[0]): (int(parts[1]), int(parts[2]))
               for parts in (line.split(",") for line in rows.splitlines())}
    memory, utilization = devices[index]
    return {"checked": datetime.now(timezone.utc).isoformat(), "index": index,
            "memory_mib": memory, "utilization_percent": utilization,
            "idle": memory <= 100 and utilization <= 5}


def wait_for_gpu(index, state, path):
    while True:
        if gpu_snapshot(index)["idle"]:
            time.sleep(5)
            snapshot = gpu_snapshot(index)
            if snapshot["idle"]:
                return snapshot
        state["message"] = f"Waiting for GPU {index} to become idle"
        save(path, state)
        time.sleep(120)


def export_fold(args, cohort, folder, fold, target):
    manifest = target / "cohort.json"
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record["scope"] != "full_fold" or record["fold"] != fold or Path(record["source"]) != folder.resolve():
            raise ValueError("Existing fold export does not match the requested full split")
        if all((target / f"adhd_{split}_sample.npy").exists() for split in ("train", "val", "test")):
            return
    if target.exists():
        raise ValueError("Incomplete fold export exists; inspect it before retrying")
    target.parent.mkdir(parents=True, exist_ok=True)
    with (args.workspace / f"export_{cohort}_round{fold}.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(args.backend / "scripts/export_cohort_fold.py"),
                        str(folder), str(target), "--fold", str(fold), "--full"],
                       cwd=args.backend, stdout=log, stderr=subprocess.STDOUT, check=True)


def configure_backend(args, target, cohort, fold):
    runtime_path = args.state / "runtime.json"
    backup = args.workspace / f"runtime_before_{cohort}_round{fold}.json"
    shutil.copy2(runtime_path, backup)
    config = json.loads(runtime_path.read_text(encoding="utf-8"))
    config["data"] = str(target.resolve())
    config["device"] = "cuda:0"
    save(runtime_path, config)
    subprocess.run(["systemctl", "--user", "restart", "unibrain-backend.service"], check=True)
    for _ in range(30):
        try:
            current = api(args.port, args.token, "runtime")
            if current["data"] == str(target.resolve()):
                return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError("Backend did not load the selected full fold")


def attach(args, target):
    job = api(args.port, args.token, "data/jobs", {"mode": "inspect", "sources": [str(target)]})
    inspected = wait_for(lambda: api(args.port, args.token, "data/jobs/" + job["id"]),
                         lambda item: item["status"] == "complete")
    if not any(item["path"] == str(target) and item["status"] == "ready"
               for item in inspected["result"]["items"]):
        raise ValueError("Full fold did not pass backend data inspection")
    selected = api(args.port, args.token, "data/use", {"id": job["id"], "dataset": str(target)})
    if not selected["audit"]["passed"]:
        raise ValueError("Full fold did not pass backend attachment audit")


def prepare(args):
    build = api(args.port, args.token, "prepare",
                {"backend": "rules", "selection": DEFAULTS, "coordination": "single", "steps": args.epochs})
    record = wait_for(lambda: api(args.port, args.token, "builds/" + build["build_id"]),
                      lambda item: item["status"] == "ready")
    if record["profile"] != "original" or record["coordination"] != "single":
        raise ValueError("Prepared pipeline is not the original baseline")
    return record


def run_fold(args, build):
    response = api(args.port, args.token, "builds/" + build["id"] + "/run",
                   {"backend": "rules", "steps": args.epochs, "coordination": "single"})
    if not response["started"]:
        raise RuntimeError("Backend did not start training")
    while True:
        status = api(args.port, args.token, "status")
        if status.get("error"):
            raise RuntimeError(status["error"])
        if status.get("completed"):
            report = api(args.port, args.token, "report")["report"]
            if not report or report.get("status") != "complete" or report.get("build", {}).get("id") != build["id"]:
                raise RuntimeError("Backend has no completed training report")
            return report
        if not status.get("running"):
            raise RuntimeError("Backend stopped without a completed report")
        time.sleep(30)


def evaluate_test(args, target, report):
    run_dir = Path(report["run_dir"])
    checkpoint = run_dir / report["candidate"]["checkpoint"]
    output = run_dir / "test_metrics.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.progress["active"]["test_gpu_check"] = wait_for_gpu(args.gpu, args.progress, args.progress_path)
    save(args.progress_path, args.progress)
    with (run_dir / "test_eval.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(args.backend / "scripts/evaluate_upstream.py"),
                        str(args.backend / "sources/reg_unibrain"), str(target), str(checkpoint), str(output)],
                       cwd=args.backend, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["sample_count"] != result["full_split_count"]:
        raise ValueError("Held-out test evaluation did not cover the full split")
    return result["metrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "cohort_root", "backend", "state"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=18777)
    args = parser.parse_args()
    if not 2 <= args.epochs <= 1000:
        parser.error("Epoch budget must be between 2 and 1000")
    args.workspace = args.workspace.resolve()
    args.cohort_root = args.cohort_root.resolve()
    args.backend = args.backend.resolve()
    args.state = args.state.resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    env_lines = (args.state / "backend.env").read_text(encoding="utf-8").splitlines()
    private = dict(line.split("=", 1) for line in env_lines if "=" in line)
    args.token = private["UNIBRAIN_ACCESS_TOKEN"].strip('"')
    if private.get("CUDA_VISIBLE_DEVICES", "").strip('"') != str(args.gpu):
        raise ValueError("Backend GPU visibility does not match the requested device")
    args.progress_path = args.workspace / "baseline_queue.json"
    args.progress = {"status": "running", "epochs": args.epochs, "profile": "original_budgeted_baseline",
                     "upstream_commit": "f9f8d906b1725046d532f862a28876482aca2cfa", "completed": [],
                     "message": "Starting original baseline queue"}
    if args.progress_path.exists():
        raise ValueError("Baseline queue state already exists; inspect before restarting")
    save(args.progress_path, args.progress)
    try:
        for cohort, name in (("abide", "abide_rebuild_20261110"), ("adhd", "adhd_rebuild_20261110")):
            folder = args.cohort_root / name
            for fold in range(1, 6):
                target = args.workspace / "data" / cohort / f"round{fold}"
                args.progress.update(active={"cohort": cohort, "fold": fold, "stage": "export"},
                                     message=f"Exporting {cohort} round {fold}")
                save(args.progress_path, args.progress)
                export_fold(args, cohort, folder, fold, target)
                args.progress["active"]["stage"] = "attach"
                save(args.progress_path, args.progress)
                configure_backend(args, target, cohort, fold)
                attach(args, target)
                build = prepare(args)
                if Path(build["runtime"]["data"]) != target:
                    raise ValueError("Prepared build selected a different dataset")
                args.progress["active"].update(stage="waiting_for_gpu", build_id=build["id"])
                save(args.progress_path, args.progress)
                args.progress["active"]["train_gpu_check"] = wait_for_gpu(args.gpu, args.progress, args.progress_path)
                args.progress["active"]["stage"] = "training"
                save(args.progress_path, args.progress)
                report = run_fold(args, build)
                args.progress["active"].update(stage="test", run_dir=report["run_dir"])
                save(args.progress_path, args.progress)
                metrics = evaluate_test(args, target, report)
                args.progress["completed"].append({"cohort": cohort, "fold": fold, "run_dir": report["run_dir"],
                                                   "train_gpu_check": args.progress["active"]["train_gpu_check"],
                                                   "test_gpu_check": args.progress["active"]["test_gpu_check"],
                                                   "validation": report["candidate"]["validation"], "test": metrics})
                args.progress["active"] = None
                save(args.progress_path, args.progress)
        summary = {cohort: {metric: {"mean": statistics.mean(row["test"][metric] for row in args.progress["completed"] if row["cohort"] == cohort),
                                    "sd": statistics.stdev(row["test"][metric] for row in args.progress["completed"] if row["cohort"] == cohort)}
                             for metric in ("accuracy", "auc", "macro_f1")}
                   for cohort in ("abide", "adhd")}
        save(args.workspace / "baseline_summary.json", {"epochs": args.epochs, "folds": args.progress["completed"],
                                                         "test_summary": summary})
        args.progress.update(status="complete", active=None, message="Ten original baseline folds completed",
                             finished=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        args.progress.update(status="failed", error=str(error), finished=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        save(args.progress_path, args.progress)


if __name__ == "__main__":
    main()
