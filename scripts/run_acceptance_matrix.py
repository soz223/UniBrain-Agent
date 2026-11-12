"""Run bounded cohort checks with strict data and child-process accounting."""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unibrain_agent.data import digest

METHODS = {"original", "synthstrip", "voxelmorph", "lapirn", "nnunet", "redwgan", "deepn4", "deepnormalize",
           "harmonizing_flows", "ernet", "fastsurfer", "nobrainer", "fbnetgen", "brainnetcnn", "bnt"}
TARGETS = {"synthstrip": ("ext_net",), "voxelmorph": ("reg_net",), "lapirn": ("reg_net",), "nnunet": ("seg_net",),
           "redwgan": ("preprocessor",), "deepn4": ("preprocessor",), "deepnormalize": ("preprocessor",),
           "harmonizing_flows": ("preprocessor",), "ernet": ("ext_net", "reg_net"), "fastsurfer": ("par_net",),
           "nobrainer": ("par_net",), "fbnetgen": ("graph_gen",), "brainnetcnn": ("gnn_pred",), "bnt": ("gnn_pred",)}
TEMPLATES = ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")
SPLITS = ("train", "val", "test")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def subset_audit(directory, dataset, fold):
    from unibrain_agent.cohort_provenance import validate_cohort
    directory = Path(directory)
    scope = "acceptance_subset_not_full_cohort"
    validate_cohort(directory, dataset, fold, required_scope=scope, allow_missing=False)
    names = [f"adhd_{split}_sample.npy" for split in SPLITS] + list(TEMPLATES)
    hashes = {name: digest(directory / name) for name in names}
    return validate_cohort(directory, dataset, fold, hashes, required_scope=scope, allow_missing=False)


def trajectory(report):
    values = [row["validation"]["reference_loss"] for row in report.get("history", [])]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Validation trajectory contains nonfinite measurements")
    trend = "insufficient_observations"
    if len(values) >= 3:
        trend = "decreased" if values[-1] < values[0] else "did_not_decrease"
    return {"validation_reference_loss": values, "observed_trend": trend,
            "boundary": "A small fixed validation probe is not evidence of cohort convergence or improved generalization"}


def last_json(log):
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[::-1]:
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "status" in value:
                return value
        except json.JSONDecodeError:
            pass
    return {}


def run_command(command, log, timeout):
    with Path(log).open("w", encoding="utf-8") as stream:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags,
                              start_new_session=os.name != "nt") as process:
            try:
                return process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    if os.name == "nt":
                        stopped = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                 timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        if stopped.returncode and process.poll() is None:
                            raise RuntimeError("Owned child process tree could not be terminated")
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait(timeout=10)
                except Exception as error:
                    if process.poll() is None:
                        process.kill()
                    raise RuntimeError("Owned child process tree cleanup failed; stop subsequent jobs") from error
                raise


def resource_failure(message):
    text = str(message).lower()
    return any(term in text for term in ("selected gpu is occupied", "less than 4500", "cuda out of memory",
                                          "cuda error", "no cuda", "no nvidia", "nvidia-smi", "outofmemoryerror", "child process tree",
                                          "out of memory", "not enough memory", "disk space", "no space left", "not enough space"))


def disk_check(output, checkpoint_budget_gib=8):
    if type(checkpoint_budget_gib) not in (int, float) or not math.isfinite(checkpoint_budget_gib) or checkpoint_budget_gib <= 0:
        raise ValueError("Declare a positive checkpoint budget in GiB")
    parent = Path(output).resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    required = int((10 + checkpoint_budget_gib) * 1024 ** 3)
    return {"ready": free >= required, "free_bytes": free, "required_bytes": required,
            "reserve_gib": 10, "checkpoint_budget_gib": checkpoint_budget_gib}


def validate_result(job, report_path, updates):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_status = {"complete", "budget_exhausted"}
    config = report.get("config", {})
    if (report.get("status") not in expected_status or type(report.get("steps")) is not int
            or report["steps"] != updates or report.get("method") != job["method"]
            or report.get("setting") != job["setting"] or report.get("seed") != job["seed"]
            or config.get("dataset") != job["dataset"] or config.get("fold") != job["fold"]
            or config.get("max_steps") != updates):
        raise ValueError("Child report does not prove the requested method, fold and exact bounded update budget")
    if report.get("audit", {}).get("data_hashes") != job["data_audit"]["data_hashes"]:
        raise ValueError("Child used data that differ from the audited acceptance export")
    path = report_path.parent / "updates.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(events) != updates or [event.get("step") for event in events] != list(range(1, updates + 1)):
        raise ValueError("Optimizer update trace does not match the declared step count")
    changed = set()
    for event in events:
        loss = event.get("total_loss")
        if type(loss) not in (float, int) or not math.isfinite(loss):
            raise ValueError("Update trace contains a missing or nonfinite objective")
        observed = {name for name, item in event.get("sampled_parameter_updates", {}).items() if item.get("changed") is True}
        if not observed:
            raise ValueError("Update trace lacks an observed parameter change")
        changed.update(observed)
    if not set(TARGETS.get(job["method"], ())).issubset(changed):
        raise ValueError("No observed parameter update for every replaced module")
    gradient_audit = report.get("gradient_verification", {})
    requested_protocol = job.get("gradient_protocol_version", 1)
    strict_boundary = False
    if job.get("verify_gradients"):
        if gradient_audit.get("enabled") is not True or gradient_audit.get("verified_steps") != updates:
            raise ValueError("Child did not verify gradients at every requested update")
        strict_boundary = gradient_audit.get("protocol_version", 1) >= 2
        if requested_protocol >= 2 and not strict_boundary:
            raise ValueError("Child lacks gradient protocol 2 classification-to-input verification")
        for event in events:
            verification = event.get("gradient_verification", {})
            if strict_boundary:
                if verification.get("protocol_version", 1) < 2:
                    raise ValueError("Gradient protocol differs between report and update trace")
                boundary = verification.get("classification_to_input", {})
                required = job["method"] not in {"original", "nnunet"}
                if boundary.get("required") is not required:
                    raise ValueError("Classification-to-input boundary requirement was not declared")
                if required and (boundary.get("finite") is not True or boundary.get("nonzero") is not True):
                    raise ValueError("Classification gradient does not reach the input image")
            input_gradient = verification.get("total_to_input", {})
            if (job["method"] != "original" and verification.get("input_required") is not True
                    or input_gradient.get("finite") is not True
                    or input_gradient.get("nonzero") is not True):
                raise ValueError("Input gradient was not finite and nonzero")
            for module in TARGETS.get(job["method"], ()):
                total = verification.get("total_to_candidates", {}).get(module, {})
                classification = verification.get("classification_to_candidates", {}).get(module, {})
                if total.get("finite") is not True or total.get("nonzero") is not True:
                    raise ValueError("Total objective does not train every candidate module")
                if job["method"] == "nnunet" and module == "seg_net":
                    if classification.get("expected_absent") is not True or classification.get("required") is not False:
                        raise ValueError("Auxiliary segmentation classification boundary was not declared")
                elif (classification.get("required") is not True or classification.get("finite") is not True
                      or classification.get("nonzero") is not True):
                    raise ValueError("Classification gradient does not reach a required candidate")
    checkpoint = report_path.parent / "last.pt"
    if not checkpoint.is_file():
        raise ValueError("Bounded run did not retain its checkpoint")
    checkpoint_mode = job.get("checkpoint_mode")
    if checkpoint_mode:
        policy = report.get("checkpoint_policy", {})
        if (config.get("checkpoint_mode") != checkpoint_mode or policy.get("mode") != checkpoint_mode
                or policy.get("resume_allowed") is not (checkpoint_mode == "resume")):
            raise ValueError("Checkpoint policy differs from the requested mode")
    coordination = report.get("coordination", {})
    mode = job.get("coordination", "none")
    mode = mode.get("mode", "none") if isinstance(mode, dict) else mode
    if config.get("coordination", "none") != mode or coordination.get("mode", "none") != mode:
        raise ValueError("Child coordination setting does not match the requested run")
    policy = {"mode": mode, "policy_learning_observed": False}
    if mode == "taskforce":
        for name in ("policy_updates", "actor_optimizer_steps", "critic_optimizer_steps"):
            count = coordination.get(name)
            if type(count) is not int or count < 0:
                raise ValueError("TaskForce report lacks actual optimizer update counts")
            policy[name] = count
        decisions = [event.get("gradient_policy") for event in events]
        if any(not isinstance(decision, dict) or not decision.get("weights") for decision in decisions):
            raise ValueError("TaskForce update trace lacks per-step weight decisions")
        for decision in decisions:
            values = decision["weights"].values() if isinstance(decision["weights"], dict) else decision["weights"]
            if not isinstance(values, (list, tuple, type({}.values()))) or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("TaskForce weight decisions must contain finite nonnegative numbers")
        for name in ("policy_updates", "actor_optimizer_steps", "critic_optimizer_steps"):
            counts = [decision.get(name) for decision in decisions]
            if (any(type(count) is not int or count < 0 for count in counts) or counts != sorted(counts)
                    or counts[-1] != policy[name]):
                raise ValueError("TaskForce update counts disagree with the per-step trace")
        policy["weight_decisions_observed"] = len(decisions)
        policy["policy_learning_observed"] = all(policy[name] > 0 for name in ("policy_updates", "actor_optimizer_steps", "critic_optimizer_steps"))
        policy["boundary"] = "Adaptive weight decisions are distinct from actor and critic optimization"
        if job.get("phase") == "trajectory" and updates >= 5 and not policy["policy_learning_observed"]:
            raise ValueError("Requested policy trajectory did not demonstrate actor and critic updates")
    return {"status": "passed", "worker_status": report["status"], "optimizer_updates": report["steps"],
            "trajectory": trajectory(report), "report_sha256": digest(report_path), "updates_sha256": digest(path),
            "checkpoint_sha256": digest(checkpoint), "changed_modules": sorted(changed), "coordination": policy,
            "gradient_verification": gradient_audit,
            "gradient_boundary_scope": "protocol_2" if strict_boundary else "legacy_boundary",
            "strict_current_gradient_verified": strict_boundary,
            "checkpoint_policy": report.get("checkpoint_policy", {"mode": "legacy_unspecified"})}


def prepare(config, config_path, output, phase):
    base = Path(config_path).resolve().parent
    resolve = lambda value: (base / Path(value)).resolve()
    updates = 2 if phase == "acceptance" else config.get("trajectory_updates", 8)
    if type(updates) is not int or not 1 <= updates <= 100:
        raise ValueError("Use a bounded integer probe of at most 100 updates")
    methods = config["methods"] if phase == "acceptance" else config.get("trajectory_methods", ["synthstrip", "voxelmorph"])
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError("Declare unique supported methods")
    fold, seed = config.get("fold", 1), config.get("seed", 17)
    if type(fold) is not int or fold not in range(1, 6) or type(seed) is not int:
        raise ValueError("Declare an original fold and integer seed")
    coordination = config.get("coordination", "none")
    if coordination not in {"none", "taskforce"}:
        raise ValueError("Coordination must be none or taskforce")
    checkpoint_mode = config.get("checkpoint_mode", "weights" if phase == "acceptance" else "resume")
    if checkpoint_mode not in {"weights", "resume"} or phase == "trajectory" and checkpoint_mode != "resume":
        raise ValueError("Checkpoint mode must be weights or resume; trajectories require resume")
    available = {f"{dataset}:{method}" for dataset in ("abide", "adhd") for method in methods}
    selected = config.get("job_subset", sorted(available))
    if (not isinstance(selected, list) or not selected or any(not isinstance(item, str) for item in selected)
            or len(selected) != len(set(selected)) or not set(selected).issubset(available)):
        raise ValueError("job_subset must contain unique existing dataset:method jobs and cannot be empty")
    python = config.get("python") or sys.executable
    if "/" in python or "\\" in python:
        python = str(resolve(python))
    if not Path(python).is_file() and not shutil.which(python):
        raise ValueError("Python interpreter was not found")
    source, output = resolve(config["source"]), Path(output).resolve()
    jobs = []
    for dataset in ("abide", "adhd"):
        if not any(f"{dataset}:{method}" in selected for method in methods):
            continue
        data = resolve(config["datasets"][dataset])
        if any(output == path or path in output.parents for path in (source, data)):
            raise ValueError("Output must be outside source and data directories")
        reasons, audit = [], None
        try:
            audit = subset_audit(data, dataset, fold)
        except (ValueError, KeyError, OSError) as error:
            reasons.append(str(error))
        for method in methods:
            if f"{dataset}:{method}" not in selected:
                continue
            job_output = output / f"{dataset}_{method}"
            setting = "original" if method == "original" else "joint"
            command = [python, str(ROOT / "scripts/train_replacement.py"), "--source", str(source), "--data", str(data),
                       "--output", str(job_output), "--method", method, "--setting", setting, "--epochs", str(updates),
                       "--max-steps", str(updates), "--dataset", dataset, "--fold", str(fold), "--seed", str(seed),
                       "--gpu", str(config.get("gpu", 0)), "--coordination", coordination, "--verify-gradients",
                       "--checkpoint-mode", checkpoint_mode]
            blockers = list(reasons)
            if method != "original":
                command.append("--connect-paths")
            if method == "synthstrip":
                if not config.get("adapter"):
                    blockers.append("SynthStrip requires an accepted adapter")
                else:
                    command.extend(("--adapter", str(resolve(config["adapter"]))))
            if method == "nnunet":
                plans = config.get("plans", {}).get(dataset)
                if not plans:
                    blockers.append("nnU Net requires actual plans for this training subset")
                else:
                    command.extend(("--plans", str(resolve(plans))))
            jobs.append({"dataset": dataset, "method": method, "setting": setting, "fold": fold, "seed": seed,
                         "command": command, "output": str(job_output), "data_audit": audit,
                         "coordination": coordination, "phase": phase, "verify_gradients": True,
                         "gradient_protocol_version": 2, "checkpoint_mode": checkpoint_mode,
                         "status": "blocked" if blockers else "planned", "reasons": blockers})
    return updates, jobs


def execute(jobs, output, record, updates, timeout=1800, preflight_timeout=120, runner=run_command):
    output = Path(output)
    initial = disk_check(output, record.get("checkpoint_budget_gib", 8))
    record["initial_disk_check"] = initial
    output.mkdir(parents=True, exist_ok=False)
    record.update(status="running", jobs=jobs)
    if not initial["ready"]:
        for job in jobs:
            if job["status"] == "planned":
                job.update(status="blocked_disk_space", disk=initial)
        record["status"] = "blocked_disk_space"
        save(output / "matrix.json", record)
        return record
    save(output / "matrix.json", record)
    halted = False
    for job in jobs:
        if halted:
            if job["status"] == "planned":
                job["status"] = "not_run_after_halt"
            continue
        if job["status"] == "blocked":
            continue
        job["disk"] = disk_check(output, record.get("checkpoint_budget_gib", 8))
        if not job["disk"]["ready"]:
            job.update(status="blocked_disk_space", error="Free space is below the checkpoint budget and 10 GiB reserve")
            halted = True
            save(output / "matrix.json", record)
            continue
        job_output = Path(job["output"])
        if job_output.exists():
            job.update(status="blocked_existing_output", error="Existing child output is preserved")
            halted = True
            continue
        prefix = job["dataset"] + "_" + job["method"]
        preflight_log, training_log = output / (prefix + ".preflight.log"), output / (prefix + ".log")
        started = time.monotonic()
        try:
            returncode = runner([*job["command"], "--preflight"], preflight_log, preflight_timeout)
            message = last_json(preflight_log)
            if returncode or message.get("status") != "ready":
                job.update(status="blocked", preflight_exit_code=returncode, error=message.get("error", "Child did not report ready"))
                halted = resource_failure(job["error"]) or returncode < 0
                continue
            if job_output.exists():
                halted = True
                raise ValueError("Static preflight unexpectedly created the child output")
            job["status"] = "running"
            save(output / "matrix.json", record)
            returncode = runner(job["command"], training_log, timeout)
            message = last_json(training_log)
            job["exit_code"] = returncode
            if returncode or message.get("status") not in {"complete", "budget_exhausted"}:
                error = message.get("error", "Child did not report successful bounded completion")
                shortage = any(term in str(error).lower() for term in ("disk space", "no space left", "not enough space"))
                job.update(status="blocked_disk_space" if shortage else "failed", error=error)
                halted = resource_failure(job["error"]) or returncode < 0
                continue
            report_path = job_output / "report.json"
            job.update(validate_result(job, report_path, updates))
            if job["worker_status"] != message["status"]:
                raise ValueError("CLI and report completion statuses disagree")
        except subprocess.TimeoutExpired:
            job.update(status="timeout", error="Child exceeded its bounded execution time")
            halted = True
        except KeyboardInterrupt:
            job.update(status="interrupted", error="Execution interrupted; remaining jobs were not started")
            halted = True
        except Exception as error:
            job.update(status="failed", error=str(error))
            halted = halted or resource_failure(error)
        finally:
            job["elapsed_seconds"] = time.monotonic() - started
            save(output / "matrix.json", record)
    record["status"] = "halted" if halted else "passed" if all(job["status"] == "passed" for job in jobs) else "incomplete"
    save(output / "matrix.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("acceptance", "trajectory"), default="acceptance")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--preflight-timeout", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; retain previous evidence and choose a new directory")
    if not 1 <= args.timeout <= 86400 or not 1 <= args.preflight_timeout <= 3600:
        parser.error("Timeouts must be positive and bounded")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    updates, jobs = prepare(config, args.config, args.output, args.phase)
    record = {"scope": args.phase, "config_sha256": digest(args.config), "optimizer_update_budget_per_job": updates,
              "selection_scope": "explicit_job_subset" if "job_subset" in config else "all_configured_methods_both_datasets",
              "selected_jobs": [f"{job['dataset']}:{job['method']}" for job in jobs], "denominator": len(jobs),
              "checkpoint_budget_gib": config.get("checkpoint_budget_gib", 8),
              "planned_checkpoint_budget_gib": config.get("checkpoint_budget_gib", 8) * len(jobs),
              "boundary": "Acceptance subsets never become full cohort results", "status": "planned", "jobs": jobs}
    if args.execute:
        record = execute(jobs, args.output, record, updates, args.timeout, args.preflight_timeout)
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record["status"] == "passed" or record["status"] == "planned" and all(job["status"] == "planned" for job in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
