"""Run bounded cohort checks with strict data and child-process accounting."""

import argparse
import json
import math
import re
import shutil
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
    components = {}
    for key in ("extraction_loss", "registration_loss", "segmentation_loss", "classification_loss"):
        series = [row["validation"].get(key) for row in report.get("history", [])]
        if series and all(value is not None for value in series):
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in series):
                raise ValueError("Validation component contains nonfinite measurements")
            components[key] = series
    return {"validation_reference_loss": values, "validation_components": components, "observed_trend": trend,
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
    from unibrain_agent.owned_process import run_logged
    return run_logged(command, log, timeout, cwd=ROOT)


def resource_failure(message):
    text = str(message).lower()
    return any(term in text for term in ("selected gpu is occupied", "less than 4500", "cuda out of memory",
                                          "cuda error", "no cuda", "no nvidia", "nvidia-smi", "outofmemoryerror", "process tree",
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


def require_finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON measurements are not valid evidence")
    if isinstance(value, dict):
        for item in value.values():
            require_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite_json(item)


def evidence_json(text):
    value = json.loads(text)
    require_finite_json(value)
    return value


def validate_gradient_statistics(value, complete=True):
    fields = ("tensor_count", "observed_tensors", "nonzero_elements", "l1")
    if not complete and not any(key in value for key in fields):
        return
    count, observed, elements = (value.get(key) for key in fields[:3])
    norm = value.get("l1")
    if (any(type(number) is not int or number < 0 for number in (count, observed, elements))
            or observed > count or type(norm) not in (int, float) or not math.isfinite(norm) or norm < 0
            or value.get("finite") is not True or type(value.get("nonzero")) is not bool
            or value["nonzero"] != (elements > 0) or (norm > 0) != (elements > 0)
            or observed == 0 and (elements != 0 or norm != 0)):
        raise ValueError("Gradient tensor counts, nonzero elements and finite norm are inconsistent")


def validate_record_statistics(record, complete=True):
    for key in ("classification_to_input", "total_to_input"):
        if record.get(key) is not None:
            validate_gradient_statistics(record[key], complete)
    for key in ("classification_to_candidates", "total_to_candidates"):
        for value in record.get(key, {}).values():
            validate_gradient_statistics(value, complete)


def validate_encoder_source(proof):
    source = proof.get("encoder_source", {})
    manifest = evidence_json((ROOT / "catalog/sources.json").read_text(encoding="utf-8"))
    repository = next((item for item in manifest["repositories"] if item["directory"] in {"reg_unibrain", "reg_ernet"}
                       and item["repository"] == source.get("repository") and item["commit"] == source.get("commit")), None)
    checksum = next((item["sha256"] for item in repository["files"] if item["path"] == "model.py"), None) if repository else None
    if (not checksum or source.get("class_name") != "encoder" or source.get("source_file") != "model.py"
            or source.get("repository") != repository["repository"] or source.get("commit") != repository["commit"]
            or source.get("source_sha256") != checksum or source.get("head_object_verified") is not True
            or any(not isinstance(source.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", source[key])
                   for key in ("forward_code_sha256", "constructor_code_sha256"))):
        raise ValueError("Warmup encoder source is not the corresponding pinned original implementation")


def validate_warmup(events, audit):
    require_finite_json(events)
    require_finite_json(audit)
    first = None
    for event in events:
        step = event["step"]
        verification = event.get("gradient_verification", {})
        validate_record_statistics(verification)
        state = verification.get("warmup", {})
        status = state.get("status")
        if status == "not_needed":
            if first is not None or any(state.get(key) is not None for key in ("module", "first_step", "confirmed_step", "reason", "initialization")):
                raise ValueError("Warmup history cannot discard a pending boundary")
            continue
        if status not in {"pending", "confirmed"} or state.get("module") != "ext_net" or state.get("first_step") != 1:
            raise ValueError("Warmup must identify the first extraction boundary explicitly")
        proof = state.get("initialization") or {}
        validate_encoder_source(proof)
        head = proof.get("head_total_gradient", {})
        weight = proof.get("head_weight_gradient", {})
        validate_gradient_statistics(head)
        validate_gradient_statistics(weight)
        tensors = proof.get("extraction_tensor_count")
        names = proof.get("extraction_parameter_names")
        if (state.get("reason") != "zero_initialized_affine_head"
                or proof.get("head_path") not in {"reg_net.affine.fc_loc.2", "reg_net.core.affine.fc_loc.2"}
                or proof.get("weight_shape") != [12, 128] or proof.get("bias_shape") != [12]
                or any(proof.get(key) is not True for key in ("zero_weight", "identity_bias", "head_trainable", "weight_nonzero_after_update"))
                or head["tensor_count"] != 2 or head["observed_tensors"] != 2 or not head["nonzero"] or head["nonzero_elements"] > 1548
                or weight["tensor_count"] != 1 or weight["observed_tensors"] != 1 or not weight["nonzero"] or weight["nonzero_elements"] > 1536
                or not 0 <= head["nonzero_elements"] - weight["nonzero_elements"] <= 12 or head["l1"] < weight["l1"]
                or type(tensors) is not int or tensors < 1
                or not isinstance(names, list) or len(names) != tensors
                or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names)):
            raise ValueError("Warmup lacks verified original affine initialization and its actual optimizer update")
        classification = verification.get("classification_to_candidates", {}).get("ext_net", {})
        if (classification.get("tensor_count") != tensors or classification.get("observed_tensors") != tensors
                or classification.get("parameter_names") != names):
            raise ValueError("Every extraction parameter must remain observed through warmup confirmation")
        if status == "pending":
            total = verification.get("total_to_candidates", {}).get("ext_net", {})
            if (first is not None or step != 1 or state.get("confirmed_step") is not None
                    or type(tensors) is not int or tensors < 1 or classification.get("observed_tensors") != tensors
                    or classification.get("required") is not True or classification.get("finite") is not True
                    or classification.get("nonzero") is not False or classification.get("nonzero_elements") != 0
                    or classification.get("l1") != 0 or total.get("finite") is not True or total.get("nonzero") is not True
                    or event.get("sampled_parameter_updates", {}).get("ext_net", {}).get("changed") is not True):
                raise ValueError("Only a finite, connected zero extraction gradient can use the first real update as warmup")
            first = state
        elif (first is None or step < 2 or state.get("confirmed_step") != 2
              or proof != first["initialization"] or classification.get("required") is not True
              or classification.get("finite") is not True or classification.get("nonzero") is not True):
            raise ValueError("Warmup requires a measured nonzero extraction gradient after the first optimizer update")
    final = events[-1].get("gradient_verification", {}).get("warmup", {})
    if final.get("status") == "pending" or audit.get("warmup") != final:
        raise ValueError("A pending or inconsistent warmup cannot complete acceptance")
    if "last_step" in audit and audit["last_step"] != events[-1].get("gradient_verification"):
        raise ValueError("Reported final gradient measurements differ from the optimizer trace")
    if first is not None and audit.get("diagnosis_reaches_all_candidates") is not True:
        raise ValueError("Confirmed warmup must retain the measured final classification connectivity")
    return {"used": first is not None, "optimizer_updates": 1 if first else 0, "state": final}


def validate_result(job, report_path, updates):
    report = evidence_json(report_path.read_text(encoding="utf-8"))
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
    events = [evidence_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    actual_protocol = None
    warmup = {"used": False, "optimizer_updates": 0, "state": None}
    if job.get("verify_gradients"):
        if gradient_audit.get("enabled") is not True or gradient_audit.get("verified_steps") != updates:
            raise ValueError("Child did not verify gradients at every requested update")
        actual_protocol = gradient_audit.get("protocol_version", 1)
        if type(actual_protocol) is not int or actual_protocol not in (1, 2, 3):
            raise ValueError("Unknown gradient verification protocol")
        strict_boundary = actual_protocol >= 2
        if actual_protocol < requested_protocol:
            raise ValueError(f"Child lacks gradient protocol {requested_protocol} verification")
        if actual_protocol == 3:
            warmup = validate_warmup(events, gradient_audit)
        for event in events:
            verification = event.get("gradient_verification", {})
            validate_record_statistics(verification, complete=actual_protocol >= 3)
            if strict_boundary:
                if verification.get("protocol_version", 1) != actual_protocol:
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
                      or (classification.get("nonzero") is not True
                          and not (actual_protocol == 3 and warmup["used"] and event["step"] == 1 and module == "ext_net"))):
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
            "gradient_boundary_scope": f"protocol_{actual_protocol}" if strict_boundary else "legacy_boundary",
            "gradient_protocol_version": actual_protocol, "gradient_warmup": warmup,
            "initial_nonzero_gradient_verified": strict_boundary and not warmup["used"],
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
                         "gradient_protocol_version": 3, "checkpoint_mode": checkpoint_mode,
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
