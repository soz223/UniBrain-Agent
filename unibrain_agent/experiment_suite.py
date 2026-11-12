"""Plan, execute and account for the paper's experimental matrix."""

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .owned_process import run_logged

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("redwgan", "deepn4", "synthstrip", "ernet", "deepnormalize", "harmonizing_flows",
           "voxelmorph", "lapirn", "nnunet", "fastsurfer", "nobrainer", "fbnetgen", "brainnetcnn", "bnt")
TRAIN_METHODS = ("synthstrip", "voxelmorph", "lapirn", "nnunet")
AGENT_VARIANTS = ("codex", "claude", "mle_star", "codex_checks", "claude_checks", "full",
                  "no_applicability", "no_knowledge", "no_spatial", "no_gradient", "no_fidelity", "no_revision",
                  "llm_b", "llm_c")
SYSTEMS = ("human_unibrain", "openclaw_unibrain", "neuropilot", "nexus", "neura", "neura_unibrain",
           "mle_star", "codex", "codex_unibrain", "claude", "claude_unibrain", "ours")
ABLATIONS = ("cut_extraction", "cut_registration", "cut_parcellation", "cut_all", "undo_nearest",
             "undo_graph", "undo_mask", "straight_through", "gumbel_softmax", "new_only", "host_only",
             "random_initialization", "loss_0.1", "loss_10", "simultaneous", "sequential")
BENCHMARK_VARIANTS = {"codex": "general", "claude": "general", "mle_star": "general",
                      "codex_checks": "checks", "claude_checks": "checks", "full": "workflow",
                      "no_applicability": "without_screening", "no_knowledge": "without_knowledge",
                      "no_spatial": "without_spatial", "no_gradient": "without_gradient",
                      "no_fidelity": "without_fidelity", "no_revision": "without_revision",
                      "llm_b": "workflow", "llm_c": "workflow"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def example_config():
    return {"version": 1, "python": "", "source": "", "output_root": "./runs/paper",
            "datasets": {name: {"folds": {str(f): "" for f in range(1, 6)}, "sites": {}} for name in ("abide", "adhd")},
            "seeds": [17, 29, 43], "repetitions": 3, "epochs": 20, "long_epochs": 1000, "pretrain_epochs": 20,
            "selection_metric": "classification_loss",
            "gpu": 0, "adapters": {}, "weights": {}, "hosts": {}, "plans": {}, "sequential_runs": {},
            "integration_configs": {}, "case_study_run": "", "site_initialization": "random",
            "research_cycle": {"task": "", "provider": "codex", "model": "", "stage": "",
                               "acceptance_steps": 1, "max_calls": 4, "coordination": "none"}, "executors": {}}


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Unsupported experiment configuration")
    if not config.get("seeds") or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("Explicit integer random seeds are required")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate seeds would double count runs")
    if config.get("selection_metric", "classification_loss") not in {"classification_loss", "reference_loss"}:
        raise ValueError("Unsupported validation checkpoint selection metric")
    for key in ("epochs", "long_epochs", "repetitions"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Positive integer required: " + key)
    config["_base"] = str(path.parent)
    config["_hash"] = digest({k: v for k, v in config.items() if not k.startswith("_")})
    return config


def resolve(config, value):
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_base"]) / path).resolve()


def plan(config):
    jobs = []

    def add(group, method, variant, dataset="abide", fold=1, seed=17, repetition=1, **extra):
        spec = dict(group=group, method=method, variant=variant, dataset=dataset, fold=fold,
                    seed=seed, repetition=repetition, **extra)
        identifier = "_".join(str(spec[k]) for k in ("group", "dataset", "fold", "method", "variant", "seed", "repetition"))
        safe_identifier = re.sub(r"[^A-Za-z0-9_.]", "_", identifier)
        if safe_identifier != identifier:
            identifier = safe_identifier + "_" + digest(spec)[:10]
        jobs.append({"id": identifier, **spec, "job_hash": digest(spec)})

    add("checker_fixtures", "tensor_contract", "known_faults", seed=config["seeds"][0])
    for method in METHODS:
        for variant in AGENT_VARIANTS:
            for rep in range(1, config["repetitions"] + 1):
                add("integration", method, variant, repetition=rep, seed=config["seeds"][(rep - 1) % len(config["seeds"])])
    for dataset in ("abide", "adhd"):
        for fold in range(1, 6):
            for seed in config["seeds"]:
                for variant in ("original", "stagewise", "completed_paths"):
                    add("training", "original", variant, dataset, fold, seed, epochs=config["epochs"])
                for method in TRAIN_METHODS:
                    for variant in ("independent_official", "independent_finetuned", "joint"):
                        add("training", method, variant, dataset, fold, seed, epochs=config["epochs"])
                for system in SYSTEMS:
                    add("systems", "pipeline", system, dataset, fold, seed, epochs=config["epochs"])
        for site in sorted(config["datasets"].get(dataset, {}).get("sites", {})) if dataset == "abide" else ():
            for seed in config["seeds"]:
                for variant in ("original", "independent_finetuned", "joint"):
                    add("site", "synthstrip" if variant != "original" else "original", variant, dataset, site, seed,
                        epochs=config["epochs"])
    if not config["datasets"].get("abide", {}).get("sites"):
        add("site", "synthstrip", "site_split_required", fold="unconfigured")
    for fold in range(1, 6):
        for seed in config["seeds"]:
            for variant in ABLATIONS:
                add("ablation", "voxelmorph", variant, fold=fold, seed=seed, epochs=config["epochs"])
    for variant in ("original", "independent_finetuned", "joint"):
        add("long_budget", "original" if variant == "original" else "voxelmorph", variant,
            seed=config["seeds"][0], epochs=config["long_epochs"])
    for variant in ("simultaneous", "sequential"):
        for rep in range(1, config["repetitions"] + 1):
            add("composition", "synthstrip_voxelmorph", variant, repetition=rep,
                seed=config["seeds"][(rep - 1) % len(config["seeds"])], epochs=config["epochs"])
    add("case_study", "observed_failure", "repair_trace", seed=config["seeds"][0])
    return jobs


def data_path(config, job):
    dataset = config["datasets"].get(job["dataset"], {})
    collection = dataset.get("sites" if job["group"] == "site" else "folds", {})
    return resolve(config, collection.get(str(job["fold"])))


def full_fold_audit(config, job, data_hashes=None):
    from .cohort_provenance import validate_cohort
    return validate_cohort(data_path(config, job), job["dataset"], job["fold"], data_hashes,
                           required_scope="full_fold", allow_missing=False)


def site_audit(config, job, data_hashes=None):
    from .cohort_provenance import validate_site
    return validate_site(data_path(config, job), job["dataset"], str(job["fold"]), data_hashes)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cycle_options(config):
    options = config.get("research_cycle", {})
    if not isinstance(options.get("task"), str) or not options["task"].strip():
        raise ValueError("The native research cycle requires an explicit task")
    from .research_cycle import parser
    arguments = ["--source", "source", "--data", "data", "--output", "output", "--task", options["task"],
                 "--provider", options.get("provider", "codex")]
    if options.get("stage"):
        arguments += ["--stage", options["stage"]]
    try:
        parser().parse_args(arguments)
    except SystemExit as error:
        raise ValueError("Unsupported research cycle provider or stage") from error
    if type(options.get("acceptance_steps", 1)) is not int or options.get("acceptance_steps", 1) not in (1, 2):
        raise ValueError("Research cycle acceptance uses one or two updates")
    if type(options.get("max_calls", 4)) is not int or not 2 <= options.get("max_calls", 4) <= 20:
        raise ValueError("Research cycle integration call budget must be between 2 and 20")
    if options.get("coordination", "none") not in {"none", "taskforce"}:
        raise ValueError("Unsupported research cycle coordination")
    if "training_timeout_seconds" in options and (type(options["training_timeout_seconds"]) is not int or options["training_timeout_seconds"] < 1):
        raise ValueError("Research cycle training timeout must be a positive integer")
    if "checkpoint_mode" in options and options["checkpoint_mode"] not in {"weights", "resume"}:
        raise ValueError("Research cycle checkpoint mode must be weights or resume")
    return options


def cycle_result(config, job, folder, outcome, smoke_steps=None):
    from types import SimpleNamespace
    from .integration_benchmark import verdict
    from .research_cycle import validate_full_training, validate_training

    options = cycle_options(config)
    method = outcome.get("method")
    if (outcome.get("status") != "complete" or outcome.get("phase") != "complete"
            or method not in METHODS or outcome.get("seed") != job["seed"]):
        raise ValueError("Research cycle did not complete the planned selection and training")
    if (outcome.get("provider", {}).get("provider") != options.get("provider", "codex")
            or options.get("model") and outcome.get("provider", {}).get("model") != options["model"]
            or outcome.get("task") != options["task"]):
        raise ValueError("Research cycle provider or task differs from the plan")
    accepted = outcome.get("accepted_adapter", {})
    if accepted.get("file") != "accepted_adapter.py":
        raise ValueError("Research cycle did not retain its accepted adapter")
    adapter = folder / "accepted_adapter.py"
    accepted_hash = file_hash(adapter)
    if accepted_hash != accepted.get("sha256") or outcome.get("accepted_adapter_unchanged_after_training") is not True:
        raise ValueError("Accepted research cycle code changed")
    trial = outcome.get("integration", {})
    if (trial.get("status") != "passed" or trial.get("final_valid") is not True
            or not any(call.get("adapter_sha256") == accepted_hash and verdict(call.get("evaluation", {})) == "passed"
                       for call in trial.get("calls", []))):
        raise ValueError("The accepted adapter lacks a matching successful validation")
    training_path = folder / "training/report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training != outcome.get("training"):
        raise ValueError("Research cycle and retained training report disagree")
    declared = training.get("config", {})
    if declared.get("dataset") != job["dataset"] or declared.get("fold") != job["fold"]:
        raise ValueError("Research cycle trained a different dataset or fold")
    audit = full_fold_audit(config, job, training.get("audit", {}).get("data_hashes", {}))
    budget = smoke_steps or audit["split_counts"]["train"] * job["epochs"]
    updates = outcome.get("actual_updates", {})
    if (type(updates.get("acceptance")) is not int or not 1 <= updates["acceptance"] <= options.get("acceptance_steps", 1)
            or updates.get("training") != training.get("steps")):
        raise ValueError("Research cycle update accounting does not match its retained evidence")
    validate_training(training, method, adapter, job["seed"], budget, resolve(config, config["source"]),
                      data_path(config, job), audit["data_hashes"]["adhd_train_sample.npy"],
                      options.get("coordination", "none"), options.get("checkpoint_mode", "weights" if smoke_steps else "resume"))
    protocol = outcome.get("training_protocol", {})
    if protocol.get("full_training", False) is not (smoke_steps is None):
        raise ValueError("Bounded and complete research cycles cannot share a result category")
    artifacts = [adapter, training_path, folder / "training/updates.jsonl", folder / "training/last.pt"]
    if not smoke_steps:
        if protocol.get("evaluate_test") is not True or outcome.get("budgets", {}).get("training_epochs") != job["epochs"]:
            raise ValueError("Full research cycle did not retain the requested epoch and test protocol")
        validate_full_training(training, SimpleNamespace(epochs=job["epochs"], dataset=job["dataset"], fold=job["fold"],
                                                         seed=job["seed"], data=data_path(config, job), output=folder,
                                                         selection_metric=config.get("selection_metric", "classification_loss"),
                                                         evaluate_test=True), audit["split_counts"], audit)
        best = folder / "training/best.pt"
        if file_hash(best) != training["test"]["checkpoint_sha256"]:
            raise ValueError("Research cycle test checkpoint changed")
        artifacts.append(best)
        outcome["test"] = training["test"]
    elif declared.get("max_steps") != smoke_steps:
        raise ValueError("Bounded research cycle changed the requested step limit")
    return {"cohort_provenance": audit, "selected_method": method,
            "artifact_hashes": {path.relative_to(folder.parent).as_posix(): file_hash(path) for path in artifacts},
            "research_cycle_updates": updates}


def executor_for(config, job):
    executors = config.get("executors", {})
    for key in (job["id"], f"{job['group']}:{job['method']}:{job['variant']}", f"{job['group']}:{job['variant']}"):
        if key in executors:
            return executors[key]
    return None


def initial_weights(config, job):
    value = config.get("weights", {}).get(job["method"], {})
    if not isinstance(value, dict):
        raise ValueError("Weights need explicit path and provenance, not an unlabeled checkpoint")
    if "folds" in value:
        value = value["folds"].get(f"{job['dataset']}:{job['fold']}", {})
    if value.get("provenance") not in {"official", "training_fold"}:
        raise ValueError("Weights require official or training_fold provenance")
    if not isinstance(value.get("origin"), str) or not value["origin"] or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")):
        raise ValueError("Weights require a source description and SHA256")
    if value["provenance"] == "training_fold" and (value.get("dataset") != job["dataset"] or value.get("fold") != job["fold"]):
        raise ValueError("Pretrained weights must come from this dataset's training fold")
    path = resolve(config, value.get("path"))
    if path is None or not path.is_file():
        raise ValueError("Matched initial weights are not available")
    return path, value


def host_weights(config, job):
    value = config.get("hosts", {}).get(f"{job['dataset']}:{job['fold']}", {})
    path = resolve(config, value.get("path"))
    if (path is None or not path.is_file() or not value.get("origin")
            or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
        raise ValueError("A shared trained host checkpoint with provenance and SHA256 is required for registration comparisons")
    return path, value


def nnunet_plan(config, job, verify=False, required=True):
    value = config.get("plans", {}).get("nnunet")
    if isinstance(value, dict):
        folds = value.get("folds")
        if not isinstance(folds, dict):
            raise ValueError("nnU-Net plans need a path or a folds mapping")
        value = folds.get(f"{job['dataset']}:{job['fold']}")
    if value is not None and not isinstance(value, str):
        raise ValueError("Each nnU-Net fold plan must be a path")
    path = resolve(config, value)
    if path is None:
        if not required:
            return None
        raise ValueError("A real nnU-Net plan is required for this dataset and fold")
    if verify:
        from .nnunet_prepare import validate_plan_provenance
        if not path.is_file():
            raise ValueError("A real nnU-Net plans file is required")
        cohort = full_fold_audit(config, job)
        training_hash = cohort["declared_data_hashes"]["adhd_train_sample.npy"]
        validate_plan_provenance(path, training_hash, job["dataset"], job["fold"])
    return path


def integration_config(config, job):
    path = resolve(config, config.get("integration_configs", {}).get(f"{job['method']}:{job['variant']}"))
    if path is None or not path.is_file():
        raise ValueError("Missing benchmark case configuration")
    value = json.loads(path.read_text(encoding="utf-8"))
    cases = value.get("cases", [])
    if len(cases) != 1 or cases[0].get("id") != job["method"] or len(value.get("providers", [])) != 1:
        raise ValueError("Each suite integration config must specify exactly its one case and one provider")
    provider = value["providers"][0].get("provider")
    expected = {"codex": "codex", "codex_checks": "codex", "claude": "claude", "claude_checks": "claude", "mle_star": "mle_star"}.get(job["variant"])
    if expected and provider != expected:
        raise ValueError("Named agent baseline does not match the configured provider")
    for field in ("source_files", "required_files"):
        cases[0][field] = [str((path.parent / p).resolve()) for p in cases[0].get(field, [])]
    evaluator = cases[0].get("evaluator", {})
    evaluator["cwd"] = str((path.parent / evaluator.get("cwd", ".")).resolve())
    command = evaluator.get("command", [])
    evaluator["command"] = [str((path.parent / arg).resolve()) if (path.parent / arg).exists() else arg for arg in command]
    value.update(variants=[BENCHMARK_VARIANTS[job["variant"]]], seeds=[job["seed"]], repetitions=1)
    return value


def import_integration_index(config_path, index_path, variants, apply=False):
    config_path, index_path = Path(config_path).resolve(), Path(index_path).resolve()
    config = load_config(config_path)
    if not variants or any(variant not in AGENT_VARIANTS for variant in variants) or len(set(variants)) != len(variants):
        raise ValueError("Choose unique explicit suite integration variants")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mapping = {}
    for method, filename in index.get("configs", {}).items():
        if method not in METHODS:
            raise ValueError("Generated index contains an unknown candidate")
        path = (index_path.parent / filename).resolve()
        try:
            relative = os.path.relpath(path, config_path.parent)
        except ValueError:
            relative = str(path)
        for variant in variants:
            key = method + ":" + variant
            previous = config.get("integration_configs", {}).get(key)
            if previous and resolve(config, previous) != path:
                raise ValueError("Import would overwrite an existing benchmark mapping: " + key)
            candidate = {**config, "integration_configs": {**config.get("integration_configs", {}), key: relative}}
            integration_config(candidate, {"method": method, "variant": variant, "seed": config["seeds"][0]})
            mapping[key] = relative
    if not mapping:
        raise ValueError("Generated index has no benchmark configurations")
    if apply:
        config.setdefault("integration_configs", {}).update(mapping)
        save(config_path, {key: value for key, value in config.items() if not key.startswith("_")})
    return {"applied": apply, "mappings": mapping, "methods": len(index["configs"]),
            "scope": "Configuration linking only; no provider call, GPU check or training performed"}


def is_composition(job):
    return job["group"] == "composition" or job["group"] == "ablation" and job["variant"] in {"simultaneous", "sequential"}


def sequential_run(config, job):
    value = config.get("sequential_runs", {}).get(f"{job['dataset']}:{job['fold']}:{job['seed']}", {})
    checkpoint, report = resolve(config, value.get("checkpoint")), resolve(config, value.get("report"))
    if (checkpoint is None or report is None or not checkpoint.is_file() or not report.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
        raise ValueError("Sequential composition requires an actual first-stage checkpoint, report and SHA256")
    return checkpoint, report, value["sha256"]


def preflight(config, job):
    reasons = []
    external = executor_for(config, job)
    if external is not None:
        if not isinstance(external.get("argv"), list) or not external["argv"] or not all(isinstance(x, str) for x in external["argv"]):
            reasons.append("Executor argv must be a nonempty string list")
        elif "{" not in external["argv"][0] and not Path(external["argv"][0]).is_file() and not shutil.which(external["argv"][0]):
            reasons.append("External executor executable was not found")
        if not external.get("description"):
            reasons.append("Executor must document its real method and budget")
        for dependency in external.get("requires", []):
            if not resolve(config, dependency).exists():
                reasons.append("Missing executor dependency: " + dependency)
        kind = "external"
    elif job["group"] == "checker_fixtures":
        kind = "checker_fixtures"
        if not (ROOT / "scripts/benchmark_integration.py").exists():
            reasons.append("Integration checker executable is not installed")
    elif job["group"] == "case_study":
        kind = "case_study"
        trace = resolve(config, config.get("case_study_run"))
        if trace is None or not (trace / "report.json").is_file():
            reasons.append("A retained full replacement run with real attempt reports is required")
        if not (ROOT / "scripts/summarize_case_study.py").is_file():
            reasons.append("Case study evidence exporter is not installed")
    elif job["group"] == "integration":
        key = f"{job['method']}:{job['variant']}"
        path = resolve(config, config.get("integration_configs", {}).get(key))
        kind = "integration"
        if path is None or not path.is_file():
            reasons.append("Missing integration benchmark config: " + key)
        else:
            try:
                benchmark = integration_config(config, job)
                from .integration_benchmark import case_blockers
                reasons.extend(case_blockers(benchmark["cases"][0]))
                if type(benchmark["cases"][0].get("expected_fit")) is not bool:
                    reasons.append("Integration case requires an adjudicated applicability label")
            except (ValueError, KeyError, TypeError) as error:
                reasons.append(str(error))
    elif job["group"] == "systems" and job["variant"] == "ours":
        kind = "research_cycle"
        source, data = resolve(config, config.get("source")), data_path(config, job)
        if source is None or not (source / "model.py").is_file():
            reasons.append("Pinned original source is not configured")
        try:
            cycle_options(config)
            if data is None:
                raise ValueError("A preserved full fold is required for the native research cycle")
            full_fold_audit(config, job)
            if config.get("plans", {}).get("nnunet"):
                nnunet_plan(config, job, verify=True, required=cycle_options(config).get("stage") == "segmentation")
        except (ValueError, KeyError, OSError, TypeError) as error:
            reasons.append(str(error))
        if not (ROOT / "scripts/run_research_cycle.py").is_file():
            reasons.append("Native research cycle executable is not installed")
    elif is_composition(job):
        kind = "composition"
        adapter = resolve(config, config.get("adapters", {}).get("synthstrip"))
        if adapter is None or not adapter.is_file():
            reasons.append("Full composition needs an accepted SynthStrip adapter")
        registration = {**job, "group": "training", "method": "voxelmorph", "variant": "joint"}
        training = preflight(config, registration)
        reasons.extend(reason for reason in training["reasons"] if not reason.startswith("A shared trained host") or job["variant"] != "sequential")
        if job["variant"] == "sequential":
            try:
                sequential_run(config, job)
            except ValueError as error:
                reasons.append(str(error))
        if not (ROOT / "scripts/run_composition.py").is_file():
            reasons.append("Full composition executable is not installed")
    elif job["group"] in {"training", "site", "long_budget", "ablation"}:
        kind = "site_training" if job["group"] == "site" else "training"
        if job["group"] == "site":
            if (job["method"], job["variant"]) not in {("original", "original"), ("synthstrip", "joint"), ("synthstrip", "independent_finetuned")}:
                reasons.append("Site protocol requires prepared data and an original or SynthStrip comparison")
            if config.get("site_initialization", "random") not in {"random", "official"}:
                reasons.append("Site initialization must be random or official, shared across replacement comparisons")
        supported = {"original", "stagewise", "completed_paths", "independent_official", "independent_finetuned", "joint",
                     "cut_extraction", "cut_registration", "cut_parcellation", "cut_all", "undo_nearest",
                     "undo_graph", "undo_mask", "straight_through", "gumbel_softmax",
                     "new_only", "host_only", "random_initialization", "loss_0.1", "loss_10"}
        if job["variant"] not in supported:
            reasons.append("This experimental intervention has no registered full-model executor")
        if job["method"] not in ("original", *TRAIN_METHODS):
            reasons.append("No full-model implementation registered for this candidate")
        source, data = resolve(config, config.get("source")), data_path(config, job)
        if source is None or not (source / "model.py").is_file():
            reasons.append("Pinned original source is not configured")
        if data is None or any(not (data / f"adhd_{split}_sample.npy").is_file() for split in ("train", "val", "test")):
            reasons.append("Prepared, preserved train/val/test fold is not configured")
        else:
            try:
                (site_audit if job["group"] == "site" else full_fold_audit)(config, job)
            except (ValueError, KeyError, OSError, TypeError) as error:
                reasons.append(("Site provenance: " if job["group"] == "site" else "Full fold provenance: ") + str(error))
        if job["method"] in TRAIN_METHODS:
            adapter = resolve(config, config.get("adapters", {}).get(job["method"]))
            if adapter is None or not adapter.is_file():
                reasons.append("Accepted generated adapter is not configured: " + job["method"])
        if job["method"] == "nnunet":
            try:
                nnunet_plan(config, job, verify=True)
            except (ValueError, KeyError, OSError, TypeError) as error:
                reasons.append("nnU-Net plan: " + str(error))
        needs_weights = job["group"] != "site" or config.get("site_initialization", "random") == "official"
        if job["method"] != "original" and job["variant"] != "random_initialization" and needs_weights:
            try:
                _, metadata = initial_weights(config, job)
                if job["group"] == "site" and metadata["provenance"] != "official":
                    raise ValueError("Site initialization cannot use a numeric-fold checkpoint")
            except ValueError as error:
                reasons.append(str(error))
        if job["method"] in {"voxelmorph", "lapirn"} or job["variant"] == "stagewise" or job["method"] == "synthstrip" and job["variant"] == "new_only":
            try:
                host_weights(config, job)
            except ValueError as error:
                reasons.append(str(error))
        if not (ROOT / "scripts/train_replacement.py").exists():
            reasons.append("Full training executable is not installed")
    else:
        kind = "unregistered"
        reasons.append("Provide and verify an executor for this system or experiment")
    return {"job_id": job["id"], "executor": kind, "ready": not reasons, "reasons": reasons,
            "scope": "Static prerequisites only; runtime success and scientific validity are not established"}


def command_for(config, job, folder, smoke_steps=None):
    python = str(resolve(config, config["python"])) if config.get("python") else sys.executable
    source, data = resolve(config, config.get("source")), data_path(config, job)
    external = executor_for(config, job)
    if external:
        substitutions = {"python": python, "job": str(folder / "job.json"), "result": str(folder / "result.json"),
                         "output": str(folder / "payload"), "source": str(source or ""), "data": str(data or ""),
                         "seed": str(job["seed"]), "gpu": str(config.get("gpu", 0)), "repo": str(ROOT),
                         "max_steps": str(smoke_steps or "")}
        return [arg.format_map(substitutions) for arg in external["smoke_argv" if smoke_steps else "argv"]]
    if job["group"] == "checker_fixtures":
        return [python, str(ROOT / "scripts/benchmark_integration.py"), "faults", "--output", str(folder / "payload")]
    if job["group"] == "case_study":
        return [python, str(ROOT / "scripts/summarize_case_study.py"), "--run",
                str(resolve(config, config["case_study_run"])), "--output", str(folder / "payload")]
    if job["group"] == "integration":
        path = folder / "integration_config.json"
        benchmark = integration_config(config, job)
        benchmark["cases"][0]["max_optimizer_updates"] = smoke_steps or 2
        save(path, benchmark)
        return [python, str(ROOT / "scripts/benchmark_integration.py"), "run", "--config", str(path), "--output", str(folder / "payload")]
    if job["group"] == "systems" and job["variant"] == "ours":
        options = cycle_options(config)
        command = [python, str(ROOT / "scripts/run_research_cycle.py"), "--source", str(source), "--data", str(data),
                   "--output", str(folder / "payload"), "--task", options["task"], "--provider", options.get("provider", "codex"),
                   "--dataset", job["dataset"], "--fold", str(job["fold"]), "--seed", str(job["seed"]),
                   "--epochs", str(job["epochs"]), "--gpu", str(config.get("gpu", 0)),
                   "--selection-metric", config.get("selection_metric", "classification_loss"),
                   "--acceptance-steps", str(options.get("acceptance_steps", 1)), "--max-calls", str(options.get("max_calls", 4)),
                   "--coordination", options.get("coordination", "none"),
                   "--checkpoint-mode", options.get("checkpoint_mode", "weights" if smoke_steps else "resume")]
        for field in ("model", "stage"):
            if options.get(field):
                command += ["--" + field, options[field]]
        if "training_timeout_seconds" in options:
            command += ["--training-timeout-seconds", str(options["training_timeout_seconds"])]
        if config.get("plans", {}).get("nnunet"):
            plans = nnunet_plan(config, job, required=options.get("stage") == "segmentation")
            if plans is not None:
                command += ["--plans", str(plans)]
        command += ["--train-steps", str(smoke_steps), "--eval-limit", "2"] if smoke_steps else ["--full-training", "--evaluate-test"]
        return command
    if is_composition(job):
        registration = {**job, "group": "training", "method": "voxelmorph", "variant": "joint"}
        temporary = dict(config)
        if job["variant"] == "sequential":
            temporary["hosts"] = dict(config.get("hosts", {}))
            # The prior extraction checkpoint supplies all retained host parameters.
            checkpoint, report, expected = sequential_run(config, job)
            temporary["hosts"][f"{job['dataset']}:{job['fold']}"] = {"path": str(checkpoint), "sha256": expected, "origin": "Previous extraction run"}
        command = command_for(temporary, registration, folder, smoke_steps)
        command[1] = str(ROOT / "scripts/run_composition.py")
        command += ["--secondary-extraction", str(resolve(config, config["adapters"]["synthstrip"])),
                    "--composition-mode", job["variant"]]
        if job["variant"] == "sequential":
            for flag in ("--host-checkpoint", "--host-sha256", "--host-origin", "--host-dataset", "--host-fold"):
                index = command.index(flag)
                del command[index:index + 2]
            command += ["--previous-checkpoint", str(checkpoint), "--previous-report", str(report), "--previous-sha256", expected]
        return command
    variant = job["variant"]
    setting = ("original_frozen_stages" if variant == "stagewise" else "original_completed" if variant == "completed_paths" else variant
               if variant in {"original", "joint", "independent_official", "independent_finetuned"} else "joint")
    script = "run_site_experiment.py" if job["group"] == "site" else "train_replacement.py"
    command = [python, str(ROOT / "scripts" / script), "--source", str(source), "--data", str(data),
               "--output", str(folder / "payload"), "--method", job["method"], "--setting", setting,
               "--epochs", str(job["epochs"]), "--seed", str(job["seed"]), "--gpu", str(config.get("gpu", 0)),
               "--selection-metric", config.get("selection_metric", "classification_loss"),
               "--dataset", job["dataset"], "--split-id" if job["group"] == "site" else "--fold", str(job["fold"])]
    if setting == "independent_finetuned":
        command += ["--pretrain-epochs", str(config.get("pretrain_epochs", config["epochs"]))]
    if job["method"] in TRAIN_METHODS:
        command += ["--adapter", str(resolve(config, config["adapters"][job["method"]]))]
    if job["method"] == "nnunet":
        command += ["--plans", str(nnunet_plan(config, job))]
    needs_weights = job["group"] != "site" or config.get("site_initialization", "random") == "official"
    if job["method"] != "original" and variant != "random_initialization" and needs_weights:
        weights, metadata = initial_weights(config, job)
        provenance = metadata["provenance"]
        command += ["--initialization", provenance, "--weights", str(weights), "--weights-origin", metadata["origin"],
                    "--weights-sha256", metadata["sha256"]]
        if provenance == "training_fold":
            command += ["--weights-dataset", metadata["dataset"], "--weights-fold", str(metadata["fold"])]
            if setting == "independent_official":
                command[command.index("--setting") + 1] = "independent_pretrained"
    if setting not in {"original", "original_frozen_stages"}:
        command += ["--connect-paths"]
    if variant.startswith("cut_"):
        command += ["--cut-boundary", variant.removeprefix("cut_")]
    if variant.startswith("undo_"):
        command += ["--undo-repair", variant.removeprefix("undo_")]
    if variant in {"straight_through", "gumbel_softmax"}:
        command += ["--repair-strategy", variant]
    if job["method"] in {"voxelmorph", "lapirn"} or variant == "stagewise" or job["method"] == "synthstrip" and variant == "new_only":
        host, metadata = host_weights(config, job)
        command += ["--host-checkpoint", str(host), "--host-sha256", metadata["sha256"], "--host-origin", metadata["origin"],
                    "--host-dataset", job["dataset"], "--host-fold", str(job["fold"])]
    if variant in {"new_only", "host_only"}:
        command += ["--trainable", "new" if variant == "new_only" else "host"]
    if variant.startswith("loss_"):
        command += ["--loss-scale", variant.split("_", 1)[1]]
    if smoke_steps:
        command += ["--max-steps", str(smoke_steps), "--eval-limit", "2"]
    else:
        command += ["--evaluate-test"]
    return command


def run_job(config, job, smoke_steps=None, timeout=86400):
    readiness = preflight(config, job)
    if not readiness["ready"]:
        return {"status": "blocked", **readiness}
    if smoke_steps not in (None, 1, 2):
        raise ValueError("Smoke mode allows only one or two optimizer updates")
    external = executor_for(config, job)
    if smoke_steps and external is not None and not external.get("smoke_argv"):
        return {"status": "blocked", "reasons": ["External executor has no explicit bounded smoke command"]}
    folder = resolve(config, config["output_root"]) / job["id"]
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "job.json", {**job, "config_hash": config["_hash"], "smoke_steps": smoke_steps})
    record = {"status": "running", "job": job, "config_hash": config["_hash"], "smoke_steps": smoke_steps,
              "command": None, "started_unix": time.time()}
    save(folder / "execution.json", record)
    try:
        if external is None and readiness["executor"] in {"training", "composition", "site_training", "research_cycle"}:
            audit = site_audit if job["group"] == "site" else full_fold_audit
            record["cohort_provenance"] = audit(config, job)
        command = command_for(config, job, folder, smoke_steps)
        if external is None and "--adapter" in command:
            index = command.index("--adapter") + 1
            adapter = folder / "input_adapter.py"
            adapter.write_bytes(Path(command[index]).read_bytes())
            command[index] = str(adapter)
            record["accepted_adapter_sha256"] = file_hash(adapter)
            record["artifact_hashes"] = {adapter.name: record["accepted_adapter_sha256"]}
        record["command"] = command
        save(folder / "execution.json", record)
        returncode = run_logged(command, folder / "execution.log", timeout, ROOT)
        record["returncode"] = returncode
        if returncode:
            cycle_path = folder / "payload/report.json"
            if external is None and readiness["executor"] == "research_cycle" and cycle_path.is_file():
                incomplete = json.loads(cycle_path.read_text(encoding="utf-8"))
                if incomplete.get("status") in {"blocked", "rejected"}:
                    record.update(status=incomplete["status"], outcome=incomplete, raw_report="payload/report.json",
                                  raw_report_sha256=file_hash(cycle_path))
                    return record
            raise RuntimeError("Executor failed; inspect retained execution.log")
        external_result = folder / "result.json"
        if external_result.exists():
            outcome = json.loads(external_result.read_text(encoding="utf-8"))
            if outcome.get("job_hash") != job["job_hash"]:
                raise ValueError("Result does not match the planned experiment")
            if smoke_steps and (type(outcome.get("optimizer_updates")) is not int or not 0 <= outcome["optimizer_updates"] <= smoke_steps):
                raise ValueError("External result does not prove the smoke update limit")
            record["raw_report"] = "result.json"
            record["raw_report_sha256"] = hashlib.sha256(external_result.read_bytes()).hexdigest()
        else:
            reports = [folder / "payload" / name for name in ("report.json", "summary.json", "faults.json")]
            path = next((path for path in reports if path.exists()), None)
            if path is None:
                raise ValueError("Executor produced no recognized result report")
            outcome = json.loads(path.read_text(encoding="utf-8"))
            record["raw_report"] = path.relative_to(folder).as_posix()
            record["raw_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if job["group"] == "checker_fixtures":
                outcome["status"] = "passed" if outcome.get("passed") is True else "failed"
        record["outcome"] = outcome
        if external is None and readiness["executor"] in {"training", "composition", "site_training"}:
            declared = outcome.get("config", {})
            expected_method = command[command.index("--method") + 1]
            expected_setting = command[command.index("--setting") + 1]
            if (outcome.get("method") != expected_method or outcome.get("setting") != expected_setting
                    or outcome.get("seed") != job["seed"] or declared.get("dataset") != job["dataset"]
                    or declared.get("fold") != (None if job["group"] == "site" else job["fold"])
                    or job["group"] == "site" and declared.get("split_id") != str(job["fold"])):
                raise ValueError("Training report does not match the planned method, setting, seed and fold")
            if smoke_steps and (type(outcome.get("steps")) is not int or not 1 <= outcome["steps"] <= smoke_steps):
                raise ValueError("Training report does not prove the requested bounded updates")
            cohort = audit(config, job, outcome.get("audit", {}).get("data_hashes", {}))
            if cohort["manifest_sha256"] != record["cohort_provenance"]["manifest_sha256"]:
                raise ValueError("Cohort manifest changed while the experiment was running")
            record["cohort_provenance"] = cohort
            if "accepted_adapter_sha256" in record:
                architecture = outcome.get("architecture", {})
                if (file_hash(adapter) != record["accepted_adapter_sha256"]
                        or architecture.get("generated_adapter_active_in_training") is not True
                        or architecture.get("generated_adapter_sha256") != record["accepted_adapter_sha256"]):
                    raise ValueError("Training did not execute the unchanged accepted adapter")
        if external is None and readiness["executor"] == "research_cycle":
            evidence = cycle_result(config, job, folder / "payload", outcome, smoke_steps)
            if evidence["cohort_provenance"]["manifest_sha256"] != record["cohort_provenance"]["manifest_sha256"]:
                raise ValueError("Cohort manifest changed during the research cycle")
            record.update(evidence)
        if job["group"] == "integration":
            runs = outcome.get("runs", [])
            if len(runs) != 1 or runs[0].get("seed") != job["seed"] or runs[0].get("case_id") != job["method"]:
                raise ValueError("Integration result must contain the exact planned case and seed")
            if runs[0].get("status") == "blocked":
                record["status"] = "blocked"
                return record
            outcome["metrics"] = {name: int(runs[0][name]) for name in ("first_attempt_valid", "final_valid", "silent_failure")
                                  if type(runs[0].get(name)) is bool}
        status = outcome.get("status")
        if status not in {"complete", "completed", "passed", "budget_exhausted"}:
            raise ValueError("Executor did not report a recognized completion status")
        record["status"] = "smoke_complete" if smoke_steps or status == "budget_exhausted" else "complete"
    except subprocess.TimeoutExpired:
        record.update(status="timeout", error="Owned execution tree exceeded the timeout and was terminated")
    except KeyboardInterrupt:
        record.update(status="interrupted", error="Owned execution tree was interrupted and terminated")
        raise
    except Exception as error:
        record.update(status="failed", error=str(error))
    finally:
        record["elapsed_seconds"] = time.time() - record["started_unix"]
        save(folder / "execution.json", record)
    return record


def summarize(config):
    root = resolve(config, config["output_root"])
    rows, grouped = [], defaultdict(list)
    for job in plan(config):
        path = root / job["id"] / "execution.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        status = record.get("status", "not_run")
        readiness = preflight(config, job) if not record else None
        if readiness and not readiness["ready"]:
            status = "blocked"
        if record and (record.get("config_hash") != config["_hash"] or record.get("job", {}).get("job_hash") != job["job_hash"]):
            status = "stale_config"
        if status == "complete" and record.get("raw_report"):
            raw = (path.parent / record["raw_report"].replace("\\", "/")).resolve()
            if path.parent.resolve() not in raw.parents or not raw.is_file() or file_hash(raw) != record.get("raw_report_sha256"):
                status = "changed_report"
        if status == "complete":
            for relative, expected in record.get("artifact_hashes", {}).items():
                artifact = (path.parent / relative.replace("\\", "/")).resolve()
                if path.parent.resolve() not in artifact.parents or not artifact.is_file() or file_hash(artifact) != expected:
                    status = "changed_artifact"
                    break
        row = {k: job[k] for k in ("id", "group", "dataset", "method", "variant", "fold", "seed", "repetition")}
        row["status"] = status
        row["blocked_reasons"] = "; ".join(readiness["reasons"]) if readiness else ""
        rows.append(row)
        if status != "complete":
            continue
        outcome = record.get("outcome", {})
        metrics = (outcome.get("test") or {}).get("metrics", outcome.get("metrics", {}))
        for metric, value in metrics.items():
            if type(value) in (int, float) and math.isfinite(value):
                grouped[(job["group"], job["dataset"], job["method"], job["variant"], metric)].append((job["fold"], value))
    statistics_rows = []
    for key, observations in grouped.items():
        folds = defaultdict(list)
        for fold, value in observations:
            folds[fold].append(value)
        by_fold = key[0] in {"training", "ablation", "systems", "site", "long_budget", "composition"}
        values = [statistics.mean(values) for values in folds.values()] if by_fold else [value for _, value in observations]
        statistics_rows.append({"group": key[0], "dataset": key[1], "method": key[2], "variant": key[3], "metric": key[4],
                                "n_runs": len(observations), "n_units": len(values),
                                "unit": "site_mean" if key[0] == "site" else "fold_mean" if by_fold else "attempt",
                                "mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
    return {"config_hash": config["_hash"], "planned": len(rows), "complete": sum(row["status"] == "complete" for row in rows),
            "rows": rows, "metrics": statistics_rows,
            "scope": "Descriptive statistics only. Missing, failed, smoke and stale runs are excluded and listed explicitly."}


def write_summary(config, output):
    report = summarize(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "summary.json", report)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    return report
