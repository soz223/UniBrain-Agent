"""Evaluate a saved full pipeline with its recorded architecture and provenance."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

from .replacement_training import METHODS, SETTINGS, fingerprint, save_json


TEMPLATES = ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")
SPLITS = ("train", "val", "test")
IDENTITY_FIELDS = ("method", "setting", "initialization", "seed", "epochs", "pretrain_epochs", "trainable",
                   "cut_extraction", "loss_scale", "regularization", "dataset", "fold", "split_id", "weights_origin",
                   "connect_paths", "cut_boundary", "repair_strategy", "undo_repair", "patience", "coordination",
                   "coordination_batch_size", "probe_budget", "probe_modules", "verify_gradients", "checkpoint_mode")
IMPLEMENTATION_FILES = ("replacement_training.py", "full_model.py", "full_candidates.py", "full_connectivity.py",
                        "full_composition.py", "full_extended.py", "full_preprocessing.py", "full_anatomy_candidates.py",
                        "full_graph_candidates.py", "taskforce.py", "cohort_provenance.py", "candidate_integration.py",
                        "full_replacement.py", "sources.py", "site_training.py")


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    def finite(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid, parse_float=finite)


def same_value(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def file_binding(path, expected, name):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Missing recorded SHA256: " + name)
    if path is None or not Path(path).is_file() or fingerprint(path) != expected:
        raise ValueError("Missing or changed file: " + name)
    return {"path": str(Path(path).resolve()), "sha256": expected}


def implementation_hash(name):
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def architecture_files(identity):
    method = identity["method"]
    files = {"full_model.py", "full_replacement.py", "sources.py"}
    composition = identity["composition"].get("secondary_extraction") is not None
    if method in ("voxelmorph", "lapirn", "nnunet"):
        files.add("full_candidates.py")
    if method not in ("original", "synthstrip", "voxelmorph", "lapirn", "nnunet"):
        files.update(("full_extended.py", "full_connectivity.py"))
        if method in ("redwgan", "deepn4", "deepnormalize", "harmonizing_flows"):
            files.add("full_preprocessing.py")
        elif method in ("ernet", "fastsurfer", "nobrainer"):
            files.add("full_anatomy_candidates.py")
        else:
            files.add("full_graph_candidates.py")
    if identity["connect_paths"] or identity["setting"] == "original_completed":
        files.add("full_connectivity.py")
    if identity["adapter_hash"] is not None:
        files.add("full_replacement.py" if method == "synthstrip" else "candidate_integration.py")
    if composition:
        files.update(("full_composition.py", "full_replacement.py"))
    return sorted(files)


def code_audit(identity):
    recorded = identity.get("implementation_hashes")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Checkpoint lacks recorded implementation hashes")
    if any(name not in IMPLEMENTATION_FILES or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for name, value in recorded.items()):
        raise ValueError("Invalid recorded implementation file or checksum")
    required = architecture_files(identity)
    current = {name: implementation_hash(name) for name in recorded}
    for name in required:
        if recorded.get(name) != current.get(name):
            raise ValueError("Architecture implementation differs from training: " + name)
    return {"architecture_files_verified": required, "training_hashes": recorded, "current_hashes": current,
            "other_changed_files": sorted(name for name in recorded if recorded[name] != current[name]),
            "additional_runtime_hashes": {name: implementation_hash(name) for name in ("data.py", "upstream.py", "native.py")},
            "evaluator_sha256": implementation_hash(Path(__file__).name),
            "scope": "Required recorded architecture files match; helper hashes are reported separately. Historical checkpoints did not fingerprint data.py, upstream.py or native.py"}


def load_payload(path, sha256):
    import torch
    file_binding(path, sha256, "checkpoint")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        value = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(value, dict) or value.get("checkpoint_kind") not in {"resume", "evaluation", "evaluation_weights_only"}
            or not isinstance(value.get("identity"), dict) or not isinstance(value.get("progress"), dict)
            or not isinstance(value.get("model"), dict) or not value["model"]):
        raise ValueError("A complete replacement-training checkpoint with identity and model state is required")
    steps = value["progress"].get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("Checkpoint does not contain a trained pipeline")
    for name, tensor in value["model"].items():
        if not isinstance(name, str) or not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Checkpoint contains an invalid or nonfinite tensor")
    file_binding(path, sha256, "checkpoint")
    return value


def recorded_arguments(args, payload, report, report_path):
    identity, declared = payload["identity"], report.get("config", {})
    if report.get("status") not in {"complete", "budget_exhausted"} or not isinstance(declared, dict):
        raise ValueError("A completed training report is required")
    for name in IDENTITY_FIELDS:
        if name not in identity or name not in declared or not same_value(identity[name], declared[name]):
            raise ValueError("Checkpoint and training configuration disagree: " + name)
    for name in ("method", "setting", "seed"):
        if not same_value(report.get(name), identity[name]):
            raise ValueError("Training report identity disagrees: " + name)
    if identity["method"] not in METHODS or identity["setting"] not in SETTINGS:
        raise ValueError("Unsupported checkpoint method or setting")
    if type(report.get("steps")) is not int or payload["progress"]["steps"] > report["steps"]:
        raise ValueError("Checkpoint progress is ahead of its training report")
    selection = identity.get("selection_metric")
    if selection is not None and selection != declared.get("selection_metric"):
        raise ValueError("Checkpoint selection metric differs from the report")
    if args.checkpoint.name == "best.pt":
        chosen = report.get("selection", {})
        progress = payload["progress"]
        if (progress.get("best_epoch") != chosen.get("best_epoch") or progress.get("best_value") != chosen.get("best_value")
                or progress.get("best_epoch") is None or progress.get("epoch") != progress.get("best_epoch")
                or progress.get("position") != 0):
            raise ValueError("Best checkpoint disagrees with validation selection")
    for name in ("dataset", "fold", "split_id"):
        expected = getattr(args, name)
        if expected is not None and not same_value(expected, identity[name]):
            raise ValueError("Requested evaluation identity differs: " + name)
    if not isinstance(identity.get("composition"), dict) or identity["composition"].get("mode") != declared.get("composition_mode"):
        raise ValueError("Checkpoint composition identity is missing or inconsistent")
    if identity["composition"]["mode"] not in {"simultaneous", "sequential"}:
        raise ValueError("Unknown recorded composition mode")
    for name in ("connect_paths", "cut_extraction", "verify_gradients"):
        if type(identity[name]) is not bool:
            raise ValueError("Recorded switch must be boolean: " + name)
    if identity["initialization"] not in {"random", "official", "training_fold"}:
        raise ValueError("Unknown recorded initialization")
    if (type(identity["seed"]) is not int or identity["cut_boundary"] not in {"none", "extraction", "registration", "parcellation", "all"}
            or identity["repair_strategy"] not in {"soft", "straight_through", "gumbel_softmax"}
            or identity["undo_repair"] not in {"none", "nearest", "graph", "mask"}):
        raise ValueError("Recorded forward configuration is invalid")
    restored = SimpleNamespace(**declared)
    restored.source, restored.data = args.source.resolve(), args.data.resolve()
    bindings = {}
    for name, expected in (("adapter", identity.get("adapter_hash")), ("plans", identity.get("plans_hash")),
                           ("secondary_extraction", identity["composition"].get("secondary_extraction"))):
        override = getattr(args, name)
        original = declared.get(name)
        if expected is None:
            if original is not None or override is not None:
                raise ValueError("Unexpected reconstruction input: " + name)
            setattr(restored, name, None)
            continue
        path = Path(override or original) if override or original else None
        if path is not None and not path.is_absolute():
            path = report_path.parent / path
        bindings[name] = file_binding(path, expected, name)
        setattr(restored, name, path.resolve())
    if identity["method"] == "synthstrip" and restored.adapter is None:
        raise ValueError("SynthStrip checkpoint requires its original adapter")
    if identity["method"] == "nnunet" and restored.plans is None:
        raise ValueError("nnU-Net checkpoint requires its original dataset plan")
    if restored.secondary_extraction and (identity["method"] not in {"voxelmorph", "lapirn"} or identity["setting"] != "joint"):
        raise ValueError("Unsupported composed checkpoint")
    if identity["initialization"] != "random":
        if not identity["weights_origin"] or not identity.get("weights_hash"):
            raise ValueError("Checkpoint lost its initialization provenance")
        if identity["initialization"] == "training_fold" and (declared.get("weights_dataset") != identity["dataset"] or declared.get("weights_fold") != identity["fold"]):
            raise ValueError("Initialization provenance belongs to a different fold")
    if identity.get("host_hash"):
        evidence = report.get("audit", {}).get("host_training_evidence") or {}
        if evidence.get("checkpoint_sha256") != identity["host_hash"]:
            raise ValueError("Original host provenance does not match the checkpoint identity")
    return restored, bindings


def audit_sources(restored, identity, report):
    from .native import verify_original
    from .sources import verify
    from .full_extended import REPOSITORIES
    source = verify_original(restored.source)
    if source != identity.get("source") or source != report.get("audit", {}).get("source"):
        raise ValueError("Original source differs from checkpoint provenance")
    repositories = {"synthstrip": "freesurfer_synthstrip", "voxelmorph": "reg_voxelmorph_master", "lapirn": "reg_lapirn", "nnunet": "dynamic-network-architectures", **REPOSITORIES}
    names = [repositories[restored.method]] if restored.method in repositories else []
    if restored.secondary_extraction:
        names.append("freesurfer_synthstrip")
    candidates = verify(sorted(set(names))) if names else []
    if candidates != report.get("audit", {}).get("candidate_sources"):
        raise ValueError("Candidate source differs from the training report")
    return {"original": source, "candidates": candidates}


def audit_data(restored, identity, report):
    import numpy as np
    from .data import OriginalDataset
    from .cohort_provenance import validate_cohort, validate_site
    names = [f"adhd_{split}_sample.npy" for split in SPLITS] + list(TEMPLATES)
    expected = identity.get("data_hashes")
    if not isinstance(expected, dict) or set(expected) != set(names) or expected != report.get("audit", {}).get("data_hashes"):
        raise ValueError("Checkpoint and report data identities disagree")
    for name in names:
        file_binding(restored.data / name, expected[name], name)
    datasets = {split: OriginalDataset(restored.data / f"adhd_{split}_sample.npy") for split in SPLITS}
    ids = {split: set(str(value) for value in dataset.array["id"]) for split, dataset in datasets.items()}
    if any(len(ids[s]) != len(datasets[s]) for s in SPLITS) or any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Evaluation subjects are duplicated or overlap splits")
    for name in TEMPLATES:
        array = np.load(restored.data / name, mmap_mode="r", allow_pickle=False)
        if array.shape != (96, 96, 96) or not np.isfinite(array).all():
            raise ValueError("Invalid evaluation template: " + name)
    cohort = (validate_site(restored.data, restored.dataset, restored.split_id, expected) if restored.split_id else
              validate_cohort(restored.data, restored.dataset, restored.fold, expected, allow_missing=False))
    if restored.plans:
        from .nnunet_prepare import validate_plan_provenance
        provenance = validate_plan_provenance(restored.plans, expected["adhd_train_sample.npy"], restored.dataset, restored.fold)
        if provenance["sidecar_sha256"] != identity.get("plans_provenance_hash"):
            raise ValueError("Dataset plan provenance changed")
    return {"hashes": expected, "cohort": cohort, "split_sizes": {name: len(value) for name, value in datasets.items()}}


def prepare(args):
    if args.limit is not None and args.limit < 1 or args.threads < 1:
        raise ValueError("Evaluation limits and thread counts must be positive")
    if args.output.exists():
        raise ValueError("Use a new evaluation output directory")
    if any(args.output.resolve() == path.resolve() or path.resolve() in args.output.resolve().parents for path in (args.source, args.data)):
        raise ValueError("Evaluation output must be separate from source and data")
    report_path = (args.training_report or args.checkpoint.parent / "report.json").resolve()
    report = read_json(report_path)
    report_sha256 = fingerprint(report_path)
    if args.training_report_sha256 and args.training_report_sha256 != report_sha256:
        raise ValueError("Training report checksum differs from the requested artifact")
    payload = load_payload(args.checkpoint, args.checkpoint_sha256)
    restored, bindings = recorded_arguments(args, payload, report, report_path)
    code = code_audit(payload["identity"])
    sources = audit_sources(restored, payload["identity"], report)
    data = audit_data(restored, payload["identity"], report)
    if fingerprint(report_path) != report_sha256:
        raise ValueError("Training report changed during inspection")
    progress = {key: value if not isinstance(value, float) or math.isfinite(value) else None
                for key, value in payload["progress"].items()}
    return payload, restored, {"checkpoint": file_binding(args.checkpoint, args.checkpoint_sha256, "checkpoint"),
                               "training_report": {"path": str(report_path), "sha256": report_sha256},
                               "identity": payload["identity"], "checkpoint_progress": progress,
                               "nonfinite_progress_fields": [key for key, value in payload["progress"].items()
                                                             if isinstance(value, float) and not math.isfinite(value)],
                               "bindings": bindings, "implementation": code, "sources": sources, "data": data,
                               "architecture": report.get("architecture", {})}


def reconstruct(restored, payload, audit):
    from .full_model import build_full_model
    from .full_candidates import build_candidate_model
    from .full_extended import TARGETS, build_extended_model
    from .full_connectivity import connect_model
    from .candidate_integration import bind_adapter, bind_extended_adapters
    if restored.secondary_extraction:
        from .full_composition import build_composed_model
        # Final state contains both stages; prior initialization files are unnecessary.
        model, _, _ = build_composed_model(restored.source, restored.method, restored.secondary_extraction, mode="simultaneous")
        audit["reconstruction"] = {"recorded_composition_mode": restored.composition_mode,
                                    "structural_constructor_mode": "simultaneous",
                                    "reason": "Both modes have identical topology; all trained parameters come from this final checkpoint"}
    elif restored.method in ("original", "synthstrip"):
        model, _, _ = build_full_model(restored.source, restored.adapter if restored.method == "synthstrip" else None)
    elif restored.method in TARGETS:
        model, _, _ = build_extended_model(restored.source, restored.method)
    else:
        model, _, _ = build_candidate_model(restored.source, restored.method, plans=restored.plans)
    if restored.connect_paths or restored.setting == "original_completed":
        model = connect_model(model, cut=restored.cut_boundary, repair=restored.repair_strategy, undo=restored.undo_repair)
    if restored.adapter is not None and restored.method not in ("original", "synthstrip"):
        (bind_extended_adapters if restored.method in TARGETS else bind_adapter)(model, restored.method, restored.adapter)
    state = model.state_dict()
    if set(state) != set(payload["model"]):
        raise ValueError("Checkpoint tensor names differ from the reconstructed pipeline")
    for name, value in state.items():
        other = payload["model"][name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Checkpoint tensor shape or dtype differs: " + name)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != audit["architecture"].get("parameters"):
        raise ValueError("Reconstructed parameter count differs from the training report")
    model.load_state_dict(payload["model"], strict=True)
    from .replacement_training import configure_boundaries
    model._evaluation_boundaries = configure_boundaries(model, restored.method, restored.setting, restored.cut_extraction)
    model.requires_grad_(False)
    model.eval()
    return model


def evaluate_checkpoint(args):
    payload, restored, audit = prepare(args)
    result = {"status": "ready", "scope": "Saved full pipeline evaluation; no optimizer or training updates",
              "optimizer_updates": 0, "method": restored.method, "setting": restored.setting, "split": args.split,
              "dataset": restored.dataset, "fold": restored.fold, "split_id": restored.split_id,
              "limit": args.limit, "provenance": audit, "historical_initializers_reloaded": False}
    if args.preflight:
        return result
    from .full_replacement_worker import available_gpu, selected_device
    device = selected_device(args.gpu)
    result["gpu"] = available_gpu(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import numpy as np
    import torch
    from .data import OriginalDataset
    from .replacement_training import evaluate
    torch.set_num_threads(args.threads)
    torch.manual_seed(restored.seed)
    np.random.seed(restored.seed)
    random.seed(restored.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        model = reconstruct(restored, payload, audit)
        del payload
        model.cuda()
        templates = [torch.from_numpy(np.load(restored.data / name, allow_pickle=False)).float().cuda().reshape(1, 1, 96, 96, 96) for name in TEMPLATES]
        dataset = OriginalDataset(restored.data / f"adhd_{args.split}_sample.npy")
        result["evaluation"] = evaluate(model, dataset, templates, args.limit, args.output / "predictions" if args.save_predictions else None)
        for name, binding in audit["bindings"].items():
            file_binding(binding["path"], binding["sha256"], name)
        file_binding(args.checkpoint, args.checkpoint_sha256, "checkpoint")
        file_binding(audit["training_report"]["path"], audit["training_report"]["sha256"], "training report")
        result.update(status="complete", strict_state_loaded=True,
                      full_split_evaluated=result["evaluation"]["sample_count"] == len(dataset))
        return result
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save_json(args.output / "report.json", result)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source", "data", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--training-report-sha256")
    for name in ("training-report", "adapter", "plans", "secondary-extraction"):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--dataset")
    result.add_argument("--fold", type=int)
    result.add_argument("--split-id")
    result.add_argument("--split", choices=SPLITS, default="test")
    result.add_argument("--limit", type=int)
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--save-predictions", action="store_true")
    result.add_argument("--preflight", action="store_true")
    return result
