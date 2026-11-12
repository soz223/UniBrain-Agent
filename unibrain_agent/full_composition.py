"""Compose full extraction and dense registration with audited state transfer."""

import hashlib
import json
from pathlib import Path

import torch


def fingerprint(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_previous(checkpoint, report, expected_sha256, dataset=None, fold=None):
    if not checkpoint or not report or not expected_sha256:
        raise ValueError("Sequential composition requires a checkpoint, its report and expected SHA256")
    actual = fingerprint(checkpoint)
    if actual != expected_sha256:
        raise ValueError("Previous checkpoint SHA256 mismatch")
    evidence = json.loads(Path(report).read_text(encoding="utf-8"))
    if evidence.get("status") not in {"passed", "complete", "budget_exhausted"}:
        raise ValueError("Previous extraction run did not complete its declared budget")
    steps = evidence.get("steps", len(evidence.get("updates", [])))
    if type(steps) is not int or steps < 1:
        raise ValueError("Sequential composition requires an actually trained first stage")
    if evidence.get("method") not in {None, "synthstrip"}:
        raise ValueError("Supported order is trained SynthStrip extraction followed by dense registration")
    if evidence.get("method") is None:
        if evidence.get("checkpoint", {}).get("sha256") != actual or not evidence.get("gradient_check", {}).get("mask_requires_grad"):
            raise ValueError("Unidentified first-stage report lacks verified extraction training evidence")
    declared = evidence.get("config", {})
    if dataset is not None and (declared.get("dataset") != dataset or declared.get("fold") != fold):
        raise ValueError("Sequential checkpoint must use the same dataset and training fold")
    return {"checkpoint_sha256": actual, "report_sha256": fingerprint(report), "prior_optimizer_steps": steps,
            "dataset": declared.get("dataset"), "fold": declared.get("fold"),
            "order": ["extraction", "registration"]}


def composition_preflight(args):
    if args.method not in {"voxelmorph", "lapirn"} or args.setting != "joint" or args.trainable != "all":
        raise ValueError("Two-stage composition currently requires dense registration, joint setting and all trainable modules")
    if not args.secondary_extraction.is_file():
        raise ValueError("Generated extraction adapter is missing")
    from .full_replacement import check_code
    check_code(args.secondary_extraction.read_text(encoding="utf-8"))
    if args.composition_mode == "sequential":
        if args.host_checkpoint or args.extraction_weights:
            raise ValueError("Sequential mode must preserve the previous host and extraction weights")
        return verify_previous(args.previous_checkpoint, args.previous_report, args.previous_sha256, args.dataset, args.fold)
    if any((args.previous_checkpoint, args.previous_report, args.previous_sha256)):
        raise ValueError("Previous run inputs require sequential composition")
    if args.extraction_weights:
        expected = getattr(args, "extraction_weights_sha256", None)
        origin = getattr(args, "extraction_weights_origin", None)
        if not expected or not origin or fingerprint(args.extraction_weights) != expected:
            raise ValueError("Extraction weights need matching SHA256 and explicit provenance")
    return {"mode": "simultaneous", "scope": "Source and file checks only"}


def migrate_state(model, previous, renamed_prefixes, new_prefixes):
    target = model.state_dict()
    transferred, renamed = {}, {}
    for name, value in previous.items():
        mapped = name
        for old, new in renamed_prefixes.items():
            if name.startswith(old):
                mapped = new + name[len(old):]
                renamed[name] = mapped
                break
        if mapped not in target or mapped in transferred:
            raise ValueError("Unexpected or duplicate transferred tensor: " + mapped)
        if value.shape != target[mapped].shape or value.dtype != target[mapped].dtype:
            raise ValueError("Transferred tensor shape or dtype changed: " + mapped)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor: " + name)
        transferred[mapped] = value
    missing = sorted(set(target) - set(transferred))
    illegal = [name for name in missing if not any(name.startswith(prefix) for prefix in new_prefixes)]
    if illegal:
        raise ValueError("Missing retained host tensors: " + ", ".join(illegal[:8]))
    merged = {**target, **transferred}
    model.load_state_dict(merged, strict=True)
    return {"retained_tensors": len(transferred), "renamed_tensors": renamed,
            "new_tensors": missing, "allowed_new_prefixes": list(new_prefixes), "strict_load": True}


def build_composed_model(source, registration, adapter, mode="simultaneous", registration_weights=None,
                         extraction_weights=None, previous_checkpoint=None, previous_report=None,
                         previous_sha256=None, host_checkpoint=None, dataset=None, fold=None):
    from .full_candidates import DenseHost, FullRegistration
    from .full_model import build_full_model
    from .sources import verify

    if registration not in {"voxelmorph", "lapirn"} or mode not in {"simultaneous", "sequential"}:
        raise ValueError("Composition supports SynthStrip and VoxelMorph or LapIRN")
    if mode == "simultaneous" and any((previous_checkpoint, previous_report, previous_sha256)):
        raise ValueError("Use sequential mode to transfer a previous extraction run")
    if mode == "sequential" and (host_checkpoint or extraction_weights):
        raise ValueError("Sequential mode preserves the previous trained host and extraction weights")
    previous_evidence = verify_previous(previous_checkpoint, previous_report, previous_sha256, dataset, fold) if mode == "sequential" else None
    provenance = verify(["reg_voxelmorph_master" if registration == "voxelmorph" else "reg_lapirn"])
    host, config, architecture = build_full_model(source, adapter, host_checkpoint=host_checkpoint)
    if extraction_weights:
        state = torch.load(extraction_weights, map_location="cpu", weights_only=True)
        host.ext_net.core.load_state_dict(state.get("model_state_dict", state), strict=True)
    if previous_checkpoint:
        bundle = torch.load(previous_checkpoint, map_location="cpu", weights_only=True)
        identity = bundle.get("identity", {})
        if identity and (identity.get("method") != "synthstrip" or (dataset is not None and
                         (identity.get("dataset") != dataset or identity.get("fold") != fold))):
            raise ValueError("Checkpoint identity does not match the declared first-stage run")
        prior_report = json.loads(Path(previous_report).read_text(encoding="utf-8"))
        expected_adapter = identity.get("adapter_hash") or prior_report.get("adapter_sha256")
        if expected_adapter != fingerprint(adapter):
            raise ValueError("Sequential composition must preserve the previously trained extraction adapter code")
        previous_evidence["adapter_sha256"] = expected_adapter
        previous = bundle.get("model", bundle)
        host.load_state_dict(previous, strict=True)
    else:
        previous = host.state_dict()
    original_modules = {name: module for name, module in host.named_children() if name != "reg_net"}
    host.reg_net = FullRegistration(registration, host.reg_net)
    model = DenseHost(host)
    model.candidate_modules = ("ext_net", "reg_net")
    if registration_weights:
        state = torch.load(registration_weights, map_location="cpu", weights_only=True)
        model.reg_net.core.load_state_dict(state.get("model_state_dict", state), strict=True)
    transfer = migrate_state(model, previous, {"reg_net.": "reg_net.affine."}, ("reg_net.core.",))
    if any(getattr(model, name) is not module for name, module in original_modules.items()):
        raise ValueError("Composition unexpectedly replaced an unrelated host module")
    architecture.update(method="synthstrip+" + registration, composition_mode=mode,
                        candidate_modules=list(model.candidate_modules),
                        parameters=sum(p.numel() for p in model.parameters()), compact=False,
                        input_shape=[96, 96, 96], registration_source=provenance, state_transfer=transfer,
                        previous_run=previous_evidence, adapter_sha256=fingerprint(adapter),
                        registration_weights_sha256=fingerprint(registration_weights) if registration_weights else None,
                        extraction_weights_sha256=fingerprint(extraction_weights) if extraction_weights else None,
                        scope="Full SynthStrip and dense registration, five retained affine stages and unchanged remaining host")
    architecture["unchanged_modules"].pop("reg_net", None)
    return model, config, architecture
