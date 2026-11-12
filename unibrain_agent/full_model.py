"""Reconstruct a full host and generated adapter on any configured machine."""

import sys
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .full_replacement import AdapterFailure, check_code
from .native import verify_original
from .sources import verify
from .upstream import synthstrip_core


@dataclass(frozen=True)
class VerifiedHostCheckpoint:
    path: Path
    dataset: str
    fold: int
    sha256: str
    report: Path = None

    def __fspath__(self):
        return str(self.path)


def host_file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_host_file(spec):
    if not isinstance(spec, VerifiedHostCheckpoint):
        raise ValueError("Pretrained hosts require VerifiedHostCheckpoint with target dataset, fold and SHA256")
    if (not isinstance(spec.dataset, str) or not spec.dataset.strip() or type(spec.fold) is not int or spec.fold < 1
            or not isinstance(spec.sha256, str) or len(spec.sha256) != 64):
        raise ValueError("Host checkpoint target dataset, fold and SHA256 must be explicit")
    if host_file_hash(spec.path) != spec.sha256:
        raise ValueError("Host checkpoint hash mismatch or file changed during loading")


def host_identity(identity, spec):
    if (not isinstance(identity, dict) or identity.get("method") != "original"
            or identity.get("dataset") != spec.dataset or type(identity.get("fold")) is not int
            or identity["fold"] != spec.fold or identity.get("split_id") is not None):
        raise ValueError("A trained original host from this dataset and fold is required")


def host_training_report(spec):
    record = json.loads(Path(spec.report).read_text(encoding="utf-8"))
    config = record.get("config", {})
    identity = {key: record.get(key, config.get(key)) for key in ("dataset", "fold", "method", "split_id")}
    for key in ("dataset", "fold", "method", "split_id"):
        if key in record and key in config and record[key] != config[key]:
            raise ValueError("Host report contains conflicting training identities")
    host_identity(identity, spec)
    checkpoint = record.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    expected = record.get("checkpoint_sha256") or checkpoint.get("sha256") or record.get("checkpoint_hashes", {}).get(Path(spec.path).name)
    updates = record.get("optimizer_updates", record.get("steps", record.get("updates", 0)))
    if (record.get("status") not in ("complete", "completed", "passed", "budget_exhausted")
            or expected != spec.sha256 or type(updates) is not int or updates < 1):
        raise ValueError("Host report must bind the checkpoint hash to successful training updates")
    return {"report_sha256": host_file_hash(spec.report), "reported_updates": updates}


def load_host_checkpoint(spec):
    verify_host_file(spec)
    before = Path(spec.path).stat()
    try:
        checkpoint = torch.load(spec.path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        checkpoint = torch.load(spec.path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Host checkpoint must contain a model state dictionary")
    expected_identity = {"dataset": spec.dataset, "fold": spec.fold, "method": "original", "split_id": None}
    for declaration in (checkpoint, checkpoint.get("config", {})):
        if isinstance(declaration, dict):
            for key, expected in expected_identity.items():
                if key in declaration and (declaration[key] != expected or key == "fold" and type(declaration[key]) is not int):
                    raise ValueError("Legacy host metadata contradicts the target training identity")
    internal = "identity" in checkpoint or "progress" in checkpoint
    if internal:
        host_identity(checkpoint.get("identity"), spec)
        progress = checkpoint.get("progress", {})
        updates = progress.get("steps", 0) if isinstance(progress, dict) else 0
        if type(updates) is not int or updates < 1:
            raise ValueError("A trained original host with positive optimizer steps is required")
    elif spec.report is None:
        raise ValueError("Raw original host states require an explicit hash bound training report")
    external = host_training_report(spec) if spec.report is not None else {}
    if internal and external and external["reported_updates"] < updates:
        raise ValueError("Host report training count contradicts checkpoint progress")
    wrappers = [name for name in ("model", "model_state_dict") if name in checkpoint]
    if len(wrappers) > 1:
        raise ValueError("Ambiguous host model state mapping")
    state = checkpoint[wrappers[0]] if wrappers else checkpoint
    if not isinstance(state, dict) or not state or any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("Host model state must map parameter names to tensors")
    after = Path(spec.path).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Host checkpoint changed during inspection")
    evidence = {"kind": "trainer_checkpoint_metadata" if internal else "external_training_report",
                "internal_identity_verified": internal, "updates": updates if internal else external["reported_updates"],
                "dataset": spec.dataset, "fold": spec.fold, "method": "original", "checkpoint_sha256": spec.sha256,
                "state_mapping": wrappers[0] if wrappers else "raw_state_dict", **external}
    return state, evidence


class GeneratedExtraction(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        scope = {"torch": torch, "F": F}
        exec(compile(check_code(Path(adapter).read_text(encoding="utf-8")), str(adapter), "exec"), scope)
        self.adapt = scope["adapt"]
        self.core = synthstrip_core().StripModel()

    def forward(self, image, if_train):
        try:
            result = self.adapt(self.core, image, if_train)
        except (TypeError, ValueError, IndexError, RuntimeError) as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
                raise
            raise AdapterFailure(str(error)) from error
        if (not isinstance(result, (tuple, list)) or len(result) != 2
                or any(not isinstance(values, list) or len(values) != 1 for values in result)):
            raise AdapterFailure("Extraction must return two single element tensor lists")
        if any(not isinstance(values[0], torch.Tensor) or values[0].shape != image.shape for values in result):
            raise AdapterFailure("Extraction outputs must preserve the input shape")
        return result


def build_full_model(source, adapter=None, checkpoint=None, host_checkpoint=None):
    from scripts.evaluate_upstream import original_config

    host_state, host_evidence = load_host_checkpoint(host_checkpoint) if host_checkpoint is not None else (None, None)
    source = Path(source).resolve()
    verify_original(source)
    if adapter is not None:
        verify(["freesurfer_synthstrip"])
    existing = sys.modules.get("model")
    if existing is not None and Path(existing.__file__).resolve() != source / "model.py":
        raise RuntimeError("Another model module is loaded; use an isolated worker")
    sys.path.insert(0, str(source))
    from model import UniBrain

    config = original_config(source)
    keys = ("img_size", "ext_stage", "reg_stage", "if_pred_aal", "enc_nf", "dec_nf", "enc_affine",
            "enc_seg", "dec_seg", "enc_par", "dec_par", "feature_net_hidden", "feature_dim", "gnn_hidden")
    model = UniBrain(*(config[key] for key in keys))
    count = sum(p.numel() for p in model.parameters())
    if count != 305812248 or config["reg_stage"] != 5:
        raise RuntimeError("Full original architecture contract changed")
    if host_checkpoint is not None:
        model.load_state_dict(host_state, strict=True)
        verify_host_file(host_checkpoint)
    untouched = {name: module for name, module in model.named_children() if name != "ext_net"}
    if adapter is not None:
        model.ext_net = GeneratedExtraction(adapter)
    if any(getattr(model, name) is not module for name, module in untouched.items()):
        raise RuntimeError("A host module other than extraction changed")
    architecture = {"original_parameters": count, "parameters": sum(p.numel() for p in model.parameters()),
                    "host_training_evidence": host_evidence,
                    "unchanged_modules": {name: sum(p.numel() for p in module.parameters()) for name, module in untouched.items()}}
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not all(torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Checkpoint contains nonfinite tensors")
        model.load_state_dict(state, strict=True)
    return model, config, architecture
