"""Reconstruct a full host and generated adapter on any configured machine."""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .full_replacement import AdapterFailure, check_code
from .native import verify_original
from .sources import verify
from .upstream import synthstrip_core


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
        host_state = torch.load(host_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(host_state.get("model", host_state), strict=True)
    untouched = {name: module for name, module in model.named_children() if name != "ext_net"}
    if adapter is not None:
        model.ext_net = GeneratedExtraction(adapter)
    if any(getattr(model, name) is not module for name, module in untouched.items()):
        raise RuntimeError("A host module other than extraction changed")
    architecture = {"original_parameters": count, "parameters": sum(p.numel() for p in model.parameters()),
                    "unchanged_modules": {name: sum(p.numel() for p in module.parameters()) for name, module in untouched.items()}}
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not all(torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Checkpoint contains nonfinite tensors")
        model.load_state_dict(state, strict=True)
    return model, config, architecture
