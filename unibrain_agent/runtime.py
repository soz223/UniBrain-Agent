import os
import sys

import torch

from .paths import DATA, LOCAL, ORIGINAL, RUNS, BUILDS, SOURCES


def settings():
    return {"device": os.environ.get("UNIBRAIN_DEVICE", LOCAL.get("device", "auto")),
            "threads": int(os.environ.get("UNIBRAIN_THREADS", LOCAL.get("threads", 4))),
            "original_python": os.environ.get("UNIBRAIN_ORIGINAL_PYTHON", LOCAL.get("original_python", sys.executable)),
            "data": str(DATA), "original_source": str(ORIGINAL), "runs": str(RUNS),
            "builds": str(BUILDS), "sources": str(SOURCES)}


def device_for(value=None):
    value = value or settings()["device"]
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: cpu, cuda or cuda:N")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError("Requested CUDA device is unavailable in this Python environment")
    return device


def move_model(model, device):
    model.to(device)
    for module in model.modules():
        for name, value in vars(module).copy().items():
            if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
                setattr(module, name, value.to(device))
    return model
