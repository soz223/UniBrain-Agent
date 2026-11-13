"""Bounded execution and reproducible training diagnostics."""

import math
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np


class TrainingBudget:
    def __init__(self, seconds=None):
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Wall time must be finite and positive")
        self.seconds = seconds
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc)
        self.signal_number = None
        self.handlers = {}

    @property
    def stop_reason(self):
        if self.signal_number is not None:
            return "termination_signal"
        if self.seconds is not None and time.monotonic() - self.started >= self.seconds:
            return "wall_time_budget"
        return None

    def request_stop(self, number, frame):
        self.signal_number = number

    def describe(self):
        end = self.started_at + timedelta(seconds=self.seconds) if self.seconds is not None else None
        return {"seconds": self.seconds, "started_at": self.started_at.isoformat(),
                "deadline": end.isoformat() if end else None,
                "stop_boundary": "Completed optimizer update or epoch evaluation; save resumable checkpoint"}

    def __enter__(self):
        if self.seconds is not None and threading.current_thread() is threading.main_thread():
            for number in (signal.SIGTERM, signal.SIGINT):
                self.handlers[number] = signal.getsignal(number)
                signal.signal(number, self.request_stop)
        return self

    def __exit__(self, *error):
        for number, handler in self.handlers.items():
            signal.signal(number, handler)


def normalize_template(image, mode="none"):
    image = np.asarray(image, dtype=np.float32)
    low, high = float(image.min()), float(image.max())
    if not np.isfinite(image).all() or (mode == "minmax" and high <= low):
        raise ValueError("The intensity template must be finite and nonconstant when scaled")
    if mode not in ("none", "minmax"):
        raise ValueError("Unknown template normalization")
    result = (image - low) / (high - low) if mode == "minmax" else image
    return result, {"mode": mode, "source_min": low, "source_max": high,
                    "output_min": float(result.min()), "output_max": float(result.max())}


def training_probe_indices(labels, count, seed):
    labels = np.asarray(labels).reshape(-1)
    if count < 0:
        raise ValueError("Training probe size must be nonnegative")
    if not count:
        return []
    classes = np.unique(labels)
    count = min(count, len(labels))
    if count < len(classes):
        raise ValueError("Training probe must include every training class")
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(np.flatnonzero(labels == value))) for value in classes]
    result = []
    while len(result) < count:
        for group in groups:
            if group and len(result) < count:
                result.append(int(group.pop()))
    return sorted(result)
