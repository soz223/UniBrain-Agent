"""Temporary normalization ablations for checkpoint diagnosis."""

from contextlib import contextmanager

from torch.nn.modules.batchnorm import _BatchNorm


@contextmanager
def sample_statistics(module):
    """Use current-input statistics without changing stored BatchNorm buffers."""
    saved = []
    try:
        for layer in module.modules():
            if isinstance(layer, _BatchNorm):
                saved.append((layer, layer.training, layer.track_running_stats,
                              layer.running_mean, layer.running_var))
                layer.training = False
                layer.track_running_stats = False
                layer.running_mean = None
                layer.running_var = None
        yield
    finally:
        for layer, training, tracking, mean, variance in saved:
            layer.training = training
            layer.track_running_stats = tracking
            layer.running_mean = mean
            layer.running_var = variance
