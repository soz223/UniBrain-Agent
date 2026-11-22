"""Checkpoint-compatible per-image normalization for shared registration stages."""

from torch import nn
from torch.nn import functional as F


class ImageBatchNorm3d(nn.BatchNorm3d):
    def forward(self, image):
        self._check_input_dim(image)
        if image.shape[0] != 1:
            raise ValueError('Registration normalization requires one subject per forward')
        return F.batch_norm(image, None, None, self.weight, self.bias, True, 0., self.eps)


def configure_registration(module, mode):
    if mode not in {'stored', 'per_image'}:
        raise ValueError('Unknown registration normalization')
    if mode == 'per_image':
        for layer in module.modules():
            if type(layer) is nn.BatchNorm3d:
                layer.__class__ = ImageBatchNorm3d
    return mode
