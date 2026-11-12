"""Full resolution anatomy candidates loaded from pinned upstream definitions."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .sources import load, verify


def volume_contract(image):
    if image.shape != (1, 1, 96, 96, 96):
        raise ValueError("The full UniBrain interface requires one single channel 96 cubed volume")


@contextmanager
def preserve_running_buffers(module):
    saved = [(buffer, buffer.detach().clone()) for layer in module.modules()
             if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.training
             for buffer in layer.buffers(recurse=False)]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, previous in saved:
                buffer.copy_(previous)


def memory_forward(module, value):
    if not torch.is_grad_enabled():
        return module(value)
    return checkpoint(module, value, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), preserve_running_buffers(module)))


def fastsurfer_core():
    prefix = "FastSurfer/FastSurferCNN/models/"
    blocks = load(prefix + "sub_module.py", {"InputDenseBlock", "CompetitiveDenseBlock", "CompetitiveDenseBlockInput",
                  "CompetitiveEncoderBlock", "CompetitiveEncoderBlockInput", "CompetitiveDecoderBlock", "ClassifierBlock"})
    namespace = load(prefix + "networks.py", {"FastSurferCNNBase", "FastSurferCNN"},
                     {"torch": torch, "nn": nn, "Tensor": torch.Tensor, "sm": SimpleNamespace(**blocks)})
    parameters = {"num_channels": 7, "num_filters": 64, "num_classes": 117, "kernel_h": 5,
                  "kernel_w": 5, "kernel_c": 1, "stride_conv": 1, "pool": 2, "stride_pool": 2}
    return namespace["FastSurferCNN"](parameters, padded_size=96)


class FullFastSurferParcellation(nn.Module):
    def __init__(self, slice_batch=4):
        super().__init__()
        if not 1 <= slice_batch <= 96:
            raise ValueError("Slice batch size must be between 1 and 96")
        self.slice_batch = slice_batch
        self.core = fastsurfer_core()

    def forward(self, image):
        volume_contract(image)
        offsets = torch.arange(-3, 4, device=image.device)
        output = []
        for start in range(0, 96, self.slice_batch):
            centers = torch.arange(start, min(start + self.slice_batch, 96), device=image.device)
            indices = (centers[:, None] + offsets[None]).clamp(0, 95)
            slabs = image[0, 0][indices]
            output.append(memory_forward(self.core, slabs))
        return torch.cat(output, dim=0).permute(1, 0, 2, 3).unsqueeze(0)


class FullHighResParcellation(nn.Module):
    def __init__(self):
        super().__init__()
        namespace = load("nobrainer/nobrainer/models/highresnet.py", {"HighResNet", "_ResBlock", "_ZeroPadChannels"})
        self.core = namespace["HighResNet"](n_classes=117)

    def forward(self, image):
        volume_contract(image)
        value = self.core.init_conv(image)
        for block in self.core.stage1:
            value = memory_forward(block, value)
        value = self.core.stage2_proj(self.core.pad1(value))
        for block in self.core.stage2:
            value = memory_forward(block, value)
        value = self.core.stage3_proj(self.core.pad2(value))
        for block in self.core.stage3:
            value = memory_forward(block, value)
        return memory_forward(self.core.classifier, self.core.dropout(value))


def ernet_definitions():
    return load("reg_ernet/model.py", {"conv_block", "Unet", "multi_stage_ext", "encoder"})


class FullERNetRegistration(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = ernet_definitions()["encoder"]()
        self.stage = 5

    def forward(self, reference, image):
        volume_contract(reference)
        volume_contract(image)
        cumulative = torch.eye(4, device=image.device, dtype=image.dtype).unsqueeze(0)
        current, warped, affines, inverses = image, [], [], []
        for _ in range(self.stage):
            affine = self.affine(torch.cat((reference, current), dim=1))
            bottom = affine.new_tensor([0., 0., 0., 1.]).reshape(1, 1, 4)
            cumulative = torch.cat((affine, bottom), dim=1) @ cumulative
            inverse = torch.linalg.inv(cumulative)[:, :3]
            self.forward_grid = F.affine_grid(cumulative[:, :3], image.shape, align_corners=True)
            self.inverse_grid = F.affine_grid(inverse, image.shape, align_corners=True)
            current = F.grid_sample(image, self.forward_grid, mode="bilinear", align_corners=True)
            warped.append(current)
            affines.append(cumulative[:, :3])
            inverses.append(inverse)
        return warped, affines, inverses


def build_anatomy_modules(method, slice_batch=4):
    repositories = {"fastsurfer": "FastSurfer", "nobrainer": "nobrainer", "ernet": "reg_ernet"}
    if method not in repositories:
        raise ValueError("Unsupported full anatomy candidate")
    provenance = verify([repositories[method]])
    if method == "fastsurfer":
        modules = {"par_net": FullFastSurferParcellation(slice_batch)}
        details = {"profile": "Classic FastSurferCNN single view core, 64 filters and 5 by 5 kernels",
                   "slice_axis": "First tensor spatial axis; anatomical orientation is not inferred from NPY",
                   "slices_processed": 96, "neighbor_slices": 7, "slice_batch": slice_batch,
                   "three_view_ensemble": False, "replaced_output_head": "117 AAL logits; requires task training, not an official label conversion"}
    elif method == "nobrainer":
        modules = {"par_net": FullHighResParcellation()}
        details = {"profile": "Pinned nobrainer PyTorch HighResNet registry implementation",
                   "widths": [16, 32, 64], "blocks_per_stage": 3, "dilations": [1, 2, 4],
                   "replaced_output_head": "117 AAL logits; requires task training, not an official label conversion"}
    else:
        modules = {"ext_net": ernet_definitions()["multi_stage_ext"](96, 5, 10), "reg_net": FullERNetRegistration()}
        details = {"profile": "Full ERNet extraction and affine registration cores",
                   "extraction_stages": 5, "registration_stages": 5, "gamma": 10,
                   "extraction_encoder": [1, 16, 32, 32, 64, 64], "affine_encoder": [2, 16, 32, 64, 128, 256, 512],
                   "adaptation": "Device aware original cumulative affine composition with additional differentiable inverse matrices"}
    architecture = {"method": method, "source": provenance, "input_shape": [1, 1, 96, 96, 96],
                    "parameters": {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in modules.items()},
                    "compact": False, "spatial_downsampling_before_core": False, "initialization": "random; no official checkpoint loaded",
                    "checkpoint_scope": "Native extraction and registration parameter keys" if method == "ernet" else "Native backbone and explicitly replaced AAL output head",
                    "memory_policy": "Native ERNet activations" if method == "ernet" else "Activation recomputation restores BatchNorm buffers to avoid duplicate statistics updates",
                    **details}
    return modules, architecture
