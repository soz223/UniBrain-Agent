import json
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .paths import SOURCES
from .sources import bnt_core, load, package, voxel_core


def resize(image, size):
    size = (size,) * 3 if isinstance(size, int) else size
    return F.interpolate(image, size=size, mode="trilinear", align_corners=True)


def identity_grid(image):
    theta = torch.eye(3, 4, dtype=image.dtype, device=image.device).unsqueeze(0)
    return F.affine_grid(theta, image.shape, align_corners=True)


class ImageAdapter(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == "M01":
            self.core = load("DeepN4/src/model_all.py", {"Synbo_UNet3D"})["Synbo_UNet3D"](1, 1)
        elif method == "M04":
            self.core = normalizer()
        elif method == "M05":
            self.core = load("HarmonizingFlows/step3_Adapting_Harmonizer_using_NF/harmonizer_model.py",
                             {"Harmonizer", "ConcatELU", "double_conv"})["Harmonizer"](1)
        elif method == "M14":
            self.core = load("denoise_redwgan/model5.py", {"GeneratorNet"})["GeneratorNet"]()
        else:
            raise ValueError("Unsupported image adapter")

    def forward(self, image):
        shape = image.shape[2:]
        if self.method == "M01":
            log_field = self.core(resize(image, 16), image.device)
            return image / resize(2 * torch.tanh(log_field / 2), shape).exp()
        if self.method == "M05":
            small = resize(image, (8, 224, 224))
            slices = small[0].permute(1, 0, 2, 3)
            values = [checkpoint(self.core, x, use_reentrant=False) for x in slices.split(2)]
            return resize(torch.cat(values).permute(1, 0, 2, 3).unsqueeze(0), shape)
        small = resize(image, 32 if self.method == "M04" else 16)
        return resize(self.core(small), shape)


def normalizer(segmentation=False):
    package("deepNormalize")
    from DeepNormalize.model.UNet.unet import UNet
    config = json.loads((SOURCES / "deepNormalize/config.json").read_text())["model"]["deep_normalize"]
    config["n_filters"] = [4, 8, 16, 16, 16]
    config["n_classes"] = 4
    return UNet(config, in_channels=1, out_channels=4 if segmentation else 1, is_preprocessor=not segmentation)


class TissueAdapter(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == "M04":
            self.core = normalizer(segmentation=True)
        elif method == "M08":
            package("dynamic-network-architectures")
            from dynamic_network_architectures.architectures.unet import PlainConvUNet
            self.core = PlainConvUNet(input_channels=1, n_stages=3, features_per_stage=[4, 8, 16],
                                     conv_op=nn.Conv3d, kernel_sizes=3, strides=[1, 2, 2], n_conv_per_stage=2,
                                     num_classes=4, n_conv_per_stage_decoder=2, conv_bias=True,
                                     norm_op=nn.InstanceNorm3d, norm_op_kwargs={"affine": True},
                                     nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": 0.01}, deep_supervision=True)
        else:
            raise ValueError("Unsupported tissue adapter")

    def forward(self, image):
        result = self.core(resize(image, 32))
        outputs = result if isinstance(result, list) else [result]
        if self.method == "M04":
            outputs = [x.clamp_min(1e-8).log() for x in outputs]
        return [resize(outputs[0], image.shape[2:]), *outputs[1:]]


class ParcellationAdapter(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == "M09":
            path = "FastSurfer/FastSurferCNN/models/"
            sub = load(path + "sub_module.py", {
                "InputDenseBlock", "CompetitiveDenseBlock", "CompetitiveDenseBlockInput", "CompetitiveEncoderBlock",
                "CompetitiveEncoderBlockInput", "CompetitiveDecoderBlock", "ClassifierBlock"})
            ns = load(path + "networks.py", {"FastSurferCNNBase", "FastSurferCNN"}, {"torch": torch, "nn": nn, "Tensor": torch.Tensor, "sm": SimpleNamespace(**sub)})
            params = {"num_channels": 7, "num_filters": 16, "num_classes": 117, "kernel_h": 5,
                      "kernel_w": 5, "kernel_c": 1, "stride_conv": 1, "pool": 2, "stride_pool": 2}
            self.core = ns["FastSurferCNN"](params, padded_size=32)
        elif method == "M10":
            ns = load("nobrainer/nobrainer/models/highresnet.py", {"HighResNet", "_ResBlock", "_ZeroPadChannels"})
            self.core = ns["HighResNet"](n_classes=117)
        else:
            raise ValueError("Unsupported parcellation adapter")

    def forward(self, image):
        small = resize(image, 32 if self.method == "M09" else 16)
        if self.method == "M09":
            offsets = torch.arange(-3, 4, device=image.device)
            slices = [small[:, 0].index_select(1, (offsets + i).clamp(0, 31)) for i in range(32)]
            outputs = [checkpoint(self.core, torch.cat(slices[i:i + 4]), use_reentrant=False) for i in range(0, 32, 4)]
            logits = torch.cat(outputs).permute(1, 0, 2, 3).unsqueeze(0)
        else:
            logits = self.core(small)
        return logits


class PairGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = load("FBNETGEN/model/model.py", {"Embed2GraphByLinear"})["Embed2GraphByLinear"](16, roi_num=116)
        for name in ("rel_rec", "rel_send"):
            value = getattr(self.core, name)
            delattr(self.core, name)
            self.core.register_buffer(name, value)
        nn.init.constant_(self.core.fc_cat.bias, 0.5)

    def forward(self, features):
        graph = self.core(features).squeeze(-1)
        return (graph + graph.transpose(1, 2)) / 2


class ClassifierAdapter(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.core = (bnt_core() if method == "M13" else
                     load("FBNETGEN/model/model.py", {"BrainNetCNN", "E2EBlock"})["BrainNetCNN"](116))

    def forward(self, edges, features):
        return self.core(None, edges) if self.method == "M13" else self.core(edges)


class DenseRegistration(nn.Module):
    def __init__(self, method, affine):
        super().__init__()
        self.affine = affine
        self.method = method
        if method == "M06":
            self.core = voxel_core()
        elif method == "M07":
            ns = load("reg_lapirn/Code/Functions.py", {"generate_grid_unit"})
            load("reg_lapirn/Code/miccai2020_model_stage.py", {"Miccai2020_LDR_laplacian_unit_add_lvl1", "PreActBlock",
                 "DiffeomorphicTransform_unit", "SpatialTransform_unit"}, ns)
            self.core = ns["Miccai2020_LDR_laplacian_unit_add_lvl1"](2, 3, 2, is_train=True, imgshape=(8, 8, 8))
            value = self.core.grid_1
            del self.core.grid_1
            self.core.register_buffer("grid_1", value)
        else:
            raise ValueError("Unsupported dense registration")

    def forward(self, fixed, moving):
        warped, affines, inverses = self.affine(fixed, moving)
        small, target = resize(warped[-1], 32), resize(fixed, 32)
        if self.method == "M06":
            _, _, velocity = self.core(small, target)
            positive = self.core.integrate(velocity)
            negative = self.core.integrate(-velocity)
            # VoxelMorph stores displacement as D,H,W voxel units; grid_sample uses W,H,D normalized units.
            positive, negative = [value[:, [2, 1, 0]] * (2 / 31) for value in (positive, negative)]
        else:
            positive, _, _, velocity, _ = self.core(small, target)
            negative = self.core.diff_transform(-velocity, self.core.grid_1)
        shape = moving.shape[2:]
        identity = identity_grid(moving)
        positive = resize(positive, shape).permute(0, 2, 3, 4, 1)
        negative = resize(negative, shape).permute(0, 2, 3, 4, 1)
        forward = torch.einsum("bdhwc,bkc->bdhwk", identity + positive, affines[-1][:, :, :3]) + affines[-1][:, None, None, None, :, 3]
        inverse_affine = F.affine_grid(inverses[-1], moving.shape, align_corners=True)
        inverse = inverse_affine + F.grid_sample(negative.permute(0, 4, 1, 2, 3), inverse_affine,
                                               padding_mode="border", align_corners=True).permute(0, 2, 3, 4, 1)
        registered = F.grid_sample(moving, forward, align_corners=True)
        return registered, affines[-1], inverse, forward, velocity
