"""Pinned preprocessing components at the full UniBrain spatial resolution."""

import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .paths import SOURCES
from .sources import load, package, verify


REPOSITORIES = {"redwgan": "denoise_redwgan", "deepn4": "DeepN4", "deepnormalize": "deepNormalize",
                "harmonizing_flows": "HarmonizingFlows"}


class FullPreprocessor(nn.Module):
    def __init__(self, method):
        super().__init__()
        if method not in REPOSITORIES:
            raise ValueError("Unknown full preprocessing component: " + method)
        self.method = method
        self.provenance = verify([REPOSITORIES[method]])
        if method == "redwgan":
            self.core = load("denoise_redwgan/model5.py", {"GeneratorNet"})["GeneratorNet"]()
            self.adaptations = ["Original generator at 96 cubed; the discriminator and adversarial objective are not included"]
            widths = [32, 64, 128, 256]
        elif method == "deepn4":
            self.core = load("DeepN4/src/model_all.py", {"Synbo_UNet3D"})["Synbo_UNet3D"](1, 1)
            self.adaptations = ["Original log bias field core at 96 cubed; corrected image is input times exp of negative log field",
                                "Official external resampling and SciPy field smoothing are not included"]
            widths = [32, 64, 128, 256, 512]
        elif method == "deepnormalize":
            package("deepNormalize")
            from DeepNormalize.model.UNet.unet import UNet
            config = json.loads((SOURCES / "deepNormalize/config.json").read_text(encoding="utf-8"))["model"]["deep_normalize"]
            self.core = UNet(config, in_channels=1, out_channels=1, is_preprocessor=True)
            self.adaptations = ["Original preprocessor widths; single T1 input and output instead of two modalities",
                                "Standalone normalization component; segmentation and domain discriminator are not included"]
            widths = list(config["n_filters"])
        else:
            self.core = load("HarmonizingFlows/step3_Adapting_Harmonizer_using_NF/harmonizer_model.py",
                             {"Harmonizer", "ConcatELU", "double_conv"})["Harmonizer"](1)
            self.adaptations = ["All 96 slices are processed by the original 2D Harmonizer",
                                "Each 96 by 96 slice is zero padded to 224 by 224 and center cropped after processing",
                                "Activation recomputation preserves the full slice set and original architecture",
                                "Harmonizer component only; normalizing flow supervision and test-time adaptation are not reproduced"]
            widths = [16, 32, 48, 64, 64]
        self.architecture = {"method": method, "component": type(self.core).__name__, "widths": widths,
                             "parameters": sum(p.numel() for p in self.parameters()), "input_shape": [1, 1, 96, 96, 96],
                             "output_shape": [1, 1, 96, 96, 96], "compact": False,
                             "initialization": "random unless explicitly loaded", "provenance": self.provenance,
                             "adaptations": self.adaptations}
        if method == "harmonizing_flows":
            self.architecture.update(slice_axis=2, processed_slices=96, padded_slice_shape=[224, 224])

    def forward(self, image):
        if tuple(image.shape) != (1, 1, 96, 96, 96) or not image.is_floating_point():
            raise ValueError("Full preprocessing requires a floating [1,1,96,96,96] volume")
        if self.method == "deepn4":
            return image * torch.exp(-self.core(image, image.device))
        if self.method != "harmonizing_flows":
            return self.core(image)
        slices = image[0].permute(1, 0, 2, 3)
        values = []
        for plane in slices.split(1):
            padded = F.pad(plane, (64, 64, 64, 64))
            result = checkpoint(self.core, padded, use_reentrant=False) if torch.is_grad_enabled() else self.core(padded)
            values.append(result[:, :, 64:160, 64:160])
        return torch.cat(values, dim=0).permute(1, 0, 2, 3).unsqueeze(0)


def build_preprocessor(method):
    return FullPreprocessor(method)
