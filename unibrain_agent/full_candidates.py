"""Full resolution registration and planned segmentation candidates."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .data import digest
from .native import verify_original
from .sources import load, package, verify


def voxelmorph():
    ns = load("reg_voxelmorph_master/voxelmorph/torch/modelio.py", {"store_config_args", "LoadableModel"})
    layers = load("reg_voxelmorph_master/voxelmorph/torch/layers.py", {"SpatialTransformer", "VecInt", "ResizeTransform"})
    ns["layers"] = SimpleNamespace(**layers)
    load("reg_voxelmorph_master/voxelmorph/py/utils.py", {"default_unet_features"}, ns)
    load("reg_voxelmorph_master/voxelmorph/torch/networks.py", {"Unet", "VxmDense", "ConvBlock"}, ns)
    return ns["VxmDense"]((96, 96, 96), bidir=True)


def lapirn():
    ns = load("reg_lapirn/Code/Functions.py", {"generate_grid_unit"})
    names = {f"Miccai2020_LDR_laplacian_unit_add_lvl{i}" for i in (1, 2, 3)}
    load("reg_lapirn/Code/miccai2020_model_stage.py", names | {
        "PreActBlock", "DiffeomorphicTransform_unit", "SpatialTransform_unit"}, ns)
    previous = None
    for level, size in ((1, 24), (2, 48), (3, 96)):
        kwargs = {} if level == 1 else {f"model_lvl{level - 1}": previous}
        previous = ns[f"Miccai2020_LDR_laplacian_unit_add_lvl{level}"](
            2, 3, 7, is_train=True, imgshape=(size,) * 3, range_flow=0.4, **kwargs)
    for module in previous.modules():
        if hasattr(module, "grid_1"):
            grid = module.grid_1
            del module.grid_1
            module.register_buffer("grid_1", grid, persistent=False)
    return previous


def planned_unet(plans):
    if plans is None:
        raise ValueError("nnU-Net requires a saved dataset-specific plans JSON; no generic plan is substituted")
    plan = json.loads(Path(plans).read_text(encoding="utf-8"))
    if plan.get("transpose_forward", [0, 1, 2]) != [0, 1, 2]:
        raise ValueError("This integration requires identity planned axis order")
    config = plan["configurations"]["3d_fullres"]
    architecture = config["architecture"]
    if architecture["network_class_name"] != "dynamic_network_architectures.architectures.unet.PlainConvUNet":
        raise ValueError("Only an explicit PlainConvUNet plan is currently supported")
    if config["patch_size"] != [96, 96, 96]:
        raise ValueError("This full-volume integration requires a native 96 cubed plan")
    kwargs = dict(architecture["arch_kwargs"])
    symbols = {"torch.nn.modules.conv.Conv3d": nn.Conv3d,
               "torch.nn.modules.instancenorm.InstanceNorm3d": nn.InstanceNorm3d,
               "torch.nn.modules.activation.LeakyReLU": nn.LeakyReLU,
               "torch.nn.modules.dropout.Dropout3d": nn.Dropout3d}
    symbols.update({"torch.nn." + operator.__name__: operator for operator in tuple(symbols.values())})
    for key in architecture.get("_kw_requires_import", []):
        if kwargs[key] is not None:
            if kwargs[key] not in symbols:
                raise ValueError("Unsupported planned operator: " + kwargs[key])
            kwargs[key] = symbols[kwargs[key]]
    if kwargs.get("n_stages", 0) < 5 or min(kwargs.get("features_per_stage", [0])) < 16:
        raise ValueError("Compact network plans are not accepted by this executor")
    package("dynamic-network-architectures")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    return PlainConvUNet(input_channels=1, num_classes=4, deep_supervision=True, **kwargs)


def compose_grids(image, affine, inverse_affine, positive, negative):
    identity = F.affine_grid(torch.eye(3, 4, device=image.device, dtype=image.dtype)[None], image.shape, align_corners=True)
    forward = torch.einsum("bdhwc,bkc->bdhwk", identity + positive.permute(0, 2, 3, 4, 1), affine[:, :, :3])
    forward = forward + affine[:, None, None, None, :, 3]
    inverse = F.affine_grid(inverse_affine, image.shape, align_corners=True)
    inverse = inverse + F.grid_sample(negative, inverse, padding_mode="border", align_corners=True).permute(0, 2, 3, 4, 1)
    return forward, inverse


class FullRegistration(nn.Module):
    def __init__(self, method, affine):
        super().__init__()
        self.method, self.affine = method, affine
        self.core = voxelmorph() if method == "voxelmorph" else lapirn()

    def forward(self, fixed, moving):
        warps, transforms, inverses = self.affine(fixed, moving)
        result = self.core(warps[-1], fixed)
        if self.method == "voxelmorph":
            velocity = result[2]
            positive, negative = [self.core.integrate(v) for v in (velocity, -velocity)]
            if self.core.fullsize:
                positive, negative = [self.core.fullsize(v) for v in (positive, negative)]
            positive, negative = [v[:, [2, 1, 0]] * (2 / 95) for v in (positive, negative)]
        else:
            positive, velocity = result[0], result[3]
            negative = self.core.diff_transform(-velocity, self.core.grid_1)
        self.velocity = velocity
        self.forward_grid, self.inverse_grid = compose_grids(moving, transforms[-1], inverses[-1], positive, negative)
        return [*warps[:-1], F.grid_sample(moving, self.forward_grid, align_corners=True)], transforms, inverses


class DenseHost(nn.Module):
    def __init__(self, host):
        super().__init__()
        for name, module in host.named_children():
            self.add_module(name, module)
        self.if_pred_aal = host.if_pred_aal

    def forward(self, ref, mov, ref_am, ref_aal, if_train):
        from model import feature_perturbation, edge_perturbation
        extracted, masks = self.ext_net(mov, if_train)
        warped, transforms, inverses = self.reg_net(ref, extracted[-1])
        self.last_forward_grid, self.last_inverse_grid = self.reg_net.forward_grid, self.reg_net.inverse_grid
        tissue, parcellation = self.seg_net(mov), self.par_net(mov)
        warped_tissue = F.grid_sample(ref_am, self.last_inverse_grid, mode="nearest", align_corners=True)
        warped_atlas = F.grid_sample(ref_aal, self.last_inverse_grid, mode="nearest", align_corners=True)
        values = torch.unique(ref_aal, sorted=True)
        if values.numel() != 117 or values[0] != 0:
            raise ValueError("Expected 116 AAL regions plus background")
        probabilities = (ref_aal == values.view(1, -1, 1, 1, 1)).to(mov.dtype)
        probabilities = F.grid_sample(probabilities, self.last_inverse_grid, align_corners=True)
        features = self.feature_net((probabilities * mov).squeeze(0)[1:]).unsqueeze(0)
        if if_train:
            features = feature_perturbation(features, 0.05)
        graph = self.graph_gen(features)
        if if_train:
            graph = edge_perturbation(graph, features.size(1), 0.05)
        logits = self.gnn_pred(graph, features)
        return extracted, masks, warped, transforms, inverses, tissue, warped_tissue, parcellation, warped_atlas, graph, logits

    def regularization_loss(self):
        return sum(torch.diff(self.reg_net.velocity, dim=d).square().mean() for d in (2, 3, 4)) / 3


def build_candidate_model(source, method, adapter=None, weights=None, plans=None, host_checkpoint=None):
    from scripts.evaluate_upstream import original_config
    source = Path(source).resolve()
    verify_original(source)
    repositories = {"voxelmorph": "reg_voxelmorph_master", "lapirn": "reg_lapirn", "nnunet": "dynamic-network-architectures"}
    if method not in repositories:
        raise ValueError("Unsupported full candidate: " + method)
    provenance = verify([repositories[method]])
    existing = sys.modules.get("model")
    if existing and Path(existing.__file__).resolve() != source / "model.py":
        raise ValueError("Use an isolated process for each pinned host")
    sys.path.insert(0, str(source))
    from model import UniBrain
    config = original_config(source)
    keys = ("img_size", "ext_stage", "reg_stage", "if_pred_aal", "enc_nf", "dec_nf", "enc_affine",
            "enc_seg", "dec_seg", "enc_par", "dec_par", "feature_net_hidden", "feature_dim", "gnn_hidden")
    model = UniBrain(*(config[k] for k in keys))
    original = sum(p.numel() for p in model.parameters())
    if original != 305812248:
        raise ValueError("Original full host parameter contract changed")
    if host_checkpoint:
        state = torch.load(host_checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state.get("model_state_dict", state))
        model.load_state_dict(state, strict=True)
    if method == "nnunet":
        model.seg_net = planned_unet(plans)
        target = model.seg_net
    else:
        model.reg_net = FullRegistration(method, model.reg_net)
        model = DenseHost(model)
        target = model.reg_net.core
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state.get("network_weights", state))
        target.load_state_dict(state, strict=True)
    architecture = {"method": method, "original_parameters": original, "parameters": sum(p.numel() for p in model.parameters()),
                    "source": provenance, "input_shape": [96, 96, 96], "compact": False,
                    "weights_sha256": digest(weights) if weights else None,
                    "host_checkpoint_sha256": digest(host_checkpoint) if host_checkpoint else None,
                    "plans_sha256": digest(plans) if plans else None,
                    "adaptation": "Planned tissue head with native auxiliary outputs and normalized multi-scale tissue CE" if method == "nnunet" else
                    "Five original affine stages plus full dense core; soft atlas sampling for ROI features; original graph augmentation",
                    "checkpoint_scope": "Candidate core only; official compatibility requires strict loading"}
    return model, config, architecture
