from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .upstream import homogeneous, synthstrip_core, unibrain_core

TASKS = ("extraction", "registration", "segmentation", "classification")
MODULES = ("ext_net", "reg_net", "seg_net", "par_net", "feature_net", "graph_gen", "gnn_pred")
MODULES = ("denoise_net", "bias_net", "norm_net", *MODULES)


@dataclass
class PipelineConfig:
    extraction: str = "unibrain"
    soft_atlas: bool = False
    continuous_edges: bool = False
    registration_steps: int = 1
    stages: dict | None = None


class SynthStripAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = synthstrip_core().StripModel(nb_features=4, nb_levels=5, max_features=16)

    def forward(self, image, if_train):
        mask = torch.sigmoid(-self.core(image))
        if not if_train:
            mask = (mask >= 0.5).to(image.dtype)
        return [image * mask], [mask]


class AffineRegistration(nn.Module):
    def __init__(self, stages, encoder_class=None):
        super().__init__()
        self.affine = (encoder_class or unibrain_core().encoder)([2, 4, 8, 16, 16, 16, 16])
        self.affine.fc_loc = nn.Sequential(nn.Linear(16 * 3 ** 3, 32), nn.LeakyReLU(0.2), nn.Linear(32, 12))
        nn.init.zeros_(self.affine.fc_loc[2].weight)
        with torch.no_grad():
            self.affine.fc_loc[2].bias.copy_(torch.eye(3, 4).flatten())
        self.stage = stages

    def forward(self, ref, moving):
        previous = torch.eye(4, device=moving.device, dtype=moving.dtype).unsqueeze(0)
        current = moving
        warped, affines, inverses = [], [], []
        for _ in range(self.stage):
            affine = self.affine(torch.cat((ref, current), dim=1))
            previous = homogeneous(affine) @ previous
            inverse = torch.linalg.inv(previous)
            grid = F.affine_grid(previous[:, :3], ref.shape, align_corners=True)
            current = F.grid_sample(moving, grid, mode="bilinear", align_corners=True)
            warped.append(current)
            affines.append(previous[:, :3])
            inverses.append(inverse[:, :3])
        return warped, affines, inverses


def one_hot(labels, classes):
    return F.one_hot(labels[:, 0].long(), classes).permute(0, 4, 1, 2, 3).float()


def atlas_probabilities(labels, grid, soft):
    if soft:
        probs = F.grid_sample(one_hot(labels, 117), grid, mode="bilinear", padding_mode="border", align_corners=True)
        return probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
    warped = F.grid_sample(labels, grid, mode="nearest", align_corners=True)
    return one_hot(warped, 117)


class Pipeline(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PipelineConfig()
        if self.config.extraction not in {"unibrain", "synthstrip"}:
            raise ValueError("Unknown extraction module")
        core = unibrain_core()
        self.denoise_net = nn.Identity()
        self.bias_net = nn.Identity()
        self.norm_net = nn.Identity()
        self.ext_net = (SynthStripAdapter() if self.config.extraction == "synthstrip" else
                        core.multi_stage_ext(96, 1, enc_nf=[1, 4, 8, 8, 16, 16], dec_nf=[16, 16, 8, 8, 4, 1]))
        self.reg_net = AffineRegistration(self.config.registration_steps)
        self.seg_net = core.Unet([1, 4, 8, 8, 16, 16], [16, 16, 8, 8, 4, 4])
        self.par_net = core.Unet([1, 4, 8, 8, 16, 16], [16, 16, 8, 8, 4, 117])
        self.par_net.requires_grad_(False)
        self.feature_net = core.Roi_feature_net(8, 16)
        self.graph_gen = core.Graph_gen_net_norm()
        self.gnn_pred = core.GNN_predictor(16, 16)

    def forward(self, sample, template, augment=False, hard_mask=False):
        image = sample["raw_img_np"]
        if image.shape != (1, 1, 96, 96, 96):
            raise ValueError("Current UniBrain ROI encoder requires batch size 1 and 96 cubed input")
        extracted, masks = self.ext_net(image, if_train=not hard_mask)
        warped, affines, inverses = self.reg_net(template["image"], extracted[-1])
        tissue_logits = self.seg_net(image)
        grid = F.affine_grid(inverses[-1], image.shape, align_corners=True)
        tissue_target = F.grid_sample(template["tissue"], grid, mode="nearest", align_corners=True)
        atlas = atlas_probabilities(template["atlas"], grid, self.config.soft_atlas)
        roi_images = (atlas * image)[0, 1:]
        features = self.feature_net(roi_images).unsqueeze(0)
        if augment:
            features = unibrain_core().feature_perturbation(features, 0.05)
        raw_edges = self.graph_gen(features)
        edges = raw_edges
        if augment and not self.config.continuous_edges:
            edges = unibrain_core().edge_perturbation(edges, features.size(1), 0.05)
        prediction = self.gnn_pred(edges, features)
        return {"mask": masks[-1], "extracted": extracted[-1], "warped": warped[-1],
                "affine": affines[-1], "inverse_grid": grid, "tissue_logits": tissue_logits,
                "tissue_target": tissue_target, "atlas": atlas, "features": features,
                "raw_edges": raw_edges, "edges": edges, "logits": prediction}

    def description(self):
        return {"config": asdict(self.config),
                "parameters": {name: sum(p.numel() for p in getattr(self, name).parameters()) for name in MODULES},
                "profile": "compact widths, native 96 cubed, batch 1, one extraction stage",
                "adaptations": ["CPU/device-aware affine composition with the original transform convention",
                                "Smaller channel widths, affine head and ROI encoder for local execution",
                                "Fixed 4 tissue and 117 atlas channels preserve label identities",
                                "Unused predicted-parcellation head is frozen and skipped, matching the default atlas route",
                                "No random image augmentation in the paired comparison",
                                "This is an adapted execution baseline, not a reproduction of published accuracy"]}


def loss_terms(output, sample, template, extraction_loss="bce"):
    mask = output["mask"].clamp(1e-6, 1 - 1e-6)
    target = sample["raw_img_brain_mask_np"]
    extraction = F.binary_cross_entropy(mask, target)
    if extraction_loss == "bce_dice":
        extraction = extraction + 1 - (2 * (mask * target).sum() + 1) / (mask.sum() + target.sum() + 1)
    elif extraction_loss != "bce":
        raise ValueError("Unknown extraction loss")
    tissue = F.cross_entropy(output["tissue_logits"], output["tissue_target"][:, 0].long())
    for logits in output.get("tissue_aux", []):
        target = F.interpolate(output["tissue_target"], size=logits.shape[2:], mode="nearest")[:, 0].long()
        tissue = tissue + 0.5 * F.cross_entropy(logits, target)
    tissue = tissue + 0.2 * output.get("parcellation_loss", 0)
    return {"extraction": extraction,
            "registration": unibrain_core().GCC()(output["warped"], template["image"]) + 0.01 * output.get("regularization", 0),
            "segmentation": tissue,
            "classification": F.cross_entropy(output["logits"], sample["label"])}
