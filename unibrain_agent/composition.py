import torch
from torch import nn
from torch.nn import functional as F

from .adapters import ClassifierAdapter, DenseRegistration, ImageAdapter, PairGraph, ParcellationAdapter, TissueAdapter, resize
from .catalog import ADAPTATIONS, validate_selection
from .pipeline import AffineRegistration, Pipeline, PipelineConfig, SynthStripAdapter, atlas_probabilities
from .sources import load
from .upstream import unibrain_core


class ComposedPipeline(Pipeline):
    def __init__(self, selection):
        choices = validate_selection(selection)
        super().__init__(PipelineConfig(extraction="synthstrip" if choices["extraction"] == "M02" else "unibrain",
                                        soft_atlas=True, continuous_edges=True, stages=choices))
        for stage, name in [("denoising", "denoise_net"), ("bias", "bias_net"), ("normalization", "norm_net")]:
            if choices[stage] != "none":
                setattr(self, name, ImageAdapter(choices[stage]))
        if choices["extraction"] == "M03":
            ns = load("reg_ernet/model.py", {"conv_block", "Unet", "multi_stage_ext", "encoder"})
            self.ext_net = ns["multi_stage_ext"](96, 1, 10, enc_nf=[1, 4, 8, 8, 16, 16], dec_nf=[16, 16, 8, 8, 4, 1])
            self.reg_net = AffineRegistration(1, encoder_class=ns["encoder"])
        elif choices["registration"] in {"M06", "M07"}:
            self.reg_net = DenseRegistration(choices["registration"], self.reg_net)
        if choices["segmentation"] != "unibrain":
            self.seg_net = TissueAdapter(choices["segmentation"])
        if choices["parcellation"] != "soft_atlas":
            self.par_net = ParcellationAdapter(choices["parcellation"])
        if choices["graph"] == "M11":
            self.graph_gen = PairGraph()
        if choices["classification"] != "unibrain":
            self.gnn_pred = ClassifierAdapter(choices["classification"])

    def forward(self, sample, template, augment=False, hard_mask=False):
        raw = sample["raw_img_np"]
        if raw.shape != (1, 1, 96, 96, 96):
            raise ValueError("The current composition profile requires one 96 cubed T1 image")
        image = self.bias_net(self.denoise_net(raw))
        extracted, masks = self.ext_net(image, if_train=not hard_mask)
        normalized = self.norm_net(extracted[-1])
        registration = self.reg_net(template["image"], normalized)
        regularization = 0
        if isinstance(self.reg_net, DenseRegistration):
            warped, affine, grid, forward_grid, velocity = registration
            regularization = sum(torch.diff(velocity, dim=dim).square().mean() for dim in (2, 3, 4)) / 3
        else:
            warps, affines, inverses = registration
            warped, affine = warps[-1], affines[-1]
            grid = F.affine_grid(inverses[-1], image.shape, align_corners=True)
            forward_grid = F.affine_grid(affine, image.shape, align_corners=True)
        tissue_outputs = self.seg_net(image)
        if not isinstance(tissue_outputs, list):
            tissue_outputs = [tissue_outputs]
        tissue_target = sample["raw_img_gm_mask_np"]
        parcellation_loss = 0
        if isinstance(self.par_net, ParcellationAdapter):
            par_logits = self.par_net(warped)
            probabilities = resize(par_logits.softmax(1), image.shape[2:])
            atlas = F.grid_sample(probabilities, grid, padding_mode="border", align_corners=True)
            atlas = atlas / atlas.sum(1, keepdim=True).clamp_min(1e-8)
            values = torch.tensor(template["atlas_label_values"], device=image.device)
            labels = torch.searchsorted(values, sample["raw_img_aal_mask_np"].contiguous()).float()
            aligned_labels = F.grid_sample(labels, forward_grid, mode="nearest", align_corners=True)
            target = F.interpolate(aligned_labels, size=par_logits.shape[2:], mode="nearest")[:, 0].long()
            parcellation_loss = F.cross_entropy(par_logits, target)
        else:
            atlas = atlas_probabilities(template["atlas"], grid, True)
        roi_images = (atlas * image)[0, 1:]
        features = self.feature_net(roi_images).unsqueeze(0)
        if augment:
            features = unibrain_core().feature_perturbation(features, 0.05)
        edges = self.graph_gen(features)
        prediction = self.gnn_pred(edges, features)
        return {"mask": masks[-1], "extracted": extracted[-1], "warped": warped, "affine": affine,
                "inverse_grid": grid, "forward_grid": forward_grid, "tissue_logits": tissue_outputs[0],
                "tissue_aux": tissue_outputs[1:], "tissue_target": tissue_target, "atlas": atlas,
                "features": features, "raw_edges": edges, "edges": edges, "logits": prediction,
                "regularization": regularization, "parcellation_loss": parcellation_loss}

    def description(self):
        result = super().description()
        result["adaptations"] = [ADAPTATIONS[x] for x in dict.fromkeys(self.config.stages.values()) if x in ADAPTATIONS]
        result["supervision"] = "Native ground truth tissue CE; auxiliary nnU Net head loss; AAL CE for learned parcellation; smoothness for dense registration"
        result["initialization"] = "Random weights. Compact or low resolution CPU adapters, not pretrained full methods."
        return result
