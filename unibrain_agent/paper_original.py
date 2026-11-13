"""Original UniBrain equations and Appendix A.4 configuration."""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .full_connectivity import transfer
from .native import verify_original


def perturb_graph(graph, rate=.01):
    """Preserve retained similarity weights; added edges have unit weight."""
    positive = graph.clamp_min(0)
    result = positive.clone()
    for batch in range(len(graph)):
        present = (positive[batch] > 0).nonzero()
        absent = (positive[batch] == 0).nonzero()
        count = int(len(present) * rate)
        removed = present[torch.randperm(len(present), device=graph.device)[:count]]
        added = absent[torch.randperm(len(absent), device=graph.device)[:count]]
        result[batch, removed[:, 0], removed[:, 1]] = 0
        result[batch, added[:, 0], added[:, 1]] = 1
    return result


class PaperUniBrain(nn.Module):
    def __init__(self):
        super().__init__()
        from model import multi_stage_ext, multi_stage_reg, Unet, Roi_feature_net, Graph_gen_net_norm, GNN_predictor
        self.ext_net = multi_stage_ext(96, 1)
        self.reg_net = multi_stage_reg(96, 5)
        self.seg_net = Unet([1, 128, 256, 256, 512, 512], [512, 256, 256, 256, 128, 4])
        self.feature_net = Roi_feature_net(256, 256)
        self.graph_gen = Graph_gen_net_norm()
        self.gnn_pred = GNN_predictor(128, 256)
        self.gnn_pred.classifier = nn.Sequential(nn.Linear(128 * 116, 128), nn.LeakyReLU(.2), nn.Linear(128, 2))
        self.if_pred_aal = False
        self.paper_original = True
        self.candidate_modules = ("ext_net", "reg_net", "feature_net", "gnn_pred")

    def forward(self, ref, mov, ref_am, ref_aal, if_train):
        extracted, masks = self.ext_net(mov, if_train)
        warped, affines, inverses = self.reg_net(ref, extracted[-1])
        inverse = F.affine_grid(inverses[-1], mov.shape, align_corners=True)
        self.last_inverse_grid = inverse
        self.last_forward_grid = F.affine_grid(affines[-1], mov.shape, align_corners=True)
        tissue = self.seg_net(mov)
        target = transfer(ref_am, torch.arange(4, device=mov.device), inverse)
        values = torch.unique(ref_aal, sorted=True)
        if len(values) != 117 or values[0] != 0:
            raise ValueError("Expected AAL116 and background")
        atlas = transfer(ref_aal, values, inverse)
        self.atlas_probabilities = atlas
        features = self.feature_net((atlas * mov).squeeze(0)[1:]).unsqueeze(0)
        if if_train:
            features = features + torch.randn_like(features) * .01
        graph = self.graph_gen(features).clamp_min(0)
        if if_train:
            graph = perturb_graph(graph)
        self.graph_weights = graph
        logits = self.gnn_pred(graph, features)
        labels = values[atlas.argmax(1)].unsqueeze(1)
        return extracted, masks, warped, affines, inverses, tissue, target, None, labels, graph, logits


def build_paper_model(source, registration_weight=.1):
    from scripts.evaluate_upstream import original_config
    source = Path(source).resolve()
    verify_original(source)
    existing = sys.modules.get("model")
    if existing is not None and Path(existing.__file__).resolve() != source / "model.py":
        raise RuntimeError("Use an isolated worker for the pinned source")
    sys.path.insert(0, str(source))
    config = original_config(source)
    config.update(feature_dim=256, learning_rate=1e-5, weight_decay=1e-6, lambda_reg=registration_weight)
    model = PaperUniBrain()
    model.paper_registration_weight = registration_weight
    return model, config, {"profile": "original_paper", "parameters": sum(p.numel() for p in model.parameters()),
                          "reference": "UniBrain KDD 2024, equations 1 to 11 and Appendix A.4",
                          "loss_weights": {"extraction": 1., "registration": registration_weight, "segmentation": 1., "classification": 1.},
                          "implementation_choices": ["128-unit MLP before the binary output", "Unit weights for randomly added edges",
                                                     "Border padding for trilinear categorical transfer", "Upstream 1-CC differs from negative CC by a constant"]}
