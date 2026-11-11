"""Explicit full-host gradient path interventions for controlled ablations."""

import torch
from torch import nn
from torch.nn import functional as F


def categorical(probabilities, strategy, training):
    if strategy == "soft" or not training:
        return probabilities
    if strategy == "gumbel_softmax":
        return F.gumbel_softmax(probabilities.clamp_min(1e-8).log(), tau=1., hard=True, dim=1)
    if strategy == "straight_through":
        hard = F.one_hot(probabilities.argmax(1), probabilities.shape[1]).movedim(-1, 1).to(probabilities.dtype)
        return hard - probabilities.detach() + probabilities
    raise ValueError("Unknown repair strategy")


def transfer(labels, values, grid, strategy="soft", training=True, nearest=False):
    onehot = (labels == values.view(1, -1, 1, 1, 1)).to(grid.dtype)
    probabilities = F.grid_sample(onehot, grid, mode="nearest" if nearest else "bilinear",
                                  padding_mode="border", align_corners=True)
    probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)
    return categorical(probabilities, strategy, training)


class ConnectedHost(nn.Module):
    def __init__(self, host, cut, repair, undo):
        super().__init__()
        for name, module in host.named_children():
            self.add_module(name, module)
        self.cut, self.repair, self.undo = cut, repair, undo
        self.if_pred_aal = host.if_pred_aal

    def forward(self, ref, mov, ref_am, ref_aal, if_train):
        from model import feature_perturbation, edge_perturbation
        extracted, masks = self.ext_net(mov, if_train)
        if self.undo == "mask":
            masks = [(mask >= .5).to(mask.dtype) for mask in masks]
            value, extracted = mov, []
            for mask in masks:
                value = value * mask
                extracted.append(value)
        if self.cut in ("extraction", "all"):
            extracted = [value.detach() for value in extracted]
        warped, affines, inverses = self.reg_net(ref, extracted[-1])
        if hasattr(self.reg_net, "inverse_grid"):
            inverse, forward = self.reg_net.inverse_grid, self.reg_net.forward_grid
        else:
            inverse = F.affine_grid(inverses[-1], mov.shape, align_corners=True)
            forward = F.affine_grid(affines[-1], mov.shape, align_corners=True)
        if self.cut in ("registration", "all"):
            inverse = inverse.detach()
        self.last_forward_grid, self.last_inverse_grid = forward, inverse
        tissue, parcellation = self.seg_net(mov), self.par_net(mov)
        tissue_prob = transfer(ref_am, torch.arange(4, device=mov.device), inverse, self.repair, if_train, self.undo == "nearest")
        values = torch.unique(ref_aal, sorted=True)
        if len(values) != 117 or values[0] != 0:
            raise ValueError("AAL116 vocabulary and background are required")
        atlas = transfer(ref_aal, values, inverse, self.repair, if_train, self.undo == "nearest")
        if self.cut in ("parcellation", "all"):
            atlas = atlas.detach()
        self.atlas_probabilities = atlas
        features = self.feature_net((atlas * mov).squeeze(0)[1:]).unsqueeze(0)
        if if_train:
            features = feature_perturbation(features, .05)
        raw_graph = self.graph_gen(features)
        graph = edge_perturbation(raw_graph, features.size(1), .05) if self.undo == "graph" and if_train else F.softplus(raw_graph)
        self.graph_weights = graph
        logits = self.gnn_pred(graph, features)
        atlas_labels = values[atlas.argmax(1)].unsqueeze(1)
        return extracted, masks, warped, affines, inverses, tissue, tissue_prob, parcellation, atlas_labels, graph, logits

    def regularization_loss(self):
        if hasattr(self.reg_net, "velocity"):
            return sum(torch.diff(self.reg_net.velocity, dim=d).square().mean() for d in (2, 3, 4)) / 3
        return next(self.parameters()).new_zeros(())


def connect_model(model, cut="none", repair="soft", undo="none"):
    if cut not in ("none", "extraction", "registration", "parcellation", "all"):
        raise ValueError("Invalid gradient boundary")
    if repair not in ("soft", "straight_through", "gumbel_softmax") or undo not in ("none", "nearest", "graph", "mask"):
        raise ValueError("Invalid repair intervention")
    return ConnectedHost(model, cut, repair, undo)
