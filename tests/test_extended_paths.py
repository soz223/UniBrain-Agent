import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from unibrain_agent.full_connectivity import connect_model


class Extraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))

    def forward(self, image, training):
        mask = torch.sigmoid(image * self.weight)
        return [image * mask], [mask]


class Registration(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.01))

    def forward(self, reference, image):
        theta = torch.eye(3, 4).unsqueeze(0).to(image)
        direction = torch.zeros_like(theta)
        direction[:, 0, 3] = 1
        theta = theta + direction * (self.offset + .02 * image.mean())
        inverse = theta - 2 * direction * theta[:, 0, 3]
        grid = F.affine_grid(theta, image.shape, align_corners=True)
        return [F.grid_sample(image, grid, align_corners=True)], [theta], [inverse]


class Features(nn.Module):
    def forward(self, regions):
        return regions.flatten(1).mean(1, keepdim=True)


class Graph(nn.Module):
    def forward(self, features):
        return features @ features.transpose(1, 2)


class Classifier(nn.Module):
    def forward(self, graph, features):
        weights = torch.arange(1, 117).to(features).reshape(1, 116, 1)
        score = (features * weights).sum((1, 2)) + graph.mean((1, 2))
        return torch.stack((score, -score), dim=1)


class ExtendedPathsTests(unittest.TestCase):
    def test_learned_atlas_keeps_registration_and_parcellation_paths(self):
        torch.manual_seed(17)
        host = nn.Module()
        host.if_pred_aal = True
        host.learned_parcellation = True
        host.candidate_modules = ('par_net',)
        host.preprocessor = nn.Conv3d(1, 1, 1)
        host.ext_net, host.reg_net = Extraction(), Registration()
        host.seg_net, host.par_net = nn.Conv3d(1, 4, 1), nn.Conv3d(1, 117, 1)
        host.feature_net, host.graph_gen, host.gnn_pred = Features(), Graph(), Classifier()
        image = torch.rand(1, 1, 6, 6, 6, requires_grad=True)
        atlas = (torch.arange(216) % 117).reshape_as(image).float()
        upstream = types.ModuleType('model')
        upstream.feature_perturbation = lambda value, rate: value
        upstream.edge_perturbation = lambda value, nodes, rate: value
        with patch.dict(sys.modules, {'model': upstream}):
            for cut in ('none', 'parcellation'):
                model = connect_model(host, cut=cut)
                logits = model(image, image, atlas % 4, atlas, True)[-1]
                inputs = [host.preprocessor.weight, host.ext_net.weight,
                          host.reg_net.offset, host.par_net.weight, image]
                gradients = torch.autograd.grad(logits[0, 0], inputs, allow_unused=True)
                self.assertTrue(torch.isfinite(logits).all())
                for index, gradient in enumerate(gradients):
                    expected = cut == 'none' or index in (0, 4)
                    active = gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                    self.assertEqual(bool(active), expected, (cut, index))


if __name__ == '__main__':
    unittest.main()
