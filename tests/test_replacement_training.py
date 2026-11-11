import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from unibrain_agent.replacement_training import configure_boundaries, configure_trainable, metrics_from_cases, parser, preflight
from unibrain_agent.replacement_training import rng_state, restore_rng, train_mode, original_losses
from unibrain_agent.full_connectivity import connect_model


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext_net = nn.Linear(2, 2)
        self.reg_net = nn.Linear(2, 2)
        self.seg_net = nn.Linear(2, 2)


class ReplacementTrainingTests(unittest.TestCase):
    def test_tissue_target_keeps_all_classes_when_warp_excludes_a_class(self):
        mask = torch.full((1, 1, 4, 4, 4), .5, requires_grad=True)
        logits = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        output = [None, [mask], None, None, None, logits, torch.zeros_like(mask),
                  None, None, None, torch.randn(1, 2, requires_grad=True)]
        losses = original_losses(output, torch.ones_like(mask), torch.tensor([1]))
        losses["segmentation"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_frozen_batchnorm_buffers_stay_fixed_but_input_gradients_survive(self):
        model = Toy()
        model.ext_net = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        configure_trainable(model, "synthstrip", "joint", "host")
        train_mode(model)
        normalization = model.ext_net[1]
        before = normalization.running_mean.clone()
        image = torch.randn(4, 2, requires_grad=True)
        model.ext_net(image).square().mean().backward()
        self.assertFalse(normalization.training)
        self.assertTrue(model.reg_net.training)
        self.assertTrue(torch.equal(before, normalization.running_mean))
        self.assertGreater(image.grad.abs().sum().item(), 0)

    def test_rng_resume_replays_cpu_randomness(self):
        import numpy as np
        import random
        with patch("torch.cuda.get_rng_state_all", return_value=[]), patch("torch.cuda.set_rng_state_all"):
            saved = rng_state()
            expected = (random.random(), np.random.rand(), torch.rand(3))
            restore_rng(saved)
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(expected[:2], actual[:2])
            self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_connected_host_independent_registration_and_segmentation(self):
        class Extract(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.1))

            def forward(self, image, training):
                mask = torch.sigmoid(image * self.weight)
                return [image * mask], [mask]

        class Registration(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = nn.Parameter(torch.eye(3, 4)[None])

            def forward(self, reference, image):
                self.forward_grid = torch.nn.functional.affine_grid(self.theta, image.shape, align_corners=True)
                self.inverse_grid = self.forward_grid + .01
                self.velocity = self.forward_grid.permute(0, 4, 1, 2, 3)
                warped = torch.nn.functional.grid_sample(image, self.forward_grid, align_corners=True)
                return [warped], [self.theta], [self.theta]

        class Features(nn.Module):
            def forward(self, images):
                value = images.flatten(1).mean(1, keepdim=True)
                return torch.cat((value, value.square()), 1)

        class Graph(nn.Module):
            def forward(self, features):
                return features @ features.transpose(1, 2)

        class Predictor(nn.Module):
            def forward(self, graph, features):
                return torch.stack((graph.sum(), features.sum())).reshape(1, 2)

        for method, target in (("voxelmorph", "reg_net"), ("nnunet", "seg_net")):
            host = nn.Module()
            host.ext_net, host.reg_net = Extract(), Registration()
            host.seg_net, host.par_net = nn.Conv3d(1, 4, 1), nn.Conv3d(1, 117, 1)
            host.feature_net, host.graph_gen, host.gnn_pred = Features(), Graph(), Predictor()
            host.if_pred_aal = False
            model = connect_model(host)
            configure_trainable(model, method, "independent_official", "all")
            handles = configure_boundaries(model, method, "independent_official", False)
            image = torch.rand(1, 1, 6, 6, 6)
            tissue = (torch.arange(216) % 4).reshape_as(image).float()
            atlas = (torch.arange(216) % 117).reshape_as(image).float()
            fake = SimpleNamespace(feature_perturbation=lambda value, amount: value, edge_perturbation=lambda value, *args: value)
            with patch.dict("sys.modules", {"model": fake}):
                output = model(image, image, tissue, atlas, True)
            self.assertFalse(any(p.requires_grad for p in getattr(model, target).parameters()))
            if method == "voxelmorph":
                self.assertFalse(model.last_inverse_grid.requires_grad)
                self.assertFalse(output[6].requires_grad)
            else:
                self.assertFalse(output[5].requires_grad)
                self.assertTrue(model.last_inverse_grid.requires_grad)
            for handle in handles:
                handle.remove()

    def test_new_host_and_independent_parameters(self):
        model = Toy()
        configure_trainable(model, "synthstrip", "joint", "new")
        self.assertTrue(all(p.requires_grad for p in model.ext_net.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.reg_net.parameters()))
        configure_trainable(model, "synthstrip", "joint", "host")
        self.assertFalse(any(p.requires_grad for p in model.ext_net.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.reg_net.parameters()))
        configure_trainable(model, "synthstrip", "independent_finetuned", "all", standalone=True)
        self.assertTrue(all(p.requires_grad for p in model.ext_net.parameters()))
        configure_trainable(model, "synthstrip", "independent_finetuned", "all")
        self.assertFalse(any(p.requires_grad for p in model.ext_net.parameters()))

    def test_independent_dense_grid_cannot_leak_gradients(self):
        class Dense(nn.Module):
            def forward(self, image):
                self.forward_grid = image * 2
                self.inverse_grid = image * 3
                self.velocity = image * 4
                return (image * 5,)
        model = Toy()
        model.reg_net = Dense()
        handles = configure_boundaries(model, "voxelmorph", "independent_official", False)
        image = torch.ones(2, requires_grad=True)
        result = model.reg_net(image)
        self.assertFalse(result[0].requires_grad)
        self.assertFalse(model.reg_net.inverse_grid.requires_grad)
        for handle in handles:
            handle.remove()
        self.assertTrue(model.reg_net(image)[0].requires_grad)

    def test_metrics_use_probabilities_for_auc_and_define_missing_class(self):
        def row(label, probability):
            return {"label": label, "probability": probability, "prediction": int(probability >= .5),
                    **{key: .7 for key in ("brain_dice", "tissue_dice", "atlas_dice", "registration_dice", "reference_loss")}}
        metrics = metrics_from_cases([row(0, .1), row(1, .9), row(0, .4), row(1, .8)])
        self.assertEqual(metrics["auc"], 1.)
        self.assertEqual(metrics["sensitivity"], 1.)
        metrics = metrics_from_cases([row(0, .1), row(0, .2)])
        self.assertIsNone(metrics["auc"])
        self.assertIsNone(metrics["sensitivity"])

    def test_official_control_cannot_use_random_weights(self):
        args = parser().parse_args(["--source", "source", "--data", "data", "--output", "output", "--setting", "independent_official"])
        with self.assertRaisesRegex(ValueError, "official weights"):
            preflight(args)

    def test_training_fold_initialization_rejects_other_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"test")
            args = parser().parse_args(["--source", "source", "--data", "data", "--output", "output",
                                        "--initialization", "training_fold", "--weights", str(path),
                                        "--dataset", "abide", "--fold", "1", "--weights-dataset", "abide", "--weights-fold", "2"])
            with self.assertRaisesRegex(ValueError, "same dataset"):
                preflight(args)


if __name__ == "__main__":
    unittest.main()
