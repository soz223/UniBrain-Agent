import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from unibrain_agent.replacement_training import configure_boundaries, configure_trainable, metrics_from_cases, parser, preflight
from unibrain_agent.replacement_training import rng_state, restore_rng, train_mode, original_losses
from unibrain_agent.replacement_training import checkpoint_state, check_checkpoint_space, trained_host_evidence, write_checkpoint
from unibrain_agent.replacement_training import classification_gradients, classification_gradient_paths, total_candidate_gradients, atlas_class_indices
from unibrain_agent.full_connectivity import connect_model


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext_net = nn.Linear(2, 2)
        self.reg_net = nn.Linear(2, 2)
        self.seg_net = nn.Linear(2, 2)


class ReplacementTrainingTests(unittest.TestCase):
    def test_stagewise_trains_classifier_only_and_detaches_stages(self):
        model = Toy()
        model.ext_net = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model.gnn_pred = nn.Linear(2, 2)
        configure_trainable(model, "original", "original_frozen_stages", "all")
        handles = configure_boundaries(model, "original", "original_frozen_stages", False)
        train_mode(model)
        image = torch.randn(4, 2, requires_grad=True)
        before = model.ext_net[1].running_mean.clone()
        features = model.seg_net(model.reg_net(model.ext_net(image)))
        model.gnn_pred(features).square().sum().backward()
        self.assertFalse(features.requires_grad)
        self.assertIsNone(image.grad)
        self.assertTrue(torch.equal(before, model.ext_net[1].running_mean))
        self.assertTrue(all(parameter.grad is not None for parameter in model.gnn_pred.parameters()))
        self.assertFalse(any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith("gnn_pred.")))
        for handle in handles:
            handle.remove()

    def test_host_checkpoint_requires_trained_matching_fold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "host.pt"
            args = SimpleNamespace(host_report=None, host_checkpoint=path, host_sha256="expected", dataset="abide", fold=1)
            payload = {"identity": {"dataset": "abide", "fold": 1, "method": "original"}, "progress": {"steps": 3}}
            torch.save(payload, path)
            self.assertEqual(trained_host_evidence(args)["updates"], 3)
            args.fold = 2
            with self.assertRaisesRegex(ValueError, "this dataset and fold"):
                trained_host_evidence(args)
            args.fold = 1
            payload["progress"]["steps"] = 0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "trained original"):
                trained_host_evidence(args)

    def test_evaluation_checkpoint_is_small_and_not_resumable(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        with patch("unibrain_agent.replacement_training.rng_state", return_value={"test": 1}):
            best = checkpoint_state(model, optimizer, {"steps": 2}, {}, evaluation_only=True)
            last = checkpoint_state(model, optimizer, {"steps": 2}, {})
        self.assertEqual(best["checkpoint_kind"], "evaluation")
        self.assertEqual(set(best), {"model", "identity", "progress", "checkpoint_kind"})
        self.assertEqual(last["checkpoint_kind"], "resume")
        self.assertIn("optimizer", last)
        self.assertIn("coordination", last)
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(OSError, "Existing results were retained"):
                check_checkpoint_space(Path.cwd(), best)

    def test_weights_checkpoint_payload_and_atomic_storage_reuse(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        identity = {"method": "original", "dataset": "abide", "fold": 1}
        state = checkpoint_state(model, optimizer, {"steps": 2}, identity, weights_only=True)
        self.assertEqual(state["checkpoint_kind"], "evaluation_weights_only")
        self.assertEqual(set(state), {"model", "identity", "progress", "checkpoint_kind"})
        with tempfile.TemporaryDirectory() as folder, patch("unibrain_agent.replacement_training.check_checkpoint_space"):
            best, last = Path(folder) / "best.pt", Path(folder) / "last.pt"
            self.assertEqual(write_checkpoint(best, state), "serialized")
            snapshot = {"path": best, "progress": dict(state["progress"]), "identity": identity}
            method = write_checkpoint(last, state, snapshot)
            if method == "hardlink":
                self.assertEqual(best.stat().st_ino, last.stat().st_ino)
            saved = torch.load(best, weights_only=True)
            with torch.no_grad():
                model.weight.add_(1.)
            newer = checkpoint_state(model, optimizer, {"steps": 3}, identity, weights_only=True)
            with patch("os.link") as link:
                self.assertEqual(write_checkpoint(last, newer, snapshot), "serialized")
                link.assert_not_called()
            torch.testing.assert_close(torch.load(best, weights_only=True)["model"]["weight"], saved["model"]["weight"])
            self.assertFalse(torch.equal(torch.load(last, weights_only=True)["model"]["weight"], saved["model"]["weight"]))
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_weights_checkpoint_falls_back_when_links_are_unavailable(self):
        model = nn.Linear(2, 2)
        state = checkpoint_state(model, None, {"steps": 1}, {}, weights_only=True)
        with tempfile.TemporaryDirectory() as folder, patch("unibrain_agent.replacement_training.check_checkpoint_space"):
            best, last = Path(folder) / "best.pt", Path(folder) / "last.pt"
            write_checkpoint(best, state)
            snapshot = {"path": best, "progress": state["progress"], "identity": state["identity"]}
            with patch("os.link", side_effect=OSError("Unsupported filesystem")):
                self.assertEqual(write_checkpoint(last, state, snapshot), "serialized")
            self.assertTrue(best.is_file() and last.is_file())
            self.assertEqual(torch.load(last, weights_only=True)["checkpoint_kind"], "evaluation_weights_only")

    def test_weights_mode_cannot_request_resume(self):
        args = parser().parse_args(["--source", "source", "--data", "data", "--output", "output", "--method", "original",
                                    "--setting", "original", "--checkpoint-mode", "weights", "--resume", "output/last.pt"])
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            preflight(args)

    def test_gradient_evidence_distinguishes_auxiliary_segmentation(self):
        model = Toy()
        image = torch.randn(2, 2)
        cls = model.reg_net(model.ext_net(image)).square().mean()
        seg = model.seg_net(image).square().mean()
        record = classification_gradients(model, cls, "nnunet", True)["seg_net"]
        self.assertTrue(record["expected_absent"])
        self.assertFalse(record["required"])
        self.assertFalse(record["nonzero"])
        ext = classification_gradients(model, cls, "synthstrip", True)["ext_net"]
        self.assertTrue(ext["required"])
        self.assertTrue(ext["nonzero"])
        (cls + seg).backward()
        self.assertTrue(total_candidate_gradients(model, "nnunet")["seg_net"]["nonzero"])

    def test_classifier_input_detach_cannot_hide_behind_auxiliary_gradient(self):
        model = nn.Module()
        model.gnn_pred = nn.Linear(2, 2)
        model.candidate_modules = ("gnn_pred",)
        image = torch.tensor([[.2, .4]], requires_grad=True)
        classification = model.gnn_pred(image.detach()).square().mean()
        with patch("torch.autograd.grad", wraps=torch.autograd.grad) as gradient:
            evidence = classification_gradient_paths(model, classification, "bnt", True, image)
            self.assertEqual(gradient.call_count, 1)
        self.assertTrue(evidence["candidates"]["gnn_pred"]["nonzero"])
        self.assertTrue(evidence["input"]["required"])
        self.assertFalse(evidence["input"]["nonzero"])
        (classification + image.square().mean()).backward()
        self.assertGreater(image.grad.abs().sum(), 0)
        connected = classification_gradient_paths(model, model.gnn_pred(image).square().mean(), "bnt", True, image)
        self.assertTrue(connected["input"]["finite"] and connected["input"]["nonzero"])

    def test_candidate_and_image_paths_share_one_reverse_pass(self):
        model = Toy()
        model.candidate_modules = ("ext_net", "reg_net")
        image = torch.randn(3, 2, requires_grad=True)
        loss = model.reg_net(model.ext_net(image)).square().mean()
        with patch("torch.autograd.grad", wraps=torch.autograd.grad) as gradient:
            evidence = classification_gradient_paths(model, loss, "voxelmorph", True, image)
            self.assertEqual(gradient.call_count, 1)
        self.assertTrue(evidence["input"]["nonzero"])
        self.assertTrue(all(value["nonzero"] for value in evidence["candidates"].values()))

    def test_atlas_mapping_rejects_fractional_and_unknown_labels(self):
        template = torch.arange(117).float() * 2
        target = torch.tensor([[[0., 4., 232.]]])
        self.assertEqual(atlas_class_indices(target, template).tolist(), [[0, 2, 116]])
        for invalid in (1., .5, 234.):
            target[0, 0, 0] = invalid
            with self.assertRaises(ValueError):
                atlas_class_indices(target, template)

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
