import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from unibrain_agent.full_composition import fingerprint, migrate_state, verify_previous, build_composed_model
from unibrain_agent.replacement_training import classification_gradients


class FullCompositionTests(unittest.TestCase):
    def test_composition_verifies_extraction_and_registration_gradients(self):
        class Registration(nn.Module):
            def __init__(self, method, affine):
                super().__init__()
                self.affine = affine
                self.core = nn.Linear(3, 3)

            def forward(self, image):
                return self.core(self.affine(image))

        host, _ = self.models()
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter.py"
            adapter.write_text("def adapt(core, image, if_train):\n    return core(image)\n")
            with patch("unibrain_agent.full_model.build_full_model", return_value=(host, {}, {"unchanged_modules": {"reg_net": 1}})), \
                    patch("unibrain_agent.full_candidates.FullRegistration", Registration), \
                    patch("unibrain_agent.full_candidates.DenseHost", side_effect=lambda value: value), \
                    patch("unibrain_agent.sources.verify", return_value=[]):
                model, _, architecture = build_composed_model(Path(directory), "voxelmorph", adapter)
        self.assertEqual(architecture["candidate_modules"], ["ext_net", "reg_net"])
        image = torch.randn(2, 3)
        extracted = model.ext_net(image)
        loss = model.classifier(model.reg_net(extracted)).square().mean()
        evidence = classification_gradients(model, loss, "voxelmorph", True)
        self.assertEqual(set(evidence), {"ext_net", "reg_net"})
        self.assertTrue(all(record["required"] and record["nonzero"] for record in evidence.values()))
        detached_loss = model.classifier(model.reg_net(model.ext_net(image).detach())).square().mean()
        detached = classification_gradients(model, detached_loss, "voxelmorph", True)
        self.assertFalse(detached["ext_net"]["nonzero"])
        self.assertTrue(detached["reg_net"]["nonzero"])

    def models(self):
        previous = nn.Module()
        previous.reg_net = nn.Linear(3, 3)
        previous.ext_net = nn.Linear(3, 3)
        previous.classifier = nn.Linear(3, 1)
        target = nn.Module()
        target.reg_net = nn.Module()
        target.reg_net.affine = nn.Linear(3, 3)
        target.reg_net.core = nn.Linear(3, 3)
        target.ext_net = nn.Linear(3, 3)
        target.classifier = nn.Linear(3, 1)
        return previous, target

    def test_only_new_registration_core_is_missing(self):
        previous, target = self.models()
        new = target.reg_net.core.weight.detach().clone()
        result = migrate_state(target, previous.state_dict(), {"reg_net.": "reg_net.affine."}, ("reg_net.core.",))
        self.assertTrue(result["strict_load"])
        self.assertEqual(set(result["new_tensors"]), {"reg_net.core.weight", "reg_net.core.bias"})
        self.assertTrue(torch.equal(target.reg_net.affine.weight, previous.reg_net.weight))
        self.assertTrue(torch.equal(target.ext_net.weight, previous.ext_net.weight))
        self.assertTrue(torch.equal(target.classifier.weight, previous.classifier.weight))
        self.assertTrue(torch.equal(target.reg_net.core.weight, new))

    def test_missing_retained_tensor_is_rejected(self):
        previous, target = self.models()
        state = previous.state_dict()
        del state["classifier.bias"]
        with self.assertRaisesRegex(ValueError, "Missing retained"):
            migrate_state(target, state, {"reg_net.": "reg_net.affine."}, ("reg_net.core.",))

    def test_unexpected_tensor_is_rejected(self):
        previous, target = self.models()
        state = previous.state_dict()
        state["unrelated.weight"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            migrate_state(target, state, {"reg_net.": "reg_net.affine."}, ("reg_net.core.",))

    def test_nonfinite_state_is_rejected(self):
        previous, target = self.models()
        state = previous.state_dict()
        state["ext_net.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            migrate_state(target, state, {"reg_net.": "reg_net.affine."}, ("reg_net.core.",))

    def test_prior_training_hash_and_fold_are_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report = Path(directory) / "model.pt", Path(directory) / "report.json"
            torch.save({"weight": torch.ones(1)}, checkpoint)
            value = {"status": "complete", "method": "synthstrip", "steps": 2,
                     "config": {"dataset": "abide", "fold": 1}, "checkpoint_sha256": fingerprint(checkpoint)}
            report.write_text(json.dumps(value), encoding="utf-8")
            evidence = verify_previous(checkpoint, report, fingerprint(checkpoint), "abide", 1)
            self.assertEqual(evidence["prior_optimizer_steps"], 2)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_previous(checkpoint, report, "0" * 64)
            with self.assertRaisesRegex(ValueError, "same dataset"):
                verify_previous(checkpoint, report, fingerprint(checkpoint), "abide", 2)
            value["steps"] = 0
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "actually trained"):
                verify_previous(checkpoint, report, fingerprint(checkpoint))

    def test_sequential_report_cannot_override_zero_step_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report = Path(directory) / "model.pt", Path(directory) / "report.json"
            payload = {"identity": {"method": "synthstrip", "dataset": "abide", "fold": 1},
                       "progress": {"steps": 0}, "model": {"weight": torch.ones(1)}}
            torch.save(payload, checkpoint)
            value = {"status": "complete", "method": "synthstrip", "steps": 2,
                     "config": {"dataset": "abide", "fold": 1}}
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "actually trained checkpoint"):
                verify_previous(checkpoint, report, fingerprint(checkpoint), "abide", 1)
            payload["progress"]["steps"] = 1
            torch.save(payload, checkpoint)
            result = verify_previous(checkpoint, report, fingerprint(checkpoint), "abide", 1)
            self.assertEqual(result["prior_optimizer_steps"], 1)
            self.assertEqual(result["reported_optimizer_steps"], 2)
            payload["identity"]["fold"] = 2
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "identity contradicts"):
                verify_previous(checkpoint, report, fingerprint(checkpoint), "abide", 1)

    def test_raw_first_stage_requires_report_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report = Path(directory) / "model.pt", Path(directory) / "report.json"
            torch.save({"weight": torch.ones(1)}, checkpoint)
            report.write_text(json.dumps({"status": "complete", "method": "synthstrip", "steps": 2,
                                          "config": {"dataset": "abide", "fold": 1}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "report bound"):
                verify_previous(checkpoint, report, fingerprint(checkpoint), "abide", 1)

    def test_build_only_uses_training_adapter_and_connection_finalizer(self):
        from scripts.run_composition import main
        from unibrain_agent.candidate_integration import GeneratedCore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "registration.py"
            adapter.write_text("def adapt(core, moving, fixed):\n    return core(moving)\n")
            model = nn.Module()
            model.reg_net = nn.Module()
            model.reg_net.core = nn.Linear(2, 2)
            command = ["run_composition", "--source", str(root), "--data", str(root), "--output", str(root / "built"),
                       "--method", "voxelmorph", "--secondary-extraction", str(root / "extraction.py"),
                       "--adapter", str(adapter), "--connect-paths", "--cut-boundary", "registration",
                       "--repair-strategy", "straight_through", "--build-only"]
            with patch.object(sys, "argv", command), patch("scripts.run_composition.preflight", return_value={"status": "ready"}), \
                    patch("scripts.run_composition.build_composed_model", return_value=(model, {}, {})), \
                    patch("unibrain_agent.full_connectivity.connect_model", side_effect=lambda value, **kwargs: value) as connect:
                main()
            connect.assert_called_once_with(model, cut="registration", repair="straight_through", undo="none")
            self.assertIsInstance(model.reg_net.core, GeneratedCore)
            record = json.loads((root / "built/build_report.json").read_text())
            self.assertEqual(record["architecture"]["generated_adapter_sha256"], fingerprint(adapter))


if __name__ == "__main__":
    unittest.main()
