import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from unibrain_agent.full_composition import fingerprint, migrate_state, verify_previous


class FullCompositionTests(unittest.TestCase):
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
            checkpoint.write_bytes(b"Test checkpoint")
            value = {"status": "complete", "method": "synthstrip", "steps": 2,
                     "config": {"dataset": "abide", "fold": 1}}
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


if __name__ == "__main__":
    unittest.main()
