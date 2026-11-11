import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from scripts.run_upstream import audit_results, training_values


class UpstreamAuditTests(unittest.TestCase):
    def test_optimizer_wrappers_do_not_hide_training_scalars(self):
        ext_loss = torch.tensor(1.0)
        sim_loss_train = torch.tensor(2.0)
        seg_loss_train = torch.tensor(3.0)
        classifier_loss_train = torch.tensor(4.0)
        loss_train = ext_loss + .1 * sim_loss_train + seg_loss_train + classifier_loss_train
        parameter = torch.nn.Parameter(torch.ones(1))
        observed = []
        original_step = torch.optim.Adam.step
        def optimizer_wrapper(optimizer, *args, **kwargs):
            observed.append(training_values())
            return original_step(optimizer, *args, **kwargs)
        with patch.object(torch.optim.Adam, "step", optimizer_wrapper):
            optimizer = torch.optim.Adam([parameter])
            parameter.square().sum().backward()
            optimizer.step()
        total, losses = observed[0]
        self.assertAlmostEqual(total, 8.2, places=5)
        self.assertEqual(losses, {"extraction": 1.0, "registration": 2.0, "segmentation": 3.0, "classification": 4.0})

    def test_completion_needs_all_updates_validation_and_finite_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "dataset").mkdir()
            (source / "result/loss_log").mkdir(parents=True)
            (source / "result/model").mkdir()
            for split in ("train", "val", "test"):
                np.save(source / "dataset" / f"adhd_{split}_sample.npy", np.zeros(2))
            (source / "result/loss_log/sample_val.txt").write_text("Epoch[0/2]\nEpoch[1/2]\n")
            update = {"finite_gradient_norms": True,
                      "modules_with_observed_updates": ["ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred"]}
            record = {"source_unchanged": True, "status": "completed", "epochs": 2, "updates": [update] * 4}
            self.assertFalse(audit_results(source, record)["passed"])
            checkpoint = source / "result/model/sample.pth"
            torch.save({"weight": torch.ones(2)}, checkpoint)
            self.assertTrue(audit_results(source, record)["passed"])
            compact = {**record, "updates": [update], "update_count": 4,
                       "step_checks": {"finite_gradient_norms": True, "active_modules_updated": True,
                                       "policy_applied_every_step": True}}
            self.assertTrue(audit_results(source, compact)["passed"])
            self.assertFalse(audit_results(source, {**compact, "update_count": 3})["passed"])
            self.assertFalse(audit_results(source, {**record, "updates": [update] * 3})["passed"])
            self.assertFalse(audit_results(source, {**record, "source_unchanged": False})["passed"])
            torch.save({"weight": torch.tensor([float("nan")])}, checkpoint)
            self.assertFalse(audit_results(source, record)["passed"])


if __name__ == "__main__":
    unittest.main()
