import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_upstream import audit_results


class UpstreamAuditTests(unittest.TestCase):
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
            self.assertFalse(audit_results(source, {**record, "updates": [update] * 3})["passed"])
            self.assertFalse(audit_results(source, {**record, "source_unchanged": False})["passed"])
            torch.save({"weight": torch.tensor([float("nan")])}, checkpoint)
            self.assertFalse(audit_results(source, record)["passed"])


if __name__ == "__main__":
    unittest.main()
