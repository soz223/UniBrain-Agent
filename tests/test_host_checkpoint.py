import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from unibrain_agent.full_model import VerifiedHostCheckpoint, build_full_model, host_file_hash, load_host_checkpoint
from unibrain_agent.full_candidates import build_candidate_model
from unibrain_agent.replacement_training import parser, preflight


class HostCheckpointTests(unittest.TestCase):
    def fixture(self, root, fold=1, mapping="model", raw=False):
        path = root / "host.pt"
        state = {"weight": torch.tensor([1., 2.])}
        payload = state if raw else {"identity": {"method": "original", "dataset": "abide", "fold": fold},
                                     "progress": {"steps": 4}, mapping: state}
        torch.save(payload, path)
        return VerifiedHostCheckpoint(path, "abide", 1, host_file_hash(path)), payload

    def report(self, root, spec, **changes):
        path = root / "training_report.json"
        record = {"method": "original", "dataset": "abide", "fold": 1, "status": "complete",
                  "optimizer_updates": 4, "checkpoint_sha256": spec.sha256, **changes}
        path.write_text(json.dumps(record), encoding="utf-8")
        return VerifiedHostCheckpoint(spec.path, spec.dataset, spec.fold, spec.sha256, path)

    def test_matching_bundle_and_historical_mapping_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            for mapping in ("model", "model_state_dict"):
                folder = Path(directory) / mapping
                folder.mkdir()
                spec, _ = self.fixture(folder, mapping=mapping)
                state, evidence = load_host_checkpoint(spec)
                self.assertTrue(evidence["internal_identity_verified"])
                self.assertEqual(evidence["state_mapping"], mapping)
                torch.testing.assert_close(state["weight"], torch.tensor([1., 2.]))

    def test_external_report_cannot_override_wrong_internal_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = self.fixture(root, fold=2)
            spec = self.report(root, spec)
            with self.assertRaisesRegex(ValueError, "this dataset and fold"):
                load_host_checkpoint(spec)

    def test_raw_original_state_requires_explicit_bound_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = self.fixture(root, raw=True)
            with self.assertRaisesRegex(ValueError, "explicit hash bound training report"):
                load_host_checkpoint(spec)
            valid = self.report(root, spec)
            _, evidence = load_host_checkpoint(valid)
            self.assertFalse(evidence["internal_identity_verified"])
            self.assertEqual(evidence["updates"], 4)
            for changes in ({"fold": 2}, {"method": None}, {"optimizer_updates": 0}, {"checkpoint_sha256": "wrong"}):
                with self.assertRaises(ValueError):
                    load_host_checkpoint(self.report(root, spec, **changes))

    def test_partial_internal_metadata_cannot_fall_back_to_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, payload = self.fixture(root)
            del payload["identity"]["dataset"]
            torch.save(payload, spec.path)
            spec = VerifiedHostCheckpoint(spec.path, "abide", 1, host_file_hash(spec.path))
            with self.assertRaisesRegex(ValueError, "this dataset and fold"):
                load_host_checkpoint(self.report(root, spec))

    def test_legacy_fold_metadata_cannot_be_overridden_by_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "legacy.pt"
            torch.save({"model_state_dict": {"weight": torch.ones(1)}, "dataset": "abide", "fold": 2}, path)
            spec = VerifiedHostCheckpoint(path, "abide", 1, host_file_hash(path))
            with self.assertRaisesRegex(ValueError, "Legacy host metadata contradicts"):
                load_host_checkpoint(self.report(root, spec))

    def test_file_mutation_during_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, payload = self.fixture(Path(directory))
            def changed(*args, **kwargs):
                spec.path.write_bytes(b"changed checkpoint")
                return payload
            with patch("torch.load", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "changed during inspection"):
                    load_host_checkpoint(spec)

    def test_unscoped_builder_paths_fail_before_model_allocation(self):
        for builder in (lambda: build_full_model("missing", host_checkpoint=Path("host.pt")),
                        lambda: build_candidate_model("missing", "voxelmorph", host_checkpoint=Path("host.pt"))):
            with self.assertRaisesRegex(ValueError, "VerifiedHostCheckpoint"):
                builder()

    def test_joint_and_official_controls_check_internal_fold_in_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = self.fixture(root, fold=2)
            weights = root / "candidate.pt"
            torch.save({}, weights)
            for setting in ("joint", "independent_official"):
                command = ["--source", "missing", "--data", "missing", "--output", str(root / "run"),
                           "--method", "voxelmorph", "--setting", setting, "--dataset", "abide", "--fold", "1",
                           "--host-checkpoint", str(spec.path), "--host-dataset", "abide", "--host-fold", "1",
                           "--host-origin", "training record", "--host-sha256", spec.sha256]
                if setting == "independent_official":
                    command += ["--initialization", "official", "--weights", str(weights), "--weights-origin", "official",
                                "--weights-sha256", host_file_hash(weights)]
                with self.assertRaisesRegex(ValueError, "this dataset and fold"):
                    preflight(parser().parse_args(command))


if __name__ == "__main__":
    unittest.main()
