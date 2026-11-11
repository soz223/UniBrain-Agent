import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from unibrain_agent.data import FIELDS, digest
from unibrain_agent.nnunet_prepare import export_training, parser, run_planner, validate_export, validate_plan_provenance


class NnunetPreparationTests(unittest.TestCase):
    def test_training_plan_rejects_cross_fold_and_changed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = Path(directory) / "nnUNetPlans.json"
            plans.write_text("{}")
            metadata = {"status": "ready", "training_only": True, "dataset": "abide", "fold": 1,
                        "source_train_sha256": "a" * 64, "plans_sha256": digest(plans),
                        "planner_environment": {"nnunetv2": "test", "planner_source_sha256": "b" * 64}}
            with self.assertRaisesRegex(ValueError, "sidecar"):
                validate_plan_provenance(plans, "a" * 64, "abide", 1)
            plans.with_suffix(".provenance.json").write_text(json.dumps(metadata))
            self.assertEqual(validate_plan_provenance(plans, "a" * 64, "abide", 1)["status"], "ready")
            with self.assertRaisesRegex(ValueError, "dataset and fold"):
                validate_plan_provenance(plans, "a" * 64, "abide", 2)
            with self.assertRaisesRegex(ValueError, "array hash"):
                validate_plan_provenance(plans, "c" * 64, "abide", 1)
            plans.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_plan_provenance(plans, "a" * 64, "abide", 1)

    def fixture(self, root):
        data = root / "data"
        data.mkdir()
        dtype = [(FIELDS[0], "U20"), (FIELDS[1], "i4")] + [(name, "f4", (96, 96, 96)) for name in FIELDS[2:]]
        array = np.zeros(1, dtype=dtype)
        array["id"] = "training_subject"
        array["raw_img_np"][0, 10, 20, 30] = 9.
        array["raw_img_gm_mask_np"][0, :, :, :24] = 1
        array["raw_img_gm_mask_np"][0, :, :, 24:48] = 2
        array["raw_img_gm_mask_np"][0, :, :, 48:72] = 3
        np.save(data / "adhd_train_sample.npy", array)
        for split in ("val", "test"):
            (data / f"adhd_{split}_sample.npy").write_bytes(b"Must never be read by the planner")
        args = parser().parse_args(["export", "--data", str(data), "--output", str(root / "export"),
                                    "--dataset", "abide", "--fold", "1", "--dataset-id", "501",
                                    "--coordinates", "normalized_voxel"])
        return args, array

    def test_train_only_export_preserves_voxels_and_marks_unknown_units(self):
        with tempfile.TemporaryDirectory() as directory:
            args, original = self.fixture(Path(directory))
            report = export_training(args)
            self.assertFalse(report["test_read"])
            self.assertEqual(report["cases"][0]["subject_id"], "training_subject")
            image = nib.load(args.output / "nnUNet_raw" / report["dataset_name"] / "imagesTr/case000000_0000.nii.gz")
            self.assertEqual(image.header.get_xyzt_units()[0], "unknown")
            np.testing.assert_array_equal(np.asarray(image.dataobj).transpose(2, 1, 0), original["raw_img_np"][0])
            self.assertEqual(validate_export(args.output)["status"], "exported")
            (args.output / "nnUNet_raw" / report["dataset_name"] / "unexpected.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Untracked"):
                validate_export(args.output)

    def test_physical_spacing_requires_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            args.coordinates = "physical_mm"
            args.spacing = [1., 1., 1.]
            with self.assertRaisesRegex(ValueError, "verified"):
                export_training(args)

    def test_dependency_failure_preserves_export_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            export_training(args)
            plan = parser().parse_args(["plan", "--output", str(args.output)])
            with patch("unibrain_agent.nnunet_prepare.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "missing nnunetv2")):
                result = run_planner(plan)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(validate_export(args.output)["status"], "exported")
            self.assertIn("missing nnunetv2", (args.output / result["log_directory"] / "dependency_check.log").read_text())

    def test_real_planner_command_and_incompatible_plan_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            exported = export_training(args)
            plan = parser().parse_args(["plan", "--output", str(args.output)])
            calls = []

            def invoke(command, **kwargs):
                calls.append(command)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                if "-c" in command:
                    return subprocess.CompletedProcess(command, 0, json.dumps({"nnunetv2": "fixture"}), "")
                destination = args.output / "nnUNet_preprocessed" / exported["dataset_name"]
                destination.mkdir(parents=True)
                payload = {"configurations": {"3d_fullres": {"patch_size": [80, 96, 96],
                           "architecture": {"network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet"}}}}
                (destination / "nnUNetPlans.json").write_text(json.dumps(payload))
                return subprocess.CompletedProcess(command, 0)

            with patch("unibrain_agent.nnunet_prepare.subprocess.run", side_effect=invoke):
                result = run_planner(plan)
            self.assertEqual(result["status"], "incompatible")
            self.assertEqual(result["patch_size"], [80, 96, 96])
            self.assertIn("--no_pp", calls[1])
            self.assertIn("ExperimentPlanner", calls[1])
            self.assertTrue((args.output / result["plans"]).is_file())


if __name__ == "__main__":
    unittest.main()
