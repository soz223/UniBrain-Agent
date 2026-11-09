import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from unibrain_agent.catalog import DEFAULTS, ADAPTER_DEFAULTS, original_selection
from unibrain_agent.data import FIELDS, OriginalDataset
from unibrain_agent.native import original_environment
from unibrain_agent.paths import PROJECT
from unibrain_agent.runtime import device_for, move_model


class PortabilityTests(unittest.TestCase):
    def test_original_defaults_have_no_replacement_or_soft_connection(self):
        self.assertTrue(original_selection(DEFAULTS))
        for stage in ("extraction", "registration", "segmentation", "parcellation", "graph", "classification"):
            self.assertEqual(DEFAULTS[stage], "unibrain")
        self.assertFalse(original_selection(ADAPTER_DEFAULTS))

    def test_paths_resolve_from_config_directory_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            config = folder / "runtime.json"
            config.write_text(json.dumps({"data": "cohort", "runs": "results", "builds": "prepared"}))
            env = {key: value for key, value in os.environ.items() if not key.startswith("UNIBRAIN_")}
            env.update(UNIBRAIN_CONFIG=str(config), PYTHONPATH=str(PROJECT))
            output = subprocess.check_output([sys.executable, "-c", "from unibrain_agent.paths import DATA, RUNS, BUILDS; print(DATA); print(RUNS); print(BUILDS)"], cwd=folder, env=env, text=True)
            self.assertEqual([Path(line) for line in output.splitlines()], [folder / "cohort", folder / "results", folder / "prepared"])

    def test_original_dataset_reads_one_sample_and_preserves_source(self):
        dtype = [("id", "i8"), ("label", "i8")] + [(name, "f4", (96, 96, 96)) for name in FIELDS[2:]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npy"
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(2,))
            array["id"] = [1, 2]
            array["label"] = [0, 1]
            array.flush()
            data = OriginalDataset(path)
            self.assertIsInstance(data.array, np.memmap)
            row = data[1]
            self.assertEqual(row[0], 2)
            row[2].fill(9)
            self.assertEqual(float(data.array[1]["raw_img_np"].max()), 0)
            del data, array

    def test_gpu_selection_respects_scheduler_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5"}):
            self.assertEqual(original_environment({"device": "cuda:1"})["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(original_environment({"device": "auto"})["CUDA_VISIBLE_DEVICES"], "3,5")
            with self.assertRaises(ValueError):
                original_environment({"device": "cuda:2"})
        with self.assertRaises(ValueError):
            original_environment({"device": "cpu"})
        with self.assertRaises(ValueError):
            original_environment({"device": "cuda:-1"})
        with patch("torch.cuda.is_available", return_value=False), self.assertRaises(ValueError):
            device_for("cuda:0")

    def test_constant_tensors_follow_model_device(self):
        model = torch.nn.Linear(2, 2)
        model.grid = torch.ones(2)
        move_model(model, "meta")
        self.assertEqual(model.weight.device.type, "meta")
        self.assertEqual(model.grid.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
