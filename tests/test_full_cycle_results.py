import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from unibrain_agent.research_cycle import validate_full_training, validate_best_checkpoint, fingerprint
from unibrain_agent.data import digest
from unibrain_agent.replacement_training import metrics_from_cases


class FullCycleResultTests(unittest.TestCase):
    def fixture(self, root):
        array = np.array([("case_one", 0), ("case_two", 1)], dtype=[("id", "U12"), ("label", "i8")])
        np.save(root / "adhd_test_sample.npy", array)
        args = SimpleNamespace(epochs=2, dataset="abide", fold=1, evaluate_test=True, data=root,
                               output=root, seed=17, selection_metric="classification_loss")
        cases = [{"id": identifier, "label": label, "prediction": label, "probability": .2 if label == 0 else .8,
                  **{name: .7 for name in ("brain_dice", "tissue_dice", "atlas_dice", "registration_dice", "reference_loss", "classification_loss")}}
                 for identifier, label in (("case_one", 0), ("case_two", 1))]
        metrics = metrics_from_cases(cases)
        audit = {"data_hashes": {"train": "a", "val": "b", "test": "c"}}
        adapter = root / "accepted_adapter.py"
        adapter.write_text("accepted fixture adapter")
        training = root / "training"
        training.mkdir()
        identity = {"method": "synthstrip", "setting": "joint", "seed": 17, "dataset": "abide", "fold": 1,
                    "selection_metric": "classification_loss", "adapter_hash": fingerprint(adapter), "data_hashes": audit["data_hashes"]}
        progress = {"epoch": 1, "best_epoch": 1, "best_value": .7, "best_loss": .7, "selection_metric": "classification_loss",
                    "steps": 3, "pretrain_steps": 0}
        torch.save({"checkpoint_kind": "evaluation_weights_only", "model": {"weight": torch.ones(1)},
                    "identity": identity, "progress": progress}, training / "best.pt")
        record = {"status": "complete", "steps": 6, "audit": audit, "method": "synthstrip", "seed": 17,
                  "config": {"epochs": 2, "dataset": "abide", "fold": 1, "max_steps": None, "eval_limit": None, "evaluate_test": True,
                             "selection_metric": "classification_loss", "adapter": str(adapter), "output": str(training)},
                  "selection": {"metric": "classification_loss", "direction": "minimize", "scope": "validation_only", "best_epoch": 1, "best_value": .7},
                  "best_validation": dict(metrics),
                  "history": [{"epoch": epoch, "validation": dict(metrics)} for epoch in (1, 2)],
                  "test": {"sample_count": 2, "full_split_count": 2, "metrics": metrics, "cases": cases, "checkpoint_sha256": digest(training / "best.pt")}}
        return args, {"train": 3, "val": 2, "test": 2}, audit, record

    def test_actual_test_subjects_and_finite_metrics_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            args, splits, audit, record = self.fixture(Path(directory))
            validate_full_training(record, args, splits, audit)

    def test_other_subject_ids_and_wrong_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, splits, audit, record = self.fixture(Path(directory))
            for key, value in (("id", "unrelated"), ("label", 1)):
                changed = copy.deepcopy(record)
                changed["test"]["cases"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_full_training(changed, args, splits, audit)

    def test_missing_or_nonfinite_evaluation_cannot_pass(self):
        mutations = [lambda value: value["history"][0].update(validation={}),
                     lambda value: value["history"][0]["validation"].update(reference_loss=float("nan")),
                     lambda value: value["test"].update(metrics={}),
                     lambda value: value["test"]["cases"][0].pop("probability"),
                     lambda value: value["test"]["cases"][0].update(probability=1.2),
                     lambda value: value["test"]["cases"][0].update(brain_dice=float("inf"))]
        with tempfile.TemporaryDirectory() as directory:
            args, splits, audit, record = self.fixture(Path(directory))
            for mutate in mutations:
                changed = copy.deepcopy(record)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validate_full_training(changed, args, splits, audit)

    def test_undefined_auc_and_rate_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, splits, audit, record = self.fixture(root)
            array = np.load(root / "adhd_test_sample.npy")
            array["label"] = 0
            np.save(root / "adhd_test_sample.npy", array)
            for case in record["test"]["cases"]:
                case["label"] = 0
            record["test"]["metrics"] = metrics_from_cases(record["test"]["cases"])
            record["history"][0]["validation"].update(auc=None, sensitivity=None)
            record["best_validation"].update(auc=None, sensitivity=None)
            validate_full_training(record, args, splits, audit)

    def test_aggregate_metrics_must_match_subject_records(self):
        with tempfile.TemporaryDirectory() as directory:
            args, splits, audit, record = self.fixture(Path(directory))
            for name in ("accuracy", "balanced_accuracy", "macro_f1", "auc", "sensitivity", "specificity",
                         "brain_dice", "tissue_dice", "atlas_dice", "registration_dice", "reference_loss", "classification_loss"):
                changed = copy.deepcopy(record)
                changed["test"]["metrics"][name] += .01
                with self.assertRaisesRegex(ValueError, "aggregate metric"):
                    validate_full_training(changed, args, splits, audit)

    def test_best_checkpoint_identity_and_selected_epoch_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            args, splits, audit, record = self.fixture(Path(directory))
            path = args.output / "training/best.pt"
            original = torch.load(path, weights_only=True)
            for key, value in (("fold", 2), ("method", "voxelmorph"), ("adapter_hash", "wrong"), ("data_hashes", {}), ("seed", 4)):
                changed = copy.deepcopy(original)
                changed["identity"][key] = value
                torch.save(changed, path)
                record["test"]["checkpoint_sha256"] = digest(path)
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    validate_best_checkpoint(record, args, splits, audit)
            changed = copy.deepcopy(original)
            changed["progress"]["epoch"] = 2
            torch.save(changed, path)
            record["test"]["checkpoint_sha256"] = digest(path)
            with self.assertRaisesRegex(ValueError, "selected validation epoch"):
                validate_best_checkpoint(record, args, splits, audit)

    def test_selection_uses_first_minimum_of_declared_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            args, splits, audit, record = self.fixture(Path(directory))
            record["history"][1]["validation"]["reference_loss"] = .01
            evidence = validate_best_checkpoint(record, args, splits, audit)
            self.assertEqual(evidence["selected_epoch"], 1)
            record["selection"].update(best_epoch=2, best_value=.01)
            with self.assertRaisesRegex(ValueError, "validation history"):
                validate_best_checkpoint(record, args, splits, audit)


if __name__ == "__main__":
    unittest.main()
