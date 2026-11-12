import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unibrain_agent.experiment_suite import command_for, cycle_options, example_config, nnunet_plan, preflight


class FoldPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {**example_config(), "_base": str(self.root), "source": "source"}
        (self.root / "source").mkdir()
        (self.root / "source/model.py").touch()
        self.config["adapters"]["nnunet"] = "adapter.py"
        (self.root / "adapter.py").touch()
        for fold in (1, 2):
            folder = self.root / f"fold{fold}"
            folder.mkdir()
            for split in ("train", "val", "test"):
                (folder / f"adhd_{split}_sample.npy").touch()
            self.config["datasets"]["abide"]["folds"][str(fold)] = str(folder)
            plan = self.root / f"plan{fold}.json"
            plan.write_text(json.dumps({"fixture_fold": fold}))
            provenance = {"status": "ready", "training_only": True, "dataset": "abide", "fold": fold,
                          "source_train_sha256": str(fold) * 64, "plans_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                          "planner_environment": {"nnunetv2": "2.6.2", "planner_source_sha256": "a" * 64}}
            plan.with_suffix(".provenance.json").write_text(json.dumps(provenance))
        self.config["plans"]["nnunet"] = {"folds": {"abide:1": "plan1.json", "abide:2": "plan2.json"}}

    def job(self, fold):
        return {"id": f"fixture{fold}", "group": "training", "method": "nnunet", "variant": "random_initialization",
                "dataset": "abide", "fold": fold, "seed": 17, "epochs": 20}

    def test_each_fold_resolves_its_own_plan_in_training_and_research(self):
        self.config["research_cycle"]["task"] = "Select a tissue model"
        for fold in (1, 2):
            job = self.job(fold)
            for kind in (job, {**job, "group": "systems", "variant": "ours"}):
                command = command_for(self.config, kind, self.root / "output")
                self.assertEqual(Path(command[command.index("--plans") + 1]), self.root / f"plan{fold}.json")

    def test_wrong_fold_and_modified_plan_are_blocked_before_launch(self):
        job = self.job(2)
        cohort = {"declared_data_hashes": {"adhd_train_sample.npy": "2" * 64}}
        with patch("unibrain_agent.experiment_suite.full_fold_audit", return_value=cohort):
            self.assertTrue(preflight(self.config, job)["ready"])
            self.config["plans"]["nnunet"]["folds"]["abide:2"] = "plan1.json"
            rejected = preflight(self.config, job)
            self.assertFalse(rejected["ready"])
            self.assertTrue(any("dataset and fold" in reason for reason in rejected["reasons"]))
            self.config["plans"]["nnunet"]["folds"]["abide:2"] = "plan2.json"
            (self.root / "plan2.json").write_text("changed")
            self.assertFalse(preflight(self.config, job)["ready"])

    def test_single_plan_remains_supported_and_missing_fold_does_not_fall_back(self):
        self.config["plans"]["nnunet"] = "plan1.json"
        self.assertEqual(nnunet_plan(self.config, self.job(1)), self.root / "plan1.json")
        self.config["plans"]["nnunet"] = {"folds": {"abide:1": "plan1.json"}}
        with self.assertRaisesRegex(ValueError, "dataset and fold"):
            nnunet_plan(self.config, self.job(2))
        self.config["research_cycle"].update(task="Select a classifier", stage="classification")
        command = command_for(self.config, {**self.job(2), "group": "systems", "variant": "ours"}, self.root / "output")
        self.assertNotIn("--plans", command)

    def test_new_only_requires_and_passes_the_shared_trained_host(self):
        job = {**self.job(1), "method": "synthstrip", "variant": "new_only", "group": "ablation"}
        self.config["adapters"]["synthstrip"] = "adapter.py"
        weights = self.root / "weights.pt"
        weights.write_bytes(b"CPU fixture weights")
        metadata = {"path": "weights.pt", "origin": "CPU fixture", "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        self.config["weights"]["synthstrip"] = {**metadata, "provenance": "official"}
        with patch("unibrain_agent.experiment_suite.full_fold_audit", return_value={}):
            self.assertFalse(preflight(self.config, job)["ready"])
            self.config["hosts"]["abide:1"] = metadata
            self.assertTrue(preflight(self.config, job)["ready"])
        command = command_for(self.config, job, self.root / "output")
        self.assertEqual(command[command.index("--host-checkpoint") + 1], str(weights))
        self.assertEqual(command[command.index("--host-fold") + 1], "1")
        self.assertEqual(command[command.index("--trainable") + 1], "new")

    def test_research_training_timeout_is_explicitly_forwarded(self):
        self.config["research_cycle"].update(task="Select a classifier", training_timeout_seconds=172800)
        job = {**self.job(1), "group": "systems", "variant": "ours"}
        command = command_for(self.config, job, self.root / "output")
        self.assertEqual(command[command.index("--training-timeout-seconds") + 1], "172800")

    def test_research_checkpoint_modes_have_explicit_formal_and_smoke_defaults(self):
        self.config["research_cycle"].update(task="Select a classifier")
        job = {**self.job(1), "group": "systems", "variant": "ours"}
        for explicit in (None, "weights", "resume"):
            if explicit:
                self.config["research_cycle"]["checkpoint_mode"] = explicit
            for smoke, default in ((None, "resume"), (2, "weights")):
                command = command_for(self.config, job, self.root / "output", smoke)
                self.assertEqual(command[command.index("--checkpoint-mode") + 1], explicit or default)
        self.config["research_cycle"]["checkpoint_mode"] = "partial"
        with self.assertRaisesRegex(ValueError, "checkpoint mode"):
            cycle_options(self.config)


if __name__ == "__main__":
    unittest.main()
