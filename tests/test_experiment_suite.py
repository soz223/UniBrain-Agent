import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from unibrain_agent.experiment_suite import (
    METHODS, TRAIN_METHODS, command_for, example_config, initial_weights, integration_config,
    load_config, plan, preflight, run_job, save, summarize,
)


class ExperimentSuiteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "config.json"
        self.config = example_config()
        self.config.update(seeds=[17, 29], repetitions=2)
        save(self.path, self.config)

    def load(self):
        save(self.path, self.config)
        return load_config(self.path)

    def test_plan_preserves_full_paper_scope(self):
        jobs = plan(self.load())
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        self.assertEqual({j["method"] for j in jobs if j["group"] == "integration"}, set(METHODS))
        training = [j for j in jobs if j["group"] == "training" and j["method"] in TRAIN_METHODS]
        self.assertEqual(len(training), 2 * 5 * 2 * 4 * 3)
        self.assertEqual({j["variant"] for j in training}, {"joint", "independent_official", "independent_finetuned"})
        self.assertIn("site", {j["group"] for j in jobs})
        self.assertIn("case_study", {j["group"] for j in jobs})

    def test_unconfigured_runs_are_blocked_not_simulated(self):
        config = self.load()
        job = next(j for j in plan(config) if j["group"] == "training" and j["method"] == "nnunet")
        report = preflight(config, job)
        self.assertFalse(report["ready"])
        self.assertTrue(any("Weights require" in reason for reason in report["reasons"]))
        self.assertEqual(run_job(config, job)["status"], "blocked")
        self.assertFalse((self.root / "runs").exists())

    def test_site_names_cannot_escape_job_directory(self):
        self.config["datasets"]["abide"]["sites"] = {"../external/site": "prepared"}
        jobs = [job for job in plan(self.load()) if job["group"] == "site"]
        self.assertTrue(jobs)
        self.assertTrue(all("/" not in job["id"] and "\\" not in job["id"] for job in jobs))

    def test_fold_pretraining_requires_matching_provenance(self):
        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture")
        self.config["weights"]["nnunet"] = {"path": "weights.pt", "provenance": "training_fold", "origin": "Unit test",
                                             "sha256": hashlib.sha256(b"fixture").hexdigest(), "dataset": "abide", "fold": 2}
        config = self.load()
        job = next(j for j in plan(config) if j["group"] == "training" and j["method"] == "nnunet" and j["fold"] == 1)
        with self.assertRaisesRegex(ValueError, "training fold"):
            initial_weights(config, job)

    def fixture_executor(self, job, result_hash=True):
        script = self.root / "executor.py"
        script.write_text("import json,sys\nfrom pathlib import Path\n"
                          "job=json.loads(Path(sys.argv[1]).read_text())\n"
                          f"result={{'status':'complete','job_hash':job['job_hash'] if {result_hash!r} else 'wrong','optimizer_updates':0,'metrics':{{'auc':0.7}}}}\n"
                          "Path(sys.argv[2]).write_text(json.dumps(result))\n", encoding="utf-8")
        self.config["executors"][job["id"]] = {"description": "Explicit test fixture, not model output",
                                               "argv": [sys.executable, str(script), "{job}", "{result}"],
                                               "smoke_argv": [sys.executable, str(script), "{job}", "{result}"]}

    def test_real_subprocess_roundtrip_and_overwrite_protection(self):
        job = next(j for j in plan(self.load()) if j["group"] == "systems")
        self.fixture_executor(job)
        config = self.load()
        report = run_job(config, job)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(summarize(config)["complete"], 1)
        with self.assertRaises(FileExistsError):
            run_job(config, job)

    def test_wrong_result_identity_is_rejected(self):
        job = next(j for j in plan(self.load()) if j["group"] == "systems")
        self.fixture_executor(job, False)
        result = run_job(self.load(), job)
        self.assertEqual(result["status"], "failed")
        self.assertIn("does not match", result["error"])

    def test_smoke_results_never_enter_scientific_summary(self):
        job = next(j for j in plan(self.load()) if j["group"] == "systems")
        self.fixture_executor(job)
        config = self.load()
        self.assertEqual(run_job(config, job, smoke_steps=2)["status"], "smoke_complete")
        report = summarize(config)
        self.assertEqual(report["complete"], 0)
        self.assertFalse(report["metrics"])

    def test_seed_repeats_are_averaged_within_fold(self):
        config = self.load()
        jobs = [j for j in plan(config) if j["group"] == "training" and j["method"] == "original"
                and j["variant"] == "original" and j["dataset"] == "abide" and j["fold"] <= 2]
        for job, metric in zip(jobs, (0.2, 0.4, 0.8, 1.0)):
            save(self.root / "runs/paper" / job["id"] / "execution.json",
                 {"status": "complete", "job": job, "config_hash": config["_hash"], "outcome": {"metrics": {"auc": metric}}})
        row = summarize(config)["metrics"][0]
        self.assertEqual(row["unit"], "fold_mean")
        self.assertEqual(row["n_runs"], 4)
        self.assertEqual(row["n_units"], 2)
        self.assertAlmostEqual(row["mean"], 0.6)

    def test_integration_config_uses_planned_seed_and_variant(self):
        case = self.root / "case.json"
        (self.root / "fold_data").mkdir()
        save(case, {"version": 1, "cases": [{"id": "synthstrip", "source_files": ["source.py"],
                                            "required_files": ["fold_data/train.npy"],
                                            "evaluator": {"command": ["{python}", "--data", "fold_data"]}}],
                    "providers": [{"provider": "codex"}], "variants": ["general"], "seeds": [99], "repetitions": 10})
        self.config["integration_configs"]["synthstrip:full"] = "case.json"
        config = self.load()
        job = next(j for j in plan(config) if j["group"] == "integration" and j["method"] == "synthstrip" and j["variant"] == "full")
        value = integration_config(config, job)
        self.assertEqual(value["seeds"], [job["seed"]])
        self.assertEqual(value["variants"], ["workflow"])
        self.assertEqual(value["repetitions"], 1)
        self.assertEqual(value["cases"][0]["required_files"], [str(self.root / "fold_data/train.npy")])
        self.assertEqual(value["cases"][0]["evaluator"]["command"][-1], str(self.root / "fold_data"))

    def test_training_commands_match_actual_cli(self):
        from unibrain_agent.replacement_training import parser

        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture")
        metadata = {"path": "weights.pt", "origin": "Unit test", "sha256": hashlib.sha256(b"fixture").hexdigest()}
        self.config["hosts"]["abide:1"] = metadata
        self.config["adapters"]["synthstrip"] = "adapter.py"
        self.config["plans"]["nnunet"] = "plans.json"
        for method in TRAIN_METHODS:
            self.config["weights"][method] = {**metadata, "provenance": "official"}
        config = self.load()
        jobs = [j for j in plan(config) if j["group"] in {"training", "ablation"} and j["fold"] == 1
                and j["seed"] == 17 and j["dataset"] == "abide" and j["variant"] not in {"stagewise", "simultaneous", "sequential"}]
        for job in jobs:
            command = command_for(config, job, self.root / "out", smoke_steps=2)
            args = parser().parse_args(command[2:])
            self.assertEqual(args.max_steps, 2)
            self.assertEqual(args.seed, job["seed"])
            self.assertEqual(args.method, job["method"])

    def test_composition_commands_match_actual_cli(self):
        from unibrain_agent.replacement_training import parser

        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture")
        prior_report = self.root / "prior.json"
        save(prior_report, {"status": "complete", "method": "synthstrip", "steps": 2})
        metadata = {"path": "weights.pt", "origin": "Unit test", "sha256": hashlib.sha256(b"fixture").hexdigest()}
        self.config["hosts"]["abide:1"] = metadata
        self.config["weights"]["voxelmorph"] = {**metadata, "provenance": "official"}
        self.config["adapters"]["synthstrip"] = "adapter.py"
        for seed in self.config["seeds"]:
            self.config["sequential_runs"][f"abide:1:{seed}"] = {"checkpoint": "weights.pt", "report": "prior.json", "sha256": metadata["sha256"]}
        config = self.load()
        jobs = [j for j in plan(config) if j["group"] == "composition"]
        for job in jobs:
            command = command_for(config, job, self.root / "out", smoke_steps=2)
            self.assertEqual(Path(command[1]).name, "run_composition.py")
            args = parser().parse_args(command[2:])
            self.assertEqual(args.composition_mode, job["variant"])
            self.assertEqual(args.max_steps, 2)
            self.assertEqual(args.method, "voxelmorph")
            self.assertEqual(args.secondary_extraction, self.root / "adapter.py")
            if job["variant"] == "sequential":
                self.assertIsNone(args.host_checkpoint)
                self.assertEqual(args.previous_checkpoint, weights)
                self.assertEqual(args.previous_report, prior_report)
                self.assertEqual(args.previous_sha256, metadata["sha256"])
            else:
                self.assertEqual(args.host_checkpoint, weights)
                self.assertIsNone(args.previous_checkpoint)


if __name__ == "__main__":
    unittest.main()
