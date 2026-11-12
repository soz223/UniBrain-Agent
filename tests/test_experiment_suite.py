import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unibrain_agent.experiment_suite import (
    METHODS, TRAIN_METHODS, command_for, example_config, initial_weights, integration_config, import_integration_index,
    load_config, plan, preflight, run_job, save, summarize,
)


class ExperimentSuiteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
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

    def cycle_fixture(self):
        import numpy as np
        self.config["source"] = "source"
        self.config["epochs"] = 2
        self.config["datasets"]["abide"]["folds"]["1"] = "data"
        self.config["research_cycle"]["task"] = "CPU protocol fixture task"
        (self.root / "source").mkdir()
        (self.root / "source/model.py").touch()
        (self.root / "data").mkdir()
        np.save(self.root / "data/adhd_test_sample.npy", np.array([('fixtureA', 0), ('fixtureB', 1)], dtype=[('id', 'U12'), ('label', 'i4')]))
        config = self.load()
        job = next(job for job in plan(config) if job["group"] == "systems" and job["variant"] == "ours")
        audit = {"manifest_sha256": "m" * 64, "data_hashes": {"adhd_train_sample.npy": "a" * 64},
                 "split_counts": {"train": 2, "val": 2, "test": 2}, "scope": "full_fold"}
        return config, job, audit

    def cycle_runner(self, command, log, timeout, cwd):
        import torch
        from unibrain_agent.research_cycle import parser
        args = parser().parse_args(command[2:])
        args.output.mkdir()
        training_dir = args.output / "training"
        training_dir.mkdir()
        adapter = args.output / "accepted_adapter.py"
        adapter.write_text("# CPU protocol fixture", encoding="utf-8")
        adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
        (training_dir / "best.pt").write_bytes(b"CPU fixture; not model weights")
        (training_dir / "last.pt").write_bytes(b"CPU fixture; not model weights")
        (training_dir / "updates.jsonl").write_text("{}\n", encoding="utf-8")
        steps = 2 * args.epochs if args.full_training else args.train_steps
        metrics = {name: .7 for name in ('accuracy', 'balanced_accuracy', 'macro_f1', 'auc', 'sensitivity', 'specificity',
                                        'brain_dice', 'tissue_dice', 'atlas_dice', 'registration_dice', 'reference_loss', 'classification_loss')}
        training = {"status": "complete" if args.full_training else "budget_exhausted", "method": "brainnetcnn",
                    "setting": "joint", "seed": args.seed, "steps": steps,
                    "config": {"dataset": args.dataset, "fold": args.fold, "source": str(args.source), "data": str(args.data),
                               "adapter": str(adapter), "coordination": "none", "checkpoint_mode": args.checkpoint_mode,
                               "connect_paths": True, "verify_gradients": True, "epochs": args.epochs,
                               "selection_metric": args.selection_metric, "output": str(training_dir),
                               "max_steps": None if args.full_training else args.train_steps,
                               "eval_limit": None if args.full_training else args.eval_limit, "evaluate_test": args.evaluate_test},
                    "architecture": {"generated_adapter_active_in_training": True, "generated_adapter_sha256": adapter_hash},
                    "audit": {"data_hashes": {"adhd_train_sample.npy": "a" * 64}},
                    "gradient_verification": {"enabled": True, "protocol_version": 2, "verified_steps": steps,
                                              "diagnosis_reaches_all_candidates": True,
                                              "last_step": {"classification_to_input": {"required": True, "finite": True, "nonzero": True}}},
                    "coordination": {"mode": "none", "policy_updates": 0}, "checkpoint_policy": {"mode": args.checkpoint_mode},
                    "selection": {"metric": args.selection_metric, "direction": "minimize", "scope": "validation_only", "best_epoch": 1, "best_value": .7},
                    "best_validation": metrics,
                    "history": [{"epoch": epoch, "validation": metrics} for epoch in range(1, args.epochs + 1)]}
        if args.full_training:
            identity = {"method": "brainnetcnn", "setting": "joint", "seed": args.seed, "dataset": args.dataset, "fold": args.fold,
                        "selection_metric": args.selection_metric, "adapter_hash": adapter_hash, "data_hashes": training["audit"]["data_hashes"]}
            progress = {"epoch": 1, "best_epoch": 1, "best_value": .7, "best_loss": .7, "selection_metric": args.selection_metric,
                        "steps": 2, "pretrain_steps": 0}
            torch.save({"model": {"fixture": torch.ones(1)}, "identity": identity, "progress": progress,
                        "checkpoint_kind": "evaluation_weights_only" if args.checkpoint_mode == "weights" else "evaluation"}, training_dir / "best.pt")
            cases = [{"id": name, "label": label, "prediction": label, "probability": .2 if label == 0 else .8,
                      **{key: .7 for key in ('brain_dice', 'tissue_dice', 'atlas_dice', 'registration_dice', 'reference_loss', 'classification_loss')}}
                     for name, label in (('fixtureA', 0), ('fixtureB', 1))]
            training["test"] = {"sample_count": 2, "full_split_count": 2, "cases": cases,
                                "checkpoint_sha256": hashlib.sha256((training_dir / "best.pt").read_bytes()).hexdigest(),
                                "metrics": {**metrics, **{key: 1. for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'auc', 'sensitivity', 'specificity')}}}
        save(training_dir / "report.json", training)
        evaluation = {"status": "passed", "runs": True, "optimizer_updates": 1,
                      "checks": {name: True for name in ("contract", "gradient", "spatial", "fidelity", "updates")},
                      "data": {"sha256": "a" * 64}}
        outcome = {"status": "complete", "phase": "complete", "method": "brainnetcnn", "seed": args.seed,
                   "provider": {"provider": args.provider}, "task": args.task,
                   "accepted_adapter": {"file": adapter.name, "sha256": adapter_hash},
                   "accepted_adapter_unchanged_after_training": True,
                   "actual_updates": {"acceptance": 1, "training": steps},
                   "integration": {"status": "passed", "final_valid": True,
                                   "calls": [{"adapter_sha256": adapter_hash, "evaluation": evaluation}]},
                   "training_protocol": {"full_training": args.full_training, "evaluate_test": args.evaluate_test},
                   "budgets": {"training_epochs": args.epochs}, "training": training}
        save(args.output / "report.json", outcome)
        log.write_text("CPU protocol fixture", encoding="utf-8")
        return 0

    def test_native_ours_commands_and_full_result_hashes(self):
        from unibrain_agent.research_cycle import parser
        config, job, audit = self.cycle_fixture()
        for smoke in (None, 2):
            args = parser().parse_args(command_for(config, job, self.root / "out", smoke)[2:])
            self.assertEqual(args.full_training, smoke is None)
            self.assertEqual(args.evaluate_test, smoke is None)
            self.assertEqual(args.checkpoint_mode, "resume" if smoke is None else "weights")
        with patch("unibrain_agent.experiment_suite.full_fold_audit", return_value=audit), \
                patch("unibrain_agent.experiment_suite.run_logged", side_effect=self.cycle_runner):
            result = run_job(config, job)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["selected_method"], "brainnetcnn")
        self.assertEqual(next(row for row in summarize(config)["metrics"] if row["metric"] == "auc")["mean"], 1.)
        retained = self.root / "runs/paper" / job["id"] / "payload/accepted_adapter.py"
        retained.write_text("# changed after execution", encoding="utf-8")
        summarized = summarize(config)
        self.assertFalse(summarized["metrics"])
        self.assertIn("changed_artifact", {row["status"] for row in summarized["rows"]})

    def test_native_ours_short_run_never_enters_full_experiment_metrics(self):
        config, job, audit = self.cycle_fixture()
        with patch("unibrain_agent.experiment_suite.full_fold_audit", return_value=audit), \
                patch("unibrain_agent.experiment_suite.run_logged", side_effect=self.cycle_runner):
            result = run_job(config, job, smoke_steps=2)
        self.assertEqual(result["status"], "smoke_complete", result)
        self.assertFalse(summarize(config)["metrics"])

    def test_native_ours_refuses_a_partial_held_out_test(self):
        config, job, audit = self.cycle_fixture()
        def runner(command, log, timeout, cwd):
            result = self.cycle_runner(command, log, timeout, cwd)
            folder = Path(command[command.index("--output") + 1])
            training_path = folder / "training/report.json"
            training = json.loads(training_path.read_text(encoding="utf-8"))
            training["test"]["cases"].pop()
            save(training_path, training)
            outcome = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            outcome["training"] = training
            save(folder / "report.json", outcome)
            return result
        with patch("unibrain_agent.experiment_suite.full_fold_audit", return_value=audit), \
                patch("unibrain_agent.experiment_suite.run_logged", side_effect=runner):
            result = run_job(config, job)
        self.assertEqual(result["status"], "failed")
        self.assertIn("held out test", result["error"])

    def test_native_cycle_resource_block_is_not_a_failed_scientific_result(self):
        config, job, audit = self.cycle_fixture()
        def runner(command, log, timeout, cwd):
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            save(folder / "report.json", {"status": "blocked", "phase": "auditing", "error": "Provider fixture unavailable"})
            return 1
        with patch("unibrain_agent.experiment_suite.full_fold_audit", return_value=audit), \
                patch("unibrain_agent.experiment_suite.run_logged", side_effect=runner):
            result = run_job(config, job)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(summarize(config)["metrics"])

    def test_site_commands_use_native_runner_and_explicit_split_identity(self):
        from unibrain_agent.replacement_training import parser
        self.config["datasets"]["abide"]["sites"] = {"site A": "site_data"}
        self.config["source"] = "source"
        self.config["adapters"].update({method: "adapter.py" for method in TRAIN_METHODS})
        (self.root / "source").mkdir()
        (self.root / "source/model.py").write_text("# CPU protocol fixture", encoding="utf-8")
        (self.root / "adapter.py").write_text("# CPU protocol fixture", encoding="utf-8")
        (self.root / "site_data").mkdir()
        for split in ("train", "val", "test"):
            (self.root / f"site_data/adhd_{split}_sample.npy").touch()
        config = self.load()
        jobs = [job for job in plan(config) if job["group"] == "site"]
        with patch("unibrain_agent.experiment_suite.site_audit", return_value={"scope": "site_holdout"}), \
                patch("unibrain_agent.experiment_suite.full_fold_audit", side_effect=AssertionError("Site is not a fold")):
            for job in jobs:
                ready = preflight(config, job)
                self.assertTrue(ready["ready"], ready)
                self.assertEqual(ready["executor"], "site_training")
                command = command_for(config, job, self.root / "out", smoke_steps=2)
                self.assertTrue(command[1].endswith("run_site_experiment.py"))
                args = parser().parse_args(command[2:])
                self.assertEqual(args.split_id, "site A")
                self.assertIsNone(args.fold)
                self.assertEqual(args.initialization, "random")
                self.assertEqual(args.max_steps, 2)
        with patch("unibrain_agent.experiment_suite.site_audit", side_effect=ValueError("Held out site leaked")):
            self.assertFalse(preflight(config, jobs[0])["ready"])

    def test_site_aggregate_uses_sites_as_units(self):
        self.config["datasets"]["abide"]["sites"] = {"site A": "a", "site B": "b"}
        config = self.load()
        jobs = [job for job in plan(config) if job["group"] == "site" and job["variant"] == "original"]
        for job, metric in zip(jobs, (.2, .4, .8, 1.)):
            save(self.root / "runs/paper" / job["id"] / "execution.json",
                 {"status": "complete", "job": job, "config_hash": config["_hash"], "outcome": {"metrics": {"auc": metric}}})
        row = summarize(config)["metrics"][0]
        self.assertEqual(row["unit"], "site_mean")
        self.assertEqual(row["n_units"], 2)
        self.assertAlmostEqual(row["mean"], .6)

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
            self.config["adapters"][method] = "adapter.py"
        config = self.load()
        jobs = [j for j in plan(config) if j["group"] in {"training", "ablation"} and j["fold"] == 1
                and j["seed"] == 17 and j["dataset"] == "abide" and j["variant"] not in {"simultaneous", "sequential"}]
        for job in jobs:
            command = command_for(config, job, self.root / "out", smoke_steps=2)
            args = parser().parse_args(command[2:])
            self.assertEqual(args.max_steps, 2)
            self.assertEqual(args.seed, job["seed"])
            self.assertEqual(args.method, job["method"])
            if job["method"] in TRAIN_METHODS:
                self.assertEqual(args.adapter, self.root / "adapter.py")

    def test_training_retains_adapter_bytes_and_requires_execution_confirmation(self):
        adapter = self.root / "adapter.py"
        adapter.write_text("# accepted CPU fixture", encoding="utf-8")
        weights = self.root / "weights.pt"
        weights.write_bytes(b"CPU checkpoint fixture")
        self.config["adapters"]["synthstrip"] = "adapter.py"
        self.config["weights"]["synthstrip"] = {"path": "weights.pt", "provenance": "official", "origin": "CPU fixture",
                                                  "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        for active in (True, False):
            self.config["output_root"] = "runs/" + str(active)
            config = self.load()
            job = next(job for job in plan(config) if job["group"] == "training" and job["method"] == "synthstrip" and job["variant"] == "joint")
            def runner(command, log, timeout, cwd):
                retained = Path(command[command.index("--adapter") + 1])
                self.assertNotEqual(retained, adapter)
                self.assertEqual(retained.read_bytes(), adapter.read_bytes())
                checksum = hashlib.sha256(retained.read_bytes()).hexdigest()
                adapter.write_text("# source changed after snapshot", encoding="utf-8")
                folder = Path(command[command.index("--output") + 1])
                folder.mkdir()
                save(folder / "report.json", {"status": "complete", "method": "synthstrip", "setting": "joint", "seed": job["seed"],
                                              "steps": 2, "config": {"dataset": job["dataset"], "fold": job["fold"]},
                                              "audit": {"data_hashes": {}}, "architecture": {"generated_adapter_active_in_training": active,
                                                                                              "generated_adapter_sha256": checksum}})
                return 0
            with patch("unibrain_agent.experiment_suite.preflight", return_value={"ready": True, "executor": "training"}), \
                    patch("unibrain_agent.experiment_suite.full_fold_audit", return_value={"manifest_sha256": "fixture"}), \
                    patch("unibrain_agent.experiment_suite.run_logged", side_effect=runner):
                result = run_job(config, job, smoke_steps=2)
            self.assertEqual(result["status"], "smoke_complete" if active else "failed")
            if not active:
                self.assertIn("unchanged accepted adapter", result["error"])

    def test_generated_index_links_all_candidates_without_agent_calls(self):
        directory = self.root / "generated"
        directory.mkdir()
        for method in METHODS:
            save(directory / (method + ".json"), {"version": 1, "cases": [{"id": method, "source_files": [], "required_files": ["data.npy"]}],
                                                   "providers": [{"provider": "codex"}]})
        save(directory / "index.json", {"configs": {method: method + ".json" for method in METHODS}})
        preview = import_integration_index(self.path, directory / "index.json", ["full", "codex_checks"])
        self.assertEqual(len(preview["mappings"]), 28)
        self.assertFalse(load_config(self.path)["integration_configs"])
        import_integration_index(self.path, directory / "index.json", ["full", "codex_checks"], apply=True)
        imported = load_config(self.path)
        value = integration_config(imported, {"method": "redwgan", "variant": "full", "seed": 17})
        self.assertEqual(value["cases"][0]["required_files"], [str(directory / "data.npy")])
        with self.assertRaisesRegex(ValueError, "Named agent baseline"):
            import_integration_index(self.path, directory / "index.json", ["claude"])

    def test_stagewise_uses_same_fold_host_and_explicit_frozen_setting(self):
        checkpoint = self.root / "host.pt"
        checkpoint.write_bytes(b"unit test")
        self.config["hosts"]["abide:1"] = {"path": "host.pt", "origin": "Unit test", "sha256": hashlib.sha256(b"unit test").hexdigest()}
        config = self.load()
        job = next(job for job in plan(config) if job["group"] == "training" and job["variant"] == "stagewise")
        command = command_for(config, job, self.root / "output", smoke_steps=2)
        self.assertEqual(command[command.index("--setting") + 1], "original_frozen_stages")
        self.assertEqual(command[command.index("--host-checkpoint") + 1], str(checkpoint))
        self.assertNotIn("--connect-paths", command)

    def test_composition_commands_match_actual_cli(self):
        from unibrain_agent.replacement_training import parser

        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture")
        prior_report = self.root / "prior.json"
        save(prior_report, {"status": "complete", "method": "synthstrip", "steps": 2})
        metadata = {"path": "weights.pt", "origin": "Unit test", "sha256": hashlib.sha256(b"fixture").hexdigest()}
        self.config["hosts"]["abide:1"] = metadata
        self.config["weights"]["voxelmorph"] = {**metadata, "provenance": "official"}
        self.config["adapters"].update({method: "adapter.py" for method in TRAIN_METHODS})
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
