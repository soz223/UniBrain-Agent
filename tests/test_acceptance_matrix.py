import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from scripts import export_acceptance_subset as subset
from scripts import run_acceptance_matrix as matrix
from scripts.export_cohort_fold import export_fold
from unibrain_agent.cohort_provenance import validate_cohort


class AcceptanceExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cohort = self.root / "cohort"
        self.output = self.root / "subset"
        indices = self.cohort / "rounds/round1"
        indices.mkdir(parents=True)
        dtype = [("id", "S12"), ("label", "i1"), (subset.VOLUMES[0], "f4", (96, 96, 96)),
                 *((name, "u1", (96, 96, 96)) for name in subset.VOLUMES[1:])]
        master = np.zeros(6, dtype=dtype)
        master["id"] = [f"subject{i}".encode() for i in range(6)]
        master["label"] = np.arange(6) % 2
        np.save(self.cohort / "unibrain_master.npy", master)
        for position, split in enumerate(subset.SPLITS):
            np.save(indices / f"{split}_indices.npy", np.array([2 * position, 2 * position + 1]))
        templates = self.cohort / "templates/original_96"
        templates.mkdir(parents=True)
        for name in subset.TEMPLATES.values():
            nib.save(nib.Nifti1Image(np.zeros((96, 96, 96), dtype=np.uint8), np.eye(4)), templates / name)

    def export(self):
        return subset.export(self.cohort, self.output, dataset="abide")

    def test_export_preserves_original_hashes_and_is_portably_auditable(self):
        original = subset.sha(self.cohort / "unibrain_master.npy")
        report = self.export()
        audit = matrix.subset_audit(self.output, "abide", 1)
        self.assertEqual(report["master_sha256"], original)
        self.assertEqual(subset.sha(self.cohort / "unibrain_master.npy"), original)
        self.assertEqual(audit["master_sha256"], original)
        self.assertEqual(len(audit["data_hashes"]), 6)
        self.assertEqual(report["splits"]["test"]["ids"], ["subject4", "subject5"])
        with self.assertRaisesRegex(ValueError, "new output"):
            self.export()

    def test_changed_original_membership_is_rejected(self):
        report = self.export()
        report["splits"]["train"]["master_indices"][0] = 4
        (self.output / "cohort.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fold membership"):
            matrix.subset_audit(self.output, "abide", 1)

    def test_template_modification_and_wrong_dataset_are_rejected(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "source dataset"):
            matrix.subset_audit(self.output, "adhd", 1)
        np.save(self.output / matrix.TEMPLATES[0], np.ones((96, 96, 96), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "template checksum"):
            matrix.subset_audit(self.output, "abide", 1)

    def test_overlapping_original_indices_fail_before_export(self):
        np.save(self.cohort / "rounds/round1/val_indices.npy", [0, 3])
        with self.assertRaisesRegex(ValueError, "partition"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_nonfinite_subject_retains_failed_manifest(self):
        master = np.load(self.cohort / "unibrain_master.npy", mmap_mode="r+", allow_pickle=False)
        master["raw_img_np"][0, 0, 0, 0] = float("nan")
        master.flush()
        del master
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.export()
        report = json.loads((self.output / "cohort.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        with self.assertRaisesRegex(ValueError, "complete version 2"):
            matrix.subset_audit(self.output, "abide", 1)

    def test_complete_fold_preserves_every_index_and_original_order(self):
        for position, split in enumerate(subset.SPLITS):
            np.save(self.cohort / f"rounds/round1/{split}_indices.npy", [2 * position + 1, 2 * position])
        report = export_fold(self.cohort, 1, self.output, full=True, dataset="abide")
        self.assertEqual(report["scope"], "full_fold")
        self.assertEqual(report["splits"]["train"]["master_indices"], [1, 0])
        self.assertEqual(report["splits"]["train"]["ids"], ["subject1", "subject0"])
        names = [f"adhd_{split}_sample.npy" for split in subset.SPLITS] + list(subset.TEMPLATES)
        hashes = {name: subset.sha(self.output / name) for name in names}
        checked = validate_cohort(self.output, "abide", 1, hashes, required_scope="full_fold", allow_missing=False)
        self.assertTrue(checked["array_hashes_verified"])
        with self.assertRaisesRegex(ValueError, "scope"):
            matrix.subset_audit(self.output, "abide", 1)

    def test_prepared_commands_match_current_training_cli(self):
        from unibrain_agent.replacement_training import parser
        self.export()
        other = self.root / "adhd"
        subset.export(self.cohort, other, dataset="adhd")
        config = {"source": "source", "datasets": {"abide": "subset", "adhd": "adhd"},
                  "methods": ["voxelmorph"], "coordination": "taskforce"}
        updates, jobs = matrix.prepare(config, self.root / "config.json", self.root / "runs", "acceptance")
        self.assertEqual(updates, 2)
        for job in jobs:
            args = parser().parse_args(job["command"][2:])
            self.assertTrue(args.verify_gradients)
            self.assertEqual(args.coordination, "taskforce")
            self.assertEqual(args.max_steps, 2)
            self.assertEqual(args.checkpoint_mode, "weights")
            self.assertEqual(job["gradient_protocol_version"], 3)
            self.assertEqual(job["status"], "planned")

    def test_selected_jobs_and_trajectory_checkpoint_policy(self):
        config = {"source": "source", "datasets": {"abide": "subset"}, "methods": ["voxelmorph", "lapirn"],
                  "trajectory_methods": ["voxelmorph"], "job_subset": ["abide:voxelmorph"]}
        with patch.object(matrix, "subset_audit", return_value={}):
            _, jobs = matrix.prepare(config, self.root / "config.json", self.root / "runs", "acceptance")
            self.assertEqual([(job["dataset"], job["method"]) for job in jobs], [("abide", "voxelmorph")])
            _, jobs = matrix.prepare(config, self.root / "config.json", self.root / "trajectory", "trajectory")
            self.assertEqual(jobs[0]["checkpoint_mode"], "resume")
            for invalid in ([], ["abide:unknown"], ["abide:voxelmorph"] * 2, "abide:voxelmorph"):
                with self.assertRaisesRegex(ValueError, "job_subset"):
                    matrix.prepare({**config, "job_subset": invalid}, self.root / "config.json", self.root / "runs", "acceptance")
            with self.assertRaisesRegex(ValueError, "trajectories require resume"):
                matrix.prepare({**config, "checkpoint_mode": "weights"}, self.root / "config.json", self.root / "runs", "trajectory")


class AcceptanceExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "matrix"
        self.jobs = [{"dataset": dataset, "method": "voxelmorph", "setting": "joint", "seed": 17, "fold": 1,
                      "command": [sys.executable, "fixture.py", dataset], "output": str(self.output / f"{dataset}_voxelmorph"),
                      "data_audit": {"data_hashes": {"test.npy": "fixture"}}, "status": "planned", "reasons": []}
                     for dataset in ("abide", "adhd")]
        self.calls = []
        disk = patch.object(matrix, "disk_check", return_value={"ready": True, "free_bytes": 100 * 1024 ** 3,
                                                                 "required_bytes": 18 * 1024 ** 3, "reserve_gib": 10,
                                                                 "checkpoint_budget_gib": 8})
        disk.start()
        self.addCleanup(disk.stop)

    def runner(self, command, log, timeout):
        self.calls.append(command)
        if "--preflight" in command:
            log.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            return 0
        job = next(job for job in self.jobs if job["dataset"] == command[-1])
        folder = Path(job["output"])
        folder.mkdir()
        report = {"status": "budget_exhausted", "method": job["method"], "setting": job["setting"], "seed": 17,
                  "steps": 2, "config": {"dataset": job["dataset"], "fold": 1, "max_steps": 2},
                  "audit": job["data_audit"]}
        (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
        events = [{"step": step, "total_loss": 1.0, "sampled_parameter_updates": {"reg_net": {"changed": True}}}
                  for step in (1, 2)]
        (folder / "updates.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
        (folder / "last.pt").write_bytes(b"CPU protocol fixture; not a trained checkpoint")
        log.write_text(json.dumps({"status": "budget_exhausted"}), encoding="utf-8")
        return 0

    def execute(self, runner=None):
        return matrix.execute(self.jobs, self.output, {"scope": "CPU protocol fixture"}, 2, runner=runner or self.runner)

    def test_budget_exhausted_is_valid_only_with_complete_bounded_evidence(self):
        result = self.execute()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(job["optimizer_updates"] == 2 and job["worker_status"] == "budget_exhausted" for job in result["jobs"]))
        checked = matrix.validate_result(self.jobs[0], Path(self.jobs[0]["output"]) / "report.json", 2)
        self.assertEqual(checked["status"], "passed")
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_zero_exit_with_failed_status_does_not_pass(self):
        def runner(command, log, timeout):
            code = self.runner(command, log, timeout)
            if "--preflight" not in command:
                log.write_text(json.dumps({"status": "failed", "error": "Synthetic failure"}), encoding="utf-8")
            return code
        result = self.execute(runner)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all(job["status"] == "failed" for job in result["jobs"]))

    def test_wrong_data_or_short_step_count_is_rejected(self):
        def runner(command, log, timeout):
            code = self.runner(command, log, timeout)
            if "--preflight" not in command:
                job = next(job for job in self.jobs if job["dataset"] == command[-1])
                path = Path(job["output"]) / "report.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                if job["dataset"] == "abide":
                    report["audit"]["data_hashes"] = {"other": "changed"}
                else:
                    report["steps"] = 1
                path.write_text(json.dumps(report), encoding="utf-8")
            return code
        result = self.execute(runner)
        self.assertTrue(all(job["status"] == "failed" for job in result["jobs"]))

    def test_gpu_busy_halts_all_later_jobs(self):
        def runner(command, log, timeout):
            if "--preflight" in command:
                return self.runner(command, log, timeout)
            self.calls.append(command)
            log.write_text(json.dumps({"status": "failed", "error": "Selected GPU is occupied or has less than 4500 MiB free"}), encoding="utf-8")
            return 1
        result = self.execute(runner)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(self.jobs[1]["status"], "not_run_after_halt")
        self.assertEqual(len(self.calls), 2)

    def test_disk_shortage_blocks_before_any_child(self):
        with patch.object(matrix, "disk_check", return_value={"ready": False, "free_bytes": 1, "required_bytes": 18 * 1024 ** 3}):
            result = self.execute()
        self.assertEqual(result["status"], "blocked_disk_space")
        self.assertFalse(self.calls)
        self.assertTrue(all(job["status"] == "blocked_disk_space" for job in self.jobs))

    def test_timeout_halts_all_later_jobs(self):
        def runner(command, log, timeout):
            self.calls.append(command)
            raise subprocess.TimeoutExpired(command, timeout)
        result = self.execute(runner)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(self.jobs[0]["status"], "timeout")
        self.assertEqual(self.jobs[1]["status"], "not_run_after_halt")
        self.assertEqual(len(self.calls), 1)

    def test_preflight_cannot_create_or_reuse_a_child_directory(self):
        def runner(command, log, timeout):
            Path(self.jobs[0]["output"]).mkdir()
            log.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            return 0
        result = self.execute(runner)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(self.jobs[1]["status"], "not_run_after_halt")

    def test_real_child_timeout_returns_promptly(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            matrix.run_command([sys.executable, "-c", "import time; time.sleep(5)"], self.root / "timeout.log", .05)

    def test_two_policy_decisions_do_not_claim_policy_learning(self):
        for job in self.jobs:
            job.update(coordination="taskforce", phase="acceptance")
        def runner(command, log, timeout):
            result = self.runner(command, log, timeout)
            if "--preflight" not in command:
                job = next(job for job in self.jobs if job["dataset"] == command[-1])
                folder = Path(job["output"])
                report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                report["config"]["coordination"] = "taskforce"
                counters = {"policy_updates": 0, "actor_optimizer_steps": 0, "critic_optimizer_steps": 0}
                report["coordination"] = {"mode": "taskforce", **counters}
                (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
                events = [json.loads(line) for line in (folder / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
                for event in events:
                    event["gradient_policy"] = {"weights": [.25] * 4, **counters}
                (folder / "updates.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            return result
        result = self.execute(runner)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(not job["coordination"]["policy_learning_observed"] for job in result["jobs"]))

    def gradient_runner(self, command, log, timeout, auxiliary=False):
        result = self.runner(command, log, timeout)
        if "--preflight" in command:
            return result
        job = next(job for job in self.jobs if job["dataset"] == command[-1])
        folder = Path(job["output"])
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        report["gradient_verification"] = {"enabled": True, "verified_steps": 2,
                                           "diagnosis_reaches_all_candidates": not auxiliary}
        (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
        events = [json.loads(line) for line in (folder / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
        module = "seg_net" if auxiliary else "reg_net"
        good = {"finite": True, "nonzero": True}
        for event in events:
            event["sampled_parameter_updates"] = {module: {"changed": True}}
            classification = {**good, "required": True}
            if auxiliary:
                classification = {"required": False, "expected_absent": True, "finite": True, "nonzero": False}
            elif job["dataset"] == "adhd":
                classification["nonzero"] = False
            event["gradient_verification"] = {"input_required": True, "total_to_input": good,
                                               "total_to_candidates": {module: good},
                                               "classification_to_candidates": {module: classification}}
        (folder / "updates.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
        return result

    def test_disconnected_classification_cannot_pass_on_parameter_updates_alone(self):
        for job in self.jobs:
            job["verify_gradients"] = True
        result = self.execute(self.gradient_runner)
        self.assertEqual(result["jobs"][0]["status"], "passed")
        self.assertEqual(result["jobs"][1]["status"], "failed")
        self.assertIn("Classification gradient", result["jobs"][1]["error"])

    def test_auxiliary_segmentation_requires_total_gradient_and_declares_classification_boundary(self):
        for job in self.jobs:
            job.update(method="nnunet", verify_gradients=True)
        result = self.execute(lambda command, log, timeout: self.gradient_runner(command, log, timeout, auxiliary=True))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(job["gradient_verification"]["diagnosis_reaches_all_candidates"] is False for job in self.jobs))

    def test_protocol_two_detects_detached_input_and_legacy_revalidation_is_explicit(self):
        for job in self.jobs:
            job.update(verify_gradients=True, gradient_protocol_version=2, checkpoint_mode="weights")
        def runner(command, log, timeout):
            result = self.gradient_runner(command, log, timeout)
            if "--preflight" in command:
                return result
            job = next(job for job in self.jobs if job["dataset"] == command[-1])
            folder = Path(job["output"])
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            report["config"]["checkpoint_mode"] = "weights"
            report["checkpoint_policy"] = {"mode": "weights", "resume_allowed": False}
            report["gradient_verification"]["protocol_version"] = 2
            (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            events = [json.loads(line) for line in (folder / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
            for event in events:
                verification = event["gradient_verification"]
                verification.update(protocol_version=2, classification_to_input={"required": True, "finite": True,
                                                                                  "nonzero": job["dataset"] == "abide"})
                verification["classification_to_candidates"]["reg_net"]["nonzero"] = True
            (folder / "updates.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            return result
        result = self.execute(runner)
        self.assertTrue(result["jobs"][0]["strict_current_gradient_verified"])
        self.assertEqual(result["jobs"][1]["status"], "failed")
        self.assertIn("input image", result["jobs"][1]["error"])
        job = self.jobs[0]
        path = Path(job["output"]) / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["gradient_verification"].pop("protocol_version")
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "protocol 2"):
            matrix.validate_result(job, path, 2)
        job.pop("gradient_protocol_version")
        evidence = matrix.validate_result(job, path, 2)
        self.assertFalse(evidence["strict_current_gradient_verified"])
        self.assertEqual(evidence["gradient_boundary_scope"], "legacy_boundary")


class ProtocolThreeAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sequence = 0

    def source_fixture(self, directory="reg_unibrain"):
        manifest = json.loads((matrix.ROOT / "catalog/sources.json").read_text(encoding="utf-8"))
        repository = next(row for row in manifest["repositories"] if row["directory"] == directory)
        checksum = next(row["sha256"] for row in repository["files"] if row["path"] == "model.py")
        return {"class_name": "encoder", "repository": repository["repository"], "commit": repository["commit"],
                "source_file": "model.py", "source_sha256": checksum, "head_object_verified": True,
                "forward_code_sha256": "b" * 64, "constructor_code_sha256": "c" * 64}

    def fixture(self, warmup=True, method="synthstrip"):
        self.sequence += 1
        folder = self.root / str(self.sequence) / ("abide_" + method)
        folder.mkdir(parents=True)
        job = {"dataset": "abide", "method": method, "fold": 1, "seed": 17, "setting": "joint", "status": "passed",
               "verify_gradients": True, "gradient_protocol_version": 3,
               "data_audit": {"scope": "acceptance_subset_not_full_cohort", "data_hashes": {"train": "fixture"}}}
        good = {"tensor_count": 2, "observed_tensors": 2, "nonzero_elements": 3, "finite": True, "nonzero": True, "l1": .2}
        proof = {"head_path": "reg_net.affine.fc_loc.2", "weight_shape": [12, 128], "bias_shape": [12],
                 "zero_weight": True, "identity_bias": True, "head_trainable": True,
                 "head_total_gradient": copy.deepcopy(good), "head_weight_gradient": {**good, "tensor_count": 1, "observed_tensors": 1},
                 "extraction_tensor_count": 2, "extraction_parameter_names": ["weight", "bias"],
                 "encoder_source": self.source_fixture(), "weight_nonzero_after_update": True}
        unused = {"status": "not_needed", "module": None, "first_step": None, "confirmed_step": None, "reason": None, "initialization": None}
        events = []
        for step in (1, 2):
            state = {"status": "pending" if step == 1 else "confirmed", "module": "ext_net", "first_step": 1,
                     "confirmed_step": None if step == 1 else 2, "reason": "zero_initialized_affine_head", "initialization": copy.deepcopy(proof)} if warmup else copy.deepcopy(unused)
            classification = {name: {**good, "required": True, "parameter_names": ["weight", "bias"]} for name in matrix.TARGETS[method]}
            if warmup and step == 1:
                classification["ext_net"].update(nonzero=False, nonzero_elements=0, l1=0.)
            events.append({"step": step, "total_loss": 1.0,
                           "sampled_parameter_updates": {name: {"changed": True} for name in matrix.TARGETS[method]},
                           "gradient_verification": {"protocol_version": 3, "warmup": state, "input_required": True,
                                                     "total_to_input": copy.deepcopy(good), "classification_to_input": {**good, "required": True},
                                                     "total_to_candidates": {name: copy.deepcopy(good) for name in matrix.TARGETS[method]},
                                                     "classification_to_candidates": classification}})
        report = {"status": "budget_exhausted", "method": method, "setting": "joint", "seed": 17, "steps": 2,
                  "config": {"dataset": "abide", "fold": 1, "max_steps": 2},
                  "audit": {**job["data_audit"], "splits": {"train": 2, "val": 2, "test": 2}},
                  "architecture": {"parameters": 10}, "train_seconds": .1,
                  "gradient_verification": {"protocol_version": 3, "enabled": True, "verified_steps": 2,
                                            "diagnosis_reaches_all_candidates": True, "warmup": copy.deepcopy(events[-1]["gradient_verification"]["warmup"])}}
        path = folder / "report.json"
        (folder / "last.pt").write_bytes(b"CPU protocol fixture only")
        return job, path, report, events

    def write(self, path, report, events):
        path.write_text(json.dumps(report), encoding="utf-8")
        (path.parent / "updates.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")

    def test_known_initial_zero_recovers_after_one_update_and_keeps_actual_protocol(self):
        for method in ("synthstrip", "ernet"):
            job, path, report, events = self.fixture(method=method)
            self.write(path, report, events)
            job["gradient_protocol_version"] = 2
            result = matrix.validate_result(job, path, 2)
            self.assertEqual(result["gradient_protocol_version"], 3)
            self.assertEqual(result["gradient_boundary_scope"], "protocol_3")
            self.assertTrue(result["strict_current_gradient_verified"])
            self.assertFalse(result["initial_nonzero_gradient_verified"])
            self.assertEqual(result["gradient_warmup"]["optimizer_updates"], 1)

    def test_protocol_three_without_warmup_remains_initially_nonzero(self):
        job, path, report, events = self.fixture(warmup=False)
        self.write(path, report, events)
        result = matrix.validate_result(job, path, 2)
        self.assertFalse(result["gradient_warmup"]["used"])
        self.assertTrue(result["initial_nonzero_gradient_verified"])

    def test_trainer_generated_cpu_warmup_trace_passes_independent_validation(self):
        import torch
        from unibrain_agent import replacement_training as training
        torch.manual_seed(17)
        model = torch.nn.Module()
        model.ext_net = torch.nn.Linear(2, 128)
        model.reg_net = torch.nn.Module()
        model.reg_net.affine = torch.nn.Module()
        head = torch.nn.Linear(128, 12)
        model.reg_net.affine.fc_loc = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), head)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.eye(3, 4).flatten())
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        state, trace = training.empty_gradient_warmup(), []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            image = torch.ones(1, 2, requires_grad=True)
            features = model.ext_net(image)
            diagnosis = head(features).sum() + .01 * image.sum()
            total = diagnosis + features.square().mean()
            paths = training.classification_gradient_paths(model, diagnosis, "synthstrip", True, image)
            total.backward()
            audit = {"protocol_version": 3, "classification_to_candidates": paths["candidates"],
                     "classification_to_input": paths["input"], "input_required": True,
                     "total_to_input": training.gradient_statistics([image.grad]),
                     "total_to_candidates": training.total_candidate_gradients(model, "synthstrip")}
            with patch.object(training, "affine_encoder_evidence", return_value=self.source_fixture()):
                training.check_gradient_paths(model, audit, step, 2, state)
            before = model.ext_net.weight.detach().clone()
            optimizer.step()
            training.record_warmup_update(model, state, audit)
            trace.append({"step": step + 1, "total_loss": float(total.detach()), "gradient_verification": audit,
                          "sampled_parameter_updates": {"ext_net": {"changed": not torch.equal(before, model.ext_net.weight)}}})
        training.require_confirmed_warmup(state)
        job, path, report, _ = self.fixture()
        report["gradient_verification"]["warmup"] = copy.deepcopy(state)
        self.write(path, report, trace)
        measured = matrix.validate_result(job, path, 2)
        self.assertEqual(measured["gradient_warmup"]["state"]["confirmed_step"], 2)
        self.assertEqual(trace[0]["gradient_verification"]["classification_to_candidates"]["ext_net"]["l1"], 0)
        self.assertGreater(trace[1]["gradient_verification"]["classification_to_candidates"]["ext_net"]["l1"], 0)

    def test_warmup_cannot_hide_detach_nonfinite_missing_update_or_dead_path(self):
        for defect in ("detached", "partial", "nonfinite", "total_zero", "unknown_head", "wrong_shape", "wrong_bias",
                       "no_head_update", "no_head_gradient", "no_ext_update", "still_zero", "input_zero", "repeated_pending", "other_module_zero"):
            with self.subTest(defect=defect):
                job, path, report, events = self.fixture(method="ernet")
                first = events[0]["gradient_verification"]
                ext = first["classification_to_candidates"]["ext_net"]
                proof = first["warmup"]["initialization"]
                if defect in ("detached", "partial"):
                    ext["observed_tensors"] = 0 if defect == "detached" else 1
                elif defect == "nonfinite":
                    ext["finite"] = False
                elif defect == "total_zero":
                    first["total_to_candidates"]["ext_net"]["nonzero"] = False
                elif defect == "unknown_head":
                    proof["head_path"] = "some_other_layer"
                elif defect == "wrong_shape":
                    proof["weight_shape"] = [12, 16]
                elif defect == "wrong_bias":
                    proof["identity_bias"] = False
                elif defect == "no_head_update":
                    proof["weight_nonzero_after_update"] = False
                elif defect == "no_head_gradient":
                    proof["head_total_gradient"]["observed_tensors"] = 0
                elif defect == "no_ext_update":
                    events[0]["sampled_parameter_updates"]["ext_net"]["changed"] = False
                elif defect == "still_zero":
                    events[1]["gradient_verification"]["classification_to_candidates"]["ext_net"]["nonzero"] = False
                elif defect == "input_zero":
                    first["classification_to_input"]["nonzero"] = False
                elif defect == "repeated_pending":
                    events[1]["gradient_verification"]["warmup"] = copy.deepcopy(first["warmup"])
                else:
                    first["classification_to_candidates"]["reg_net"]["nonzero"] = False
                self.write(path, report, events)
                with self.assertRaises(ValueError):
                    matrix.validate_result(job, path, 2)

    def test_pending_single_update_cannot_be_completed(self):
        job, path, report, events = self.fixture()
        report["steps"] = report["config"]["max_steps"] = report["gradient_verification"]["verified_steps"] = 1
        report["gradient_verification"]["warmup"] = copy.deepcopy(events[0]["gradient_verification"]["warmup"])
        self.write(path, report, events[:1])
        with self.assertRaisesRegex(ValueError, "pending"):
            matrix.validate_result(job, path, 1)

    def test_protocol_three_rejects_inconsistent_numeric_evidence(self):
        for defect in ("false_nonzero", "head_nan", "head_inf", "nested_path", "head_count", "weight_missing",
                       "weight_zero", "weight_count", "weight_exceeds_head", "ext_count_drift", "ext_partial",
                       "input_inconsistent", "head_fractional_count", "negative_norm", "boolean_count", "wrong_source", "parameter_names"):
            with self.subTest(defect=defect):
                job, path, report, events = self.fixture()
                second = events[1]["gradient_verification"]
                if defect == "false_nonzero":
                    second["classification_to_candidates"]["ext_net"].update(observed_tensors=0, nonzero_elements=0, l1=0.)
                elif defect in ("ext_count_drift", "ext_partial"):
                    second["classification_to_candidates"]["ext_net"]["observed_tensors"] = 1
                    if defect == "ext_count_drift":
                        second["classification_to_candidates"]["ext_net"]["tensor_count"] = 1
                elif defect == "input_inconsistent":
                    second["classification_to_input"].update(observed_tensors=0, nonzero_elements=0, l1=0.)
                elif defect == "parameter_names":
                    second["classification_to_candidates"]["ext_net"]["parameter_names"] = ["other_weight", "other_bias"]
                else:
                    for event in events:
                        proof = event["gradient_verification"]["warmup"]["initialization"]
                        if defect in ("head_nan", "head_inf"):
                            proof["head_total_gradient"]["l1"] = float("nan" if defect == "head_nan" else "inf")
                        elif defect == "nested_path":
                            proof["head_path"] = "reg_net.core.core.fc_loc.2"
                        elif defect == "head_count":
                            proof["head_total_gradient"].update(tensor_count=1, observed_tensors=1)
                        elif defect == "weight_missing":
                            proof.pop("head_weight_gradient")
                        elif defect == "weight_zero":
                            proof["head_weight_gradient"].update(nonzero=False, nonzero_elements=0, l1=0.)
                        elif defect == "weight_count":
                            proof["head_weight_gradient"].update(tensor_count=2, observed_tensors=2)
                        elif defect == "weight_exceeds_head":
                            proof["head_weight_gradient"]["l1"] = proof["head_total_gradient"]["l1"] + 1
                        elif defect == "head_fractional_count":
                            proof["head_total_gradient"]["nonzero_elements"] = .5
                        elif defect == "negative_norm":
                            proof["head_total_gradient"]["l1"] = -1.
                        elif defect == "wrong_source":
                            proof["encoder_source"]["source_sha256"] = "d" * 64
                        else:
                            proof["head_total_gradient"]["tensor_count"] = True
                report["gradient_verification"]["warmup"] = copy.deepcopy(second["warmup"])
                self.write(path, report, events)
                with self.assertRaises(ValueError):
                    matrix.validate_result(job, path, 2)

    def test_nonfinite_json_and_inconsistent_protocol_two_statistics_are_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "Nonfinite"):
                matrix.evidence_json('{"value": ' + token + '}')
        job, path, report, events = self.fixture(warmup=False)
        job["gradient_protocol_version"] = report["gradient_verification"]["protocol_version"] = 2
        for event in events:
            event["gradient_verification"]["protocol_version"] = 2
        self.write(path, report, events)
        self.assertEqual(matrix.validate_result(job, path, 2)["gradient_boundary_scope"], "protocol_2")
        events[1]["gradient_verification"]["classification_to_candidates"]["ext_net"].update(observed_tensors=0, nonzero_elements=0, l1=0.)
        self.write(path, report, events)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            matrix.validate_result(job, path, 2)

    def test_protocol_two_cannot_satisfy_a_protocol_three_job(self):
        job, path, report, events = self.fixture(warmup=False)
        report["gradient_verification"]["protocol_version"] = 2
        for event in events:
            event["gradient_verification"]["protocol_version"] = 2
        self.write(path, report, events)
        with self.assertRaisesRegex(ValueError, "protocol 3"):
            matrix.validate_result(job, path, 2)


if __name__ == "__main__":
    unittest.main()
