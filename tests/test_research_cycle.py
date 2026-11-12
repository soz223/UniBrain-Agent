import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from unibrain_agent.integration_benchmark import CHECKS
from unibrain_agent.research_cycle import fingerprint, parser, run_cycle, training_command, validate_full_training, validate_training


class ResearchCycleTests(unittest.TestCase):
    def arguments(self, root):
        source, data = root / "source", root / "data"
        source.mkdir()
        data.mkdir()
        return parser().parse_args(["--source", str(source), "--data", str(data), "--output", str(root / "cycle"),
                                    "--provider", "codex", "--task", "Learn a structural brain graph", "--stage", "network_generation"])

    def candidates(self, blocked=False):
        case = {"id": "fbnetgen", "method": "fbnetgen", "stage": "network_generation",
                "interface": "Full ROI graph component", "knowledge": "Preserve gradients"}
        return {"fbnetgen": {"case": case, "source": [], "source_hashes": {"source.py": "verified"},
                             "blockers": ["Missing source"] if blocked else []}}

    def trial(self, case, provider, variant, seed, repetition, folder, max_calls):
        self.assertEqual(variant, "workflow")
        folder.mkdir()
        proposal = folder / "proposal_1"
        proposal.mkdir()
        adapter = proposal / "adapter.py"
        adapter.write_text("def adapt(core, features):\n    return core(features)\n")
        evaluation = {"status": "passed", "runs": True, "optimizer_updates": 1, "checks": dict.fromkeys(CHECKS, True),
                      "data": {"sha256": "accepted-data"}}
        return {"status": "passed", "final_valid": True, "accepted_adapter": str(adapter), "optimizer_updates": 1,
                "calls": [{"role": "proposal", "adapter_sha256": fingerprint(adapter), "evaluation": evaluation}]}

    def training(self, command, log, timeout):
        if "--preflight" in command:
            Path(log).write_text(json.dumps({"status": "ready"}) + "\n")
            return 0
        self.assertIn("--adapter", command)
        self.assertIn("--verify-gradients", command)
        self.assertIn("--connect-paths", command)
        self.assertEqual(command[command.index("--max-steps") + 1], "2")
        adapter = Path(command[command.index("--adapter") + 1])
        self.assertEqual(adapter.name, "accepted_adapter.py")
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        coordination = command[command.index("--coordination") + 1]
        checkpoint_mode = command[command.index("--checkpoint-mode") + 1]
        result = {"status": "budget_exhausted", "method": "fbnetgen", "setting": "joint", "seed": 17, "steps": 2,
                  "audit": {"data_hashes": {"adhd_train_sample.npy": "accepted-data"}},
                  "config": {"adapter": str(adapter), "source": command[command.index("--source") + 1],
                             "data": command[command.index("--data") + 1], "coordination": coordination,
                             "checkpoint_mode": checkpoint_mode, "connect_paths": True, "verify_gradients": True},
                  "coordination": {"mode": coordination, "policy_updates": 0, "last_decision": {"observed": True} if coordination == "taskforce" else None},
                  "checkpoint_policy": {"mode": checkpoint_mode},
                  "gradient_verification": {"enabled": True, "verified_steps": 2, "diagnosis_reaches_all_candidates": True,
                                            "protocol_version": 2, "last_step": {"classification_to_input": {
                                                "required": True, "finite": True, "nonzero": True}}},
                  "architecture": {"generated_adapter_active_in_training": True,
                                                                         "generated_adapter_sha256": fingerprint(adapter)}}
        (output / "report.json").write_text(json.dumps(result))
        return 0

    def execute_cycle(self, root, choice=None, blocked=False, trial=None, execute=None, mode=None, coordination="none", checkpoint_mode="weights", training_timeout_seconds=86400):
        args = self.arguments(root)
        args.coordination, args.checkpoint_mode = coordination, checkpoint_mode
        args.training_timeout_seconds = training_timeout_seconds
        if mode:
            setattr(args, mode, True)
        choice = choice or {"method": "fbnetgen", "rationale": "Use the trainable pair MLP", "expected_benefit": "Task adaptation", "risks": []}
        with patch("unibrain_agent.research_cycle.audit_inputs", return_value={"input_shape": [96] * 3}), \
             patch("unibrain_agent.research_cycle.audit_cases", return_value=self.candidates(blocked)), \
             patch("unibrain_agent.research_cycle.provider_settings", return_value=SimpleNamespace(snapshot=lambda provider: {"provider": provider, "model": "test"})), \
             patch("unibrain_agent.research_cycle.provider_preflight", return_value={"ready": True}), \
             patch("unibrain_agent.research_cycle.invoke", return_value=(choice, {"input_tokens": 12})) as provider, \
             patch("unibrain_agent.research_cycle.run_trial", side_effect=trial or self.trial) as workflow, \
             patch("unibrain_agent.research_cycle.execute", side_effect=execute or self.training) as runner:
            result = run_cycle(args)
        return result, provider.call_count, workflow.call_count, runner.call_count

    def test_accepted_generated_code_is_passed_to_joint_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, calls, workflows, executions = self.execute_cycle(root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual((calls, workflows, executions), (1, 1, 2))
            self.assertEqual(result["actual_updates"], {"acceptance": 1, "training": 2})
            self.assertEqual(result["accepted_adapter"]["sha256"], fingerprint(root / "cycle/accepted_adapter.py"))
            self.assertFalse(result["validation_weights_reused"])

    def test_failed_integration_never_starts_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, executions = self.execute_cycle(Path(temporary), trial=lambda *a, **k: {
                "status": "failed", "final_valid": False, "optimizer_updates": 0})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(executions, 0)

    def test_training_timeout_is_separate_from_preflight_timeout(self):
        observed = []
        def execute(command, log, timeout):
            observed.append(timeout)
            return self.training(command, log, timeout)
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, _ = self.execute_cycle(Path(temporary), execute=execute, training_timeout_seconds=7200)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(observed, [3600, 7200])
            self.assertEqual(result["budgets"]["training_timeout_seconds"], 7200)

    def test_training_timeout_retains_partial_update_evidence(self):
        def execute(command, log, timeout):
            if "--preflight" in command:
                return self.training(command, log, timeout)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "report.json").write_text(json.dumps({"status": "running", "steps": 1}))
            raise subprocess.TimeoutExpired(command, timeout)
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, _ = self.execute_cycle(Path(temporary), execute=execute, training_timeout_seconds=9)
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["timed_out_phase"], "joint_training")
            self.assertEqual(result["actual_updates"]["training"], 1)
            self.assertEqual(result["timeout_seconds"], 9)

    def test_missing_sources_block_before_any_provider_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, workflows, executions = self.execute_cycle(Path(temporary), blocked=True)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual((calls, workflows, executions), (0, 0, 0))

    def test_provider_can_abstain(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, workflows, executions = self.execute_cycle(Path(temporary), choice={"method": "abstain", "rationale": "No candidate fits", "expected_benefit": "", "risks": []})
            self.assertEqual(result["status"], "rejected")
            self.assertEqual((calls, workflows, executions), (1, 0, 0))

    def test_training_preflight_failure_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, executions = self.execute_cycle(Path(temporary), execute=lambda *a: 1)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(executions, 1)

    def test_training_must_confirm_adapter_identity_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter.py"
            adapter.write_text("generated code")
            record = {"status": "complete", "method": "fbnetgen", "setting": "joint", "seed": 17, "steps": 1,
                      "config": {"adapter": str(adapter)}, "architecture": {"generated_adapter_active_in_training": True,
                                                                             "generated_adapter_sha256": "wrong"}}
            with self.assertRaisesRegex(ValueError, "accepted generated adapter"):
                validate_training(record, "fbnetgen", adapter, 17, 2)
            record["architecture"]["generated_adapter_sha256"] = fingerprint(adapter)
            record["steps"] = 3
            with self.assertRaisesRegex(ValueError, "budget"):
                validate_training(record, "fbnetgen", adapter, 17, 2)

    def test_inconsistent_training_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter.py"
            adapter.write_text("generated code")
            record = {"status": "complete", "method": "fbnetgen", "setting": "joint", "seed": 17, "steps": 1,
                      "config": {"adapter": str(adapter)}, "architecture": {"generated_adapter_active_in_training": True,
                                                                             "generated_adapter_sha256": fingerprint(adapter)},
                      "audit": {"data_hashes": {"adhd_train_sample.npy": "other-data"}}}
            with self.assertRaisesRegex(ValueError, "Training data differ"):
                validate_training(record, "fbnetgen", adapter, 17, 2, data_sha256="accepted-data")

    def test_zero_exit_without_ready_preflight_does_not_train(self):
        def unready(command, log, timeout):
            Path(log).write_text("Process ended without a readiness report\n")
            return 0
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, executions = self.execute_cycle(Path(temporary), execute=unready)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(executions, 1)

    def test_static_preflight_calls_neither_provider_nor_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, workflows, executions = self.execute_cycle(Path(temporary), mode="preflight")
            self.assertEqual(result["status"], "ready")
            self.assertEqual((calls, workflows, executions), (0, 0, 0))

    def test_selection_only_stops_before_code_generation_and_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, workflows, executions = self.execute_cycle(Path(temporary), mode="selection_only")
            self.assertEqual(result["status"], "selected")
            self.assertEqual((calls, workflows, executions), (1, 0, 0))
            self.assertEqual(result["actual_updates"], {"acceptance": 0, "training": 0})

    def test_missing_acceptance_data_hash_prevents_training(self):
        def missing_data(*args, **kwargs):
            trial = self.trial(*args, **kwargs)
            trial["calls"][0]["evaluation"].pop("data")
            return trial
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, executions = self.execute_cycle(Path(temporary), trial=missing_data)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(executions, 0)
            self.assertIn("training-data hash", result["error"])

    def test_explicit_taskforce_and_checkpoint_modes_reach_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, _ = self.execute_cycle(Path(temporary), coordination="taskforce", checkpoint_mode="resume")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["training"]["coordination"]["mode"], "taskforce")
            self.assertEqual(result["training"]["checkpoint_policy"]["mode"], "resume")
            self.assertTrue(result["accepted_adapter_unchanged_after_training"])

    def test_mismatched_policy_checkpoint_or_mutated_adapter_blocks_completion(self):
        for change in ("coordination", "checkpoint_policy", "adapter"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                def altered(command, log, timeout):
                    result = self.training(command, log, timeout)
                    if "--preflight" not in command:
                        if change == "adapter":
                            Path(command[command.index("--adapter") + 1]).write_text("changed code")
                        else:
                            path = Path(command[command.index("--output") + 1]) / "report.json"
                            report = json.loads(path.read_text())
                            report[change]["mode"] = "unexpected"
                            path.write_text(json.dumps(report))
                    return result
                result, _, _, _ = self.execute_cycle(Path(temporary), execute=altered)
                self.assertEqual(result["status"], "blocked")

    def test_full_training_command_preserves_complete_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.full_training, args.evaluate_test = True, True
            command = training_command(args, "brainnetcnn", args.output / "adapter.py", args.output / "training")
            self.assertNotIn("--max-steps", command)
            self.assertNotIn("--eval-limit", command)
            self.assertIn("--evaluate-test", command)
            self.assertIn("--verify-gradients", command)

    def test_trainable_parameters_do_not_hide_detached_input(self):
        def detached(command, log, timeout):
            result = self.training(command, log, timeout)
            if "--preflight" not in command:
                path = Path(command[command.index("--output") + 1]) / "report.json"
                report = json.loads(path.read_text())
                report["gradient_verification"]["last_step"]["classification_to_input"]["nonzero"] = False
                path.write_text(json.dumps(report))
            return result
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, _ = self.execute_cycle(Path(temporary), execute=detached)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("original input image", result["error"])

    def test_bounded_cycle_cannot_evaluate_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.evaluate_test = True
            with self.assertRaisesRegex(ValueError, "explicit full training"):
                run_cycle(args)
            self.assertFalse(args.output.exists())

    def test_full_training_requires_original_fold_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.full_training = True
            with self.assertRaisesRegex(ValueError, "dataset and original fold"):
                run_cycle(args)

    def test_full_result_cannot_hide_partial_training_or_test(self):
        import copy
        from tests.test_full_cycle_results import FullCycleResultTests
        mutations = [lambda r: r.update(steps=2), lambda r: r.update(status="budget_exhausted"),
                     lambda r: r["config"].update(eval_limit=2), lambda r: r["config"].update(fold=2),
                     lambda r: r.update(history=[{"epoch": 3}]),
                     lambda r: r["test"].update(cases=[{"id": "one"}, {"id": "one"}]),
                     lambda r: r["test"].update(sample_count=1),
                     lambda r: r["audit"].update(data_hashes={"test": "changed"})]
        with tempfile.TemporaryDirectory() as temporary:
            args, splits, audit, record = FullCycleResultTests().fixture(Path(temporary))
            validate_full_training(record, args, splits, audit)
            for mutation in mutations:
                changed = copy.deepcopy(record)
                mutation(changed)
                with self.assertRaises(ValueError):
                    validate_full_training(changed, args, splits, audit)


if __name__ == "__main__":
    unittest.main()
