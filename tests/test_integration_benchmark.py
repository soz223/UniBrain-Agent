import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from unibrain_agent.integration_benchmark import CHECKS, case_blockers, evaluate, fault_selftest, run_trial, verdict, visible_feedback


def result(**changes):
    value = {"status": "passed", "runs": True, "checks": dict.fromkeys(CHECKS, True), "optimizer_updates": 1}
    value.update(changes)
    return value


class IntegrationBenchmarkTests(unittest.TestCase):
    def test_timeout_preserves_observed_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "progress.json").write_text(json.dumps({"status": "checking", "updates": [{"step": 1}]}))
            case = {"evaluator": {"command": ["{python}", "fixture.py"], "progress_file": "progress.json"}}
            with patch("unibrain_agent.integration_benchmark.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1)):
                report = evaluate(case, folder / "adapter.py", folder, 17)
            self.assertEqual(report["status"], "infrastructure_failed")
            self.assertEqual(report["optimizer_updates"], 1)

    def test_known_contract_failure_is_failed_even_when_later_checks_are_unmeasured(self):
        report = {"status": "adapter_failed", "runs": False, "checks": {"contract": False}}
        self.assertEqual(verdict(report), "failed")
        self.assertEqual(verdict({"status": "adapter_failed", "checks": {"contract": True}}), "blocked")

    def test_missing_independent_check_is_blocked(self):
        report = result()
        report["checks"].pop("fidelity")
        self.assertEqual(verdict(report), "blocked")
        self.assertEqual(verdict(result(optimizer_updates=0)), "failed")
        self.assertEqual(verdict(result(status="infrastructure_failed")), "blocked")

    def test_ablated_feedback_cannot_change_final_verdict(self):
        report = result()
        report["checks"]["gradient"] = False
        report["diagnostic"] = {"gradient": "Disconnected", "spatial": "Aligned", "error": "gradient leak"}
        filtered = visible_feedback(report, "without_gradient")
        self.assertNotIn("gradient", filtered["checks"])
        self.assertNotIn("gradient", filtered["diagnostic"])
        self.assertNotIn("error", filtered["diagnostic"])
        self.assertEqual(verdict(report), "failed")

    def test_fault_fixture_catches_failures_and_has_no_clean_false_alarm(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fault_selftest(Path(directory) / "faults")
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), 15)
            self.assertFalse(report["cases"][0]["detected"])
            nearest = next(row for row in report["cases"] if row["fault"] == "nearest_resampling")
            self.assertFalse(nearest["checks"]["gradient"])

    def trial(self, variant, reports, actions=None, expected_fit=True, screen=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("# Unit test source", encoding="utf-8")
            case = {"id": "example", "expected_fit": expected_fit, "interface": "Return tensor", "source_files": [str(source)],
                    "evaluator": {"command": ["{python}", "{adapter}", "{output}", "{seed}"], "cwd": str(root)}}
            actions = iter(actions or ["submit"] * len(reports))
            proposals = []

            def invoke(config, prompt, schema, folder, role):
                if role == "screen":
                    return {"fits": screen, "rationale": "Unit test"}, {"input_tokens": 1}
                proposals.append(prompt)
                return {"code": "# " + str(len(proposals)), "action": next(actions), "rationale": "Unit test"}, {"output_tokens": 1}

            with patch("unibrain_agent.providers.provider_settings", return_value=SimpleNamespace(snapshot=lambda _: {"provider": "codex"})), \
                    patch("unibrain_agent.providers.invoke", side_effect=invoke), \
                    patch("unibrain_agent.integration_benchmark.evaluate", side_effect=reports):
                row = run_trial(case, {"provider": "codex"}, variant, 17, 0, root / "trial")
            return row, proposals

    def test_general_has_no_checker_feedback_retry(self):
        report = result()
        report["checks"]["gradient"] = False
        row, prompts = self.trial("general", [report])
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["silent_failure"])
        self.assertEqual(len(prompts), 1)

    def test_checks_action_calls_tool_and_then_submits(self):
        broken = result(optimizer_updates=0)
        broken["checks"]["gradient"] = False
        row, prompts = self.trial("checks", [broken, result()], ["check", "submit"])
        self.assertTrue(row["final_valid"])
        self.assertFalse(row["first_attempt_valid"])
        self.assertEqual(len(prompts), 2)
        self.assertIn('"gradient": false', prompts[1])

    def test_any_observed_update_closes_the_training_budget(self):
        broken = result(optimizer_updates=1)
        broken["checks"]["fidelity"] = False
        row, prompts = self.trial("checks", [broken, result(optimizer_updates=2)], ["check", "submit"])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["optimizer_updates"], 1)
        self.assertTrue(row["training_budget_closed"])
        self.assertEqual(len(prompts), 1)

    def test_screen_rejection_is_not_an_integration_success(self):
        row, prompts = self.trial("workflow", [], expected_fit=False, screen=False)
        self.assertEqual(row["status"], "rejected")
        self.assertTrue(row["applicability_correct"])
        self.assertFalse(row["final_valid"])
        self.assertFalse(prompts)

    def test_no_gradient_ablation_is_judged_by_full_evaluator(self):
        broken = result(optimizer_updates=0)
        broken["checks"]["gradient"] = False
        row, prompts = self.trial("without_gradient", [broken])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(len(prompts), 1)


if __name__ == "__main__":
    unittest.main()
