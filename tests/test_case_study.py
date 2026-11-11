import json
import tempfile
import unittest
from pathlib import Path

from unibrain_agent.case_study import sha256, summarize_trace


class CaseStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = {"steps_requested": 2, "attempts": [], "accepted_adapter": "private/path"}
        for number, passed in ((1, False), (2, True)):
            folder = self.directory / f"attempt_{number}"
            folder.mkdir()
            adapter = folder / "adapter.py"
            adapter.write_text("return soft_mask\n" if passed else "return hard_mask\n", encoding="utf-8")
            value = {"status": "passed" if passed else "adapter_failed", "adapter_sha256": sha256(adapter),
                     "gradient_check": {"mask_requires_grad": passed, "connected_tensors": int(passed)},
                     "updates": [{"step": 1, "finite_gradients": True}] if passed else [],
                     "data": {"subject_id": "SECRET_SUBJECT"}, "error": None if passed else "SECRET_PATH"}
            (folder / "validation.json").write_text(json.dumps(value), encoding="utf-8")
            (folder / "repair.json").write_text('"SECRET_TRANSCRIPT"', encoding="utf-8")
            self.root["attempts"].append({"attempt": number, "sha256": sha256(adapter), "validation": value,
                                           "fault_injected": not passed})
        self.save()

    def save(self):
        (self.directory / "report.json").write_text(json.dumps(self.root), encoding="utf-8")

    def test_observed_repair_retains_failure_and_removes_private_content(self):
        result = summarize_trace(self.directory)
        self.assertTrue(result["controlled_fault_injection"])
        self.assertTrue(result["repaired"])
        self.assertEqual(result["metrics"]["optimizer_updates"], 1)
        self.assertEqual(result["attempts"][0]["failure_category"], "gradient_disconnection")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertNotIn("private/path", json.dumps(result))
        self.assertEqual(result["transitions"][0]["added_lines"], 1)

    def test_modified_adapter_rejected(self):
        (self.directory / "attempt_2/adapter.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Adapter hash"):
            summarize_trace(self.directory)

    def test_disagreeing_validation_rejected(self):
        self.root["attempts"][1]["validation"]["status"] = "adapter_failed"
        self.save()
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize_trace(self.directory)

    def test_unknown_training_steps_cannot_be_claimed(self):
        value = self.root["attempts"][1]["validation"]
        value["updates"] = []
        (self.directory / "attempt_2/validation.json").write_text(json.dumps(value), encoding="utf-8")
        self.save()
        with self.assertRaisesRegex(ValueError, "lacks actual"):
            summarize_trace(self.directory)


if __name__ == "__main__":
    unittest.main()
