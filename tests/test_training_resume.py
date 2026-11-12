import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unibrain_agent.replacement_training import capture_report_state, restore_report_state


class CommittedResumeTests(unittest.TestCase):
    def fixture(self, folder):
        progress = {"steps": 2, "pretrain_steps": 0, "best_loss": .6, "best_epoch": 1, "selection_metric": "classification_loss"}
        report = {"history": [{"epoch": 1, "validation": {"classification_loss": .6}}],
                  "best_validation": {"classification_loss": .6}, "selection": {"metric": "classification_loss", "best_epoch": 1, "best_value": .6},
                  "gradient_verification": {"enabled": True, "verified_steps": 2, "last_step": {"step": 2}}}
        (folder / "updates.jsonl").write_text('{"step":1}\n{"step":2}\n')
        (folder / "best.pt").write_bytes(b"selected checkpoint fixture")
        snapshot = capture_report_state(report, folder, progress, 8.)
        return {"progress": progress, "report_state": snapshot}, report

    def test_resume_restores_counts_and_history_and_retains_abandoned_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint, original = self.fixture(folder)
            future = copy.deepcopy(original)
            future["gradient_verification"]["verified_steps"] = 3
            future["history"].append({"epoch": 2, "validation": {"classification_loss": .7}})
            future["train_seconds"] = 12.
            (folder / "report.json").write_text(json.dumps(future))
            with (folder / "updates.jsonl").open("a") as stream:
                stream.write('{"step":3}\n{"partial_attempt":')
            full_trace = (folder / "updates.jsonl").read_bytes()
            (folder / "test.json").write_text('{"stale":true}')
            result = {}
            with patch.object(Path, "read_bytes", side_effect=AssertionError("Whole-file trace reads are forbidden")):
                seconds = restore_report_state(checkpoint, result, folder)
            self.assertEqual(seconds, 12.)
            self.assertEqual(result["gradient_verification"]["verified_steps"], 2)
            self.assertEqual(result["history"], original["history"])
            self.assertEqual(result["best_validation"], original["best_validation"])
            self.assertEqual(result["resume"]["retained_attempt_seconds"], 4.)
            archive = Path(result["resume"]["archive"])
            self.assertEqual((archive / "updates.jsonl").read_bytes(), full_trace)
            self.assertTrue((archive / "test.json").is_file())
            self.assertFalse((folder / "test.json").exists())
            records = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in records if "step" in row], [1, 2])
            self.assertEqual(records[-1]["event"], "resume")
            new = capture_report_state(result, folder, checkpoint["progress"], seconds)
            self.assertEqual(new["report"]["gradient_verification"]["verified_steps"], 2)

    def test_changed_prefix_or_selected_weights_fail_before_rewriting_evidence(self):
        for target in ("updates.jsonl", "best.pt"):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                checkpoint, report = self.fixture(folder)
                (folder / target).write_bytes(b"changed evidence")
                with self.assertRaises(ValueError):
                    restore_report_state(checkpoint, {}, folder)
                self.assertEqual((folder / target).read_bytes(), b"changed evidence")
                self.assertFalse((folder / "resume_archive").exists())

    def test_unbound_old_reports_and_incorrect_verification_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint, report = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, "checkpoint-bound"):
                restore_report_state({"progress": checkpoint["progress"]}, {}, folder)
            checkpoint["report_state"]["report"]["gradient_verification"]["verified_steps"] = 0
            with self.assertRaisesRegex(ValueError, "verification count"):
                restore_report_state(checkpoint, {}, folder)

    def test_snapshot_without_selection_archives_later_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint, report = self.fixture(folder)
            checkpoint["progress"]["best_loss"] = float("inf")
            checkpoint["report_state"]["best_checkpoint_sha256"] = None
            saved = checkpoint["report_state"]["report"]
            saved.update(history=[], selection={"metric": "classification_loss", "best_epoch": None, "best_value": None})
            saved.pop("best_validation")
            result = {}
            restore_report_state(checkpoint, result, folder)
            self.assertFalse((folder / "best.pt").exists())
            self.assertTrue((Path(result["resume"]["archive"]) / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
