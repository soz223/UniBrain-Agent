import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unibrain_agent.paired_analysis import compare, paired_cases


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.paths = {key: Path(self.folder.name) / (key + ".json") for key in ("a", "b")}
        self.record = {"status": "complete", "config": {"dataset": "fixture", "fold": 1, "seed": 17,
                                                       "method": "original", "setting": "original"},
                       "audit": {"data_hashes": {"adhd_test_sample.npy": "matched"}},
                       "test": {"sample_count": 4, "full_split_count": 4,
                                "cases": [{"id": str(i), "label": i % 2, "probability": .8 if i % 2 else .2} for i in range(4)]}}
        for path in self.paths.values():
            path.write_text(json.dumps(self.record), encoding="utf-8")

    def test_equal_models_have_exact_zero_paired_difference(self):
        report = compare([self.paths], repetitions=100)
        for metric in report["metrics"].values():
            self.assertEqual(metric["difference_a_minus_b"], 0)
            self.assertEqual(metric["paired_subject_ci95"], [0, 0])

    def test_smoke_and_repeated_pairs_fail(self):
        with self.assertRaises(ValueError):
            paired_cases([self.paths, self.paths])
        self.record["config"]["max_steps"] = 2
        self.paths["a"].write_text(json.dumps(self.record), encoding="utf-8")
        with self.assertRaises(ValueError):
            paired_cases([self.paths])

    def test_different_subjects_fail(self):
        self.record["test"]["cases"][0]["id"] = "different"
        self.paths["b"].write_text(json.dumps(self.record), encoding="utf-8")
        with self.assertRaises(ValueError):
            paired_cases([self.paths])

    def additional_pair(self, name, fold=1, seed=29):
        paths = {}
        for branch in ("a", "b"):
            record = json.loads(json.dumps(self.record))
            record["config"].update(fold=fold, seed=seed)
            path = Path(self.folder.name) / f"{name}_{branch}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            paths[branch] = path
        return paths

    def test_seed_scores_are_averaged_without_repeating_subjects(self):
        second = self.additional_pair("second")
        for path in second.values():
            record = json.loads(path.read_text(encoding="utf-8"))
            for case in record["test"]["cases"]:
                case["probability"] = .6 if case["label"] else .4
            path.write_text(json.dumps(record), encoding="utf-8")
        labels, scores, _, repeats = paired_cases([self.paths, second])
        self.assertEqual(len(labels), 4)
        self.assertEqual(repeats, [2, 2, 2, 2])
        np.testing.assert_allclose(scores[:, 0], [.3, .7, .3, .7])

    def test_method_mix_within_branch_is_rejected(self):
        second = self.additional_pair("second")
        record = json.loads(second["a"].read_text(encoding="utf-8"))
        record["config"]["method"] = "synthstrip"
        second["a"].write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mix methods"):
            paired_cases([self.paths, second])

    def test_changed_test_data_across_seeds_is_rejected(self):
        second = self.additional_pair("second")
        for path in second.values():
            record = json.loads(path.read_text(encoding="utf-8"))
            record["audit"]["data_hashes"]["adhd_test_sample.npy"] = "changed"
            path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed across seeds"):
            paired_cases([self.paths, second])

    def test_missing_fold_seed_combination_is_rejected(self):
        second = self.additional_pair("second")
        third = self.additional_pair("third", fold=2, seed=17)
        with self.assertRaisesRegex(ValueError, "same set of matched seeds"):
            paired_cases([self.paths, second, third])


if __name__ == "__main__":
    unittest.main()
