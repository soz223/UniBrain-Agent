import json
import tempfile
import unittest
from pathlib import Path

from unibrain_agent.cohort_provenance import validate_cohort


class CohortProvenanceTests(unittest.TestCase):
    def test_public_samples_without_manifest_remain_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            result = validate_cohort(directory, None, None, {})
            self.assertEqual(result["status"], "unverified")
            self.assertFalse(result["array_hashes_verified"])
            with self.assertRaisesRegex(ValueError, "manifest is required"):
                validate_cohort(directory, "abide", 1, {}, required_scope="full_fold")

    def test_incomplete_or_legacy_export_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            for record in ({"version": 1, "scope": "full_fold"}, {"version": 2, "scope": "full_fold", "status": "failed"}):
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "complete version 2"):
                    validate_cohort(directory, "abide", 1, {})


if __name__ == "__main__":
    unittest.main()
