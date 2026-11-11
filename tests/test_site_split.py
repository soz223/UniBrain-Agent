import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unibrain_agent.site_split import TEMPLATES, export_site, partitions


class SiteSplitTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"subject{x}" for x in range(12)]
        self.labels = np.arange(12) % 2
        self.metadata = {name: "held" if index < 4 else "source" for index, name in enumerate(self.ids)}

    def test_held_site_is_exclusively_test_and_seed_is_repeatable(self):
        one = partitions(self.ids, self.labels, self.metadata, "held")
        two = partitions(self.ids, self.labels, self.metadata, "held")
        np.testing.assert_array_equal(one["test"], [0, 1, 2, 3])
        for split in one:
            np.testing.assert_array_equal(one[split], two[split])
        self.assertEqual(set(np.concatenate(list(one.values()))), set(range(12)))
        self.assertEqual(sum(len(value) for value in one.values()), 12)
        for split in ("train", "val"):
            self.assertEqual(set(self.labels[one[split]]), {0, 1})

    def test_missing_or_duplicate_subjects_and_empty_site_fail(self):
        with self.assertRaises(ValueError):
            partitions(self.ids, self.labels, {}, "held")
        with self.assertRaises(ValueError):
            partitions(self.ids[:-1] + self.ids[:1], self.labels, self.metadata, "held")
        with self.assertRaises(ValueError):
            partitions(self.ids, self.labels, self.metadata, "unknown")

    def test_unstratifiable_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            partitions(self.ids, np.zeros(12), self.metadata, "held")

    def test_site_whitespace_does_not_split_a_site(self):
        metadata = dict(self.metadata)
        metadata[self.ids[0]] = " held "
        result = partitions(self.ids, self.labels, metadata, " held ")
        np.testing.assert_array_equal(result["test"], [0, 1, 2, 3])

    def test_export_preserves_indices_and_template_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volumes = ("raw_img_np", "raw_img_brain_mask_np", "raw_img_gm_mask_np", "raw_img_aal_mask_np")
            master = np.zeros(6, dtype=[("id", "U12"), ("label", "i1"),
                                       *((name, "u1", (96, 96, 96)) for name in volumes)])
            master["id"] = [f"subject{i}" for i in range(6)]
            master["label"] = np.arange(6) % 2
            np.save(root / "master.npy", master)
            (root / "metadata.csv").write_text("id,site\n" + "\n".join(
                f"subject{i},{'held' if i < 2 else 'source'}" for i in range(6)), encoding="utf-8")
            for name in TEMPLATES:
                np.save(root / name, np.zeros((96, 96, 96), dtype=np.uint8))
            report = export_site(root / "master.npy", root / "metadata.csv", root, root / "result", "held",
                                 "Unit test fixture", "external_public")
            self.assertEqual(report["splits"]["test"]["ids"], ["subject0", "subject1"])
            self.assertEqual(report["template_provenance"]["scope"], "external_public")
            self.assertEqual(sum(part["count"] for part in report["splits"].values()), 6)
            saved = json.loads((root / "result/cohort.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            with self.assertRaisesRegex(ValueError, "explicit origin"):
                export_site(root / "master.npy", root / "metadata.csv", root, root / "bad", "held", " ", "external_public")


if __name__ == "__main__":
    unittest.main()
