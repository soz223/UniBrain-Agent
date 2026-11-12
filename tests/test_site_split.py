import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from unibrain_agent.site_split import TEMPLATES, export_site, partitions
from unibrain_agent.cohort_provenance import file_hash, validate_site


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
            hashes = {name: file_hash(root / "result" / name) for name in
                      [f"adhd_{split}_sample.npy" for split in ("train", "val", "test")] + list(TEMPLATES)}
            checked = validate_site(root / "result", "abide", "held", hashes)
            self.assertEqual(checked["status"], "verified")
            self.assertIsNone(checked["fold"])
            self.assertNotIn("subject0", json.dumps(checked))
            from unibrain_agent.replacement_training import parser, preflight
            args = parser().parse_args(["--source", str(root / "source"), "--data", str(root / "result"),
                                        "--output", str(root / "run"), "--method", "original", "--setting", "original",
                                        "--dataset", "abide", "--split-id", "held", "--preflight"])
            with patch("unibrain_agent.native.verify_original", return_value={"scope": "CPU source fixture"}), \
                    patch("unibrain_agent.sources.verify", return_value=[]):
                audit = preflight(args)
                self.assertEqual(audit["cohort"]["scope"], "site_holdout")
                self.assertEqual(audit["data_hashes"], hashes)
                self.assertEqual(audit["cohort"]["status"], "verified")
            with self.assertRaisesRegex(ValueError, "declared dataset"):
                validate_site(root / "result", "adhd", "held", hashes)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_site(root / "result", "abide", "held", {**hashes, TEMPLATES[0]: "0" * 64})
            report["splits"]["train"]["master_indices"][0] = 0
            (root / "result/site_split.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "indices"):
                validate_site(root / "result", "abide", "held", hashes)
            (root / "result/site_split.json").write_text(json.dumps(saved), encoding="utf-8")
            metadata = root / "result/source_site_metadata.csv"
            metadata.write_text(metadata.read_text(encoding="utf-8").replace("subject0,held", "subject0,source"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata checksum"):
                validate_site(root / "result", "abide", "held", hashes)
            saved["sources"]["metadata_sha256"] = file_hash(metadata)
            (root / "result/site_split.json").write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with the source metadata"):
                validate_site(root / "result", "abide", "held", hashes)
            with self.assertRaisesRegex(ValueError, "explicit origin"):
                export_site(root / "master.npy", root / "metadata.csv", root, root / "bad", "held", " ", "external_public")

    def test_site_training_refuses_fold_aliases_and_reused_fold_weights(self):
        from unibrain_agent.replacement_training import parser
        from unibrain_agent.site_training import validate_site_options
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "site_split.json").write_text("{}", encoding="utf-8")
            base = ["--source", str(root), "--data", str(root), "--output", str(root / "output"),
                    "--dataset", "abide", "--split-id", "held"]
            for method, setting in (("original", "original"), ("synthstrip", "joint"), ("synthstrip", "independent_finetuned")):
                args = parser().parse_args(base + ["--method", method, "--setting", setting])
                validate_site_options(args)
                args.fold = 1
                with self.assertRaisesRegex(ValueError, "no numeric fold"):
                    validate_site_options(args)
                args.fold = None
                args.initialization = "training_fold"
                with self.assertRaisesRegex(ValueError, "numeric-fold checkpoints"):
                    validate_site_options(args)
            args = parser().parse_args(base)
            args.split_id = None
            with self.assertRaisesRegex(ValueError, "explicit split ID"):
                validate_site_options(args)

    def test_site_entry_point_dispatches_full_trainer_after_protocol_validation(self):
        from unibrain_agent.site_training import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "site_split.json").write_text("{}", encoding="utf-8")
            args = ["--source", str(root), "--data", str(root), "--output", str(root / "output"),
                    "--dataset", "abide", "--split-id", "held", "--preflight"]
            with patch("unibrain_agent.replacement_training.train", return_value={"status": "ready"}) as train:
                self.assertEqual(main(args), 0)
                self.assertEqual(train.call_args.args[0].split_id, "held")
                self.assertIsNone(train.call_args.args[0].fold)


if __name__ == "__main__":
    unittest.main()
