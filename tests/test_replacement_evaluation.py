import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from unibrain_agent import replacement_evaluation as evaluation
from unibrain_agent.replacement_training import parser as training_parser


class ReplacementEvaluationTests(unittest.TestCase):
    def fixture(self, root, method="brainnetcnn"):
        data, source = root / "data", root / "source"
        data.mkdir()
        source.mkdir()
        adapter = root / "adapter.py"
        adapter.write_text("def adapt(core, graph, features):\n    return core(graph, features)\n")
        settings = vars(training_parser().parse_args([
            "--source", str(source), "--data", str(data), "--output", str(root / "training"),
            "--method", method, "--adapter", str(adapter), "--dataset", "abide", "--fold", "1",
            "--connect-paths", "--checkpoint-mode", "weights"]))
        config = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
        identity = {key: config[key] for key in evaluation.IDENTITY_FIELDS}
        identity.update(selection_metric="classification_loss", adapter_hash=evaluation.fingerprint(adapter),
                        weights_hash=None, plans_hash=None, host_hash=None, host_report_hash=None,
                        plans_provenance_hash=None, data_hashes={}, source=[{"commit": "source"}],
                        composition={"mode": "simultaneous", "secondary_extraction": None,
                                     "previous_checkpoint": None, "previous_report": None, "extraction_weights": None},
                        implementation_hashes={name: evaluation.implementation_hash(name) for name in evaluation.IMPLEMENTATION_FILES})
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(.25)
            model.bias.fill_(.5)
        payload = {"checkpoint_kind": "evaluation_weights_only", "identity": identity,
                   "progress": {"steps": 2, "best_loss": float("inf"), "best_value": None, "best_epoch": None},
                   "model": model.state_dict()}
        checkpoint = root / "last.pt"
        torch.save(payload, checkpoint)
        report = {"status": "budget_exhausted", "method": method, "setting": "joint", "seed": 17,
                  "steps": 2, "config": config, "architecture": {"parameters": 3},
                  "audit": {"source": identity["source"], "candidate_sources": [], "data_hashes": {}}}
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report))
        args = evaluation.parser().parse_args([
            "--source", str(source), "--data", str(data), "--output", str(root / "evaluation"),
            "--checkpoint", str(checkpoint), "--checkpoint-sha256", evaluation.fingerprint(checkpoint)])
        return args, payload, report, report_path

    def test_real_weights_only_payload_and_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, _, _ = self.fixture(Path(temporary))
            loaded = evaluation.load_payload(args.checkpoint, args.checkpoint_sha256)
            self.assertTrue(torch.equal(loaded["model"]["weight"], payload["model"]["weight"]))
            with self.assertRaisesRegex(ValueError, "changed file"):
                evaluation.load_payload(args.checkpoint, "0" * 64)
            payload["model"]["weight"].fill_(float("nan"))
            corrupt = args.checkpoint.with_name("corrupt.pt")
            torch.save(payload, corrupt)
            with self.assertRaisesRegex(ValueError, "nonfinite tensor"):
                evaluation.load_payload(corrupt, evaluation.fingerprint(corrupt))

    def test_migrated_adapter_retains_original_content_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, payload, report, report_path = self.fixture(root, "synthstrip")
            migrated = root / "migrated_adapter.py"
            Path(report["config"]["adapter"]).rename(migrated)
            args.adapter = migrated
            restored, bindings = evaluation.recorded_arguments(args, payload, report, report_path)
            self.assertEqual(restored.adapter, migrated.resolve())
            self.assertEqual(bindings["adapter"]["sha256"], payload["identity"]["adapter_hash"])
            migrated.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed file"):
                evaluation.recorded_arguments(args, payload, report, report_path)

    def test_different_fold_settings_and_best_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary))
            args.fold = 2
            with self.assertRaisesRegex(ValueError, "identity differs: fold"):
                evaluation.recorded_arguments(args, payload, report, report_path)
            args.fold = None
            changed = copy.deepcopy(report)
            changed["config"]["connect_paths"] = False
            with self.assertRaisesRegex(ValueError, "configuration disagree"):
                evaluation.recorded_arguments(args, payload, changed, report_path)
            args.checkpoint = args.checkpoint.with_name("best.pt")
            report["selection"] = {"best_epoch": 1, "best_value": .5}
            with self.assertRaisesRegex(ValueError, "validation selection"):
                evaluation.recorded_arguments(args, payload, report, report_path)

    def test_architecture_changes_block_but_trainer_changes_are_disclosed(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, payload, _, _ = self.fixture(Path(temporary))
            identity = payload["identity"]
            identity["implementation_hashes"]["replacement_training.py"] = "1" * 64
            audit = evaluation.code_audit(identity)
            self.assertEqual(audit["other_changed_files"], ["replacement_training.py"])
            identity["implementation_hashes"]["full_graph_candidates.py"] = "2" * 64
            with self.assertRaisesRegex(ValueError, "Architecture implementation differs"):
                evaluation.code_audit(identity)
            identity["implementation_hashes"]["../outside.py"] = "3" * 64
            with self.assertRaisesRegex(ValueError, "Invalid recorded implementation"):
                evaluation.code_audit(identity)

    def test_exact_connections_adapter_state_and_frozen_evaluation_are_reconstructed(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary))
            report["config"].update(cut_boundary="registration", repair_strategy="straight_through", undo_repair="mask")
            payload["identity"].update(cut_boundary="registration", repair_strategy="straight_through", undo_repair="mask")
            restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
            tiny = torch.nn.Linear(2, 1)
            with patch("unibrain_agent.full_extended.build_extended_model", return_value=(tiny, {}, {})) as build, \
                 patch("unibrain_agent.full_connectivity.connect_model", side_effect=lambda model, **kwargs: model) as connect, \
                 patch("unibrain_agent.candidate_integration.bind_extended_adapters") as bind, \
                 patch("unibrain_agent.replacement_training.configure_boundaries", return_value=[]) as boundaries:
                model = evaluation.reconstruct(restored, payload, {"architecture": report["architecture"]})
            build.assert_called_once_with(args.source.resolve(), "brainnetcnn")
            connect.assert_called_once_with(tiny, cut="registration", repair="straight_through", undo="mask")
            bind.assert_called_once_with(tiny, "brainnetcnn", restored.adapter)
            boundaries.assert_called_once_with(tiny, "brainnetcnn", "joint", False)
            self.assertTrue(torch.equal(model.weight, payload["model"]["weight"]))
            self.assertFalse(model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_synthstrip_reconstruction_uses_recorded_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary), "synthstrip")
            restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
            tiny = torch.nn.Linear(2, 1)
            with patch("unibrain_agent.full_model.build_full_model", return_value=(tiny, {}, {})) as build, \
                 patch("unibrain_agent.full_connectivity.connect_model", side_effect=lambda model, **kwargs: model), \
                 patch("unibrain_agent.replacement_training.configure_boundaries", return_value=[]):
                evaluation.reconstruct(restored, payload, {"architecture": report["architecture"]})
            build.assert_called_once_with(args.source.resolve(), restored.adapter)

    def test_wrong_state_names_shapes_or_parameter_count_fail_strictly(self):
        for fault in ("name", "shape", "count"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                args, payload, report, report_path = self.fixture(Path(temporary))
                restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
                if fault == "name":
                    payload["model"]["wrong"] = payload["model"].pop("weight")
                if fault == "shape":
                    payload["model"]["weight"] = torch.zeros(1, 3)
                if fault == "count":
                    report["architecture"]["parameters"] = 4
                tiny = torch.nn.Linear(2, 1)
                with patch("unibrain_agent.full_extended.build_extended_model", return_value=(tiny, {}, {})), \
                     patch("unibrain_agent.full_connectivity.connect_model", side_effect=lambda model, **kwargs: model), \
                     patch("unibrain_agent.candidate_integration.bind_extended_adapters"), \
                     self.assertRaises(ValueError):
                    evaluation.reconstruct(restored, payload, {"architecture": report["architecture"]})

    def test_static_preflight_preserves_unset_best_metric_and_never_builds(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _, _ = self.fixture(Path(temporary))
            args.preflight = True
            with patch.object(evaluation, "audit_sources", return_value={}), \
                 patch.object(evaluation, "audit_data", return_value={}), \
                 patch.object(evaluation, "reconstruct") as build:
                result = evaluation.evaluate_checkpoint(args)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(result["provenance"]["nonfinite_progress_fields"], ["best_loss"])
            json.dumps(result, allow_nan=False)
            build.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_source_and_candidate_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary))
            restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
            with patch("unibrain_agent.native.verify_original", return_value=[{"commit": "wrong"}]), \
                 self.assertRaisesRegex(ValueError, "Original source differs"):
                evaluation.audit_sources(restored, payload["identity"], report)
            with patch("unibrain_agent.native.verify_original", return_value=payload["identity"]["source"]), \
                 patch("unibrain_agent.sources.verify", return_value=[{"commit": "wrong"}]), \
                 self.assertRaisesRegex(ValueError, "Candidate source differs"):
                evaluation.audit_sources(restored, payload["identity"], report)

    def test_changed_data_block_before_loading_or_evaluating_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary))
            restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
            names = [f"adhd_{split}_sample.npy" for split in evaluation.SPLITS] + list(evaluation.TEMPLATES)
            for name in names:
                (args.data / name).write_bytes(b"recorded data")
            hashes = {name: evaluation.fingerprint(args.data / name) for name in names}
            payload["identity"]["data_hashes"] = hashes
            report["audit"]["data_hashes"] = hashes
            (args.data / "adhd_test_sample.npy").write_bytes(b"different data")
            with patch("unibrain_agent.data.OriginalDataset") as load, \
                 self.assertRaisesRegex(ValueError, "changed file: adhd_test_sample"):
                evaluation.audit_data(restored, payload["identity"], report)
            load.assert_not_called()

    def test_cohort_manifest_is_required_after_data_hash_verification(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary))
            restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
            names = [f"adhd_{split}_sample.npy" for split in evaluation.SPLITS] + list(evaluation.TEMPLATES)
            hashes = dict.fromkeys(names, "0" * 64)
            payload["identity"]["data_hashes"] = hashes
            report["audit"]["data_hashes"] = hashes
            class Dataset:
                def __init__(self, path):
                    self.array = {"id": [Path(path).stem]}
                def __len__(self):
                    return 1
            with patch.object(evaluation, "file_binding", return_value={}), \
                 patch("unibrain_agent.data.OriginalDataset", Dataset), \
                 patch("numpy.load", return_value=np.zeros((96, 96, 96), dtype=np.uint8)), \
                 self.assertRaisesRegex(ValueError, "cohort manifest is required"):
                evaluation.audit_data(restored, payload["identity"], report)

    def test_sequential_composition_rebuilds_topology_then_loads_final_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, payload, report, report_path = self.fixture(Path(temporary), "voxelmorph")
            secondary = Path(temporary) / "extraction.py"
            secondary.write_text("extraction code")
            report["config"].update(secondary_extraction=str(secondary), composition_mode="sequential")
            payload["identity"]["composition"].update(mode="sequential", secondary_extraction=evaluation.fingerprint(secondary))
            restored, _ = evaluation.recorded_arguments(args, payload, report, report_path)
            audit = {"architecture": report["architecture"]}
            tiny = torch.nn.Linear(2, 1)
            with patch("unibrain_agent.full_composition.build_composed_model", return_value=(tiny, {}, {})) as build, \
                 patch("unibrain_agent.full_connectivity.connect_model", side_effect=lambda model, **kwargs: model), \
                 patch("unibrain_agent.candidate_integration.bind_adapter"), \
                 patch("unibrain_agent.replacement_training.configure_boundaries", return_value=[]):
                model = evaluation.reconstruct(restored, payload, audit)
            build.assert_called_once_with(args.source.resolve(), "voxelmorph", secondary.resolve(), mode="simultaneous")
            self.assertEqual(audit["reconstruction"]["recorded_composition_mode"], "sequential")
            self.assertTrue(torch.equal(model.weight, payload["model"]["weight"]))

    def test_evaluation_writes_metrics_without_optimization(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _, _ = self.fixture(Path(temporary))
            for name in evaluation.TEMPLATES:
                np.save(args.data / name, np.zeros((96, 96, 96), dtype=np.float32))
            tiny = torch.nn.Linear(2, 1)
            def evaluate(model, dataset, templates, limit, output):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                self.assertEqual([tuple(t.shape) for t in templates], [(1, 1, 96, 96, 96)] * 3)
                self.assertEqual(model(torch.ones(1, 2)).item(), 1.)
                return {"sample_count": 2, "full_split_count": 2, "metrics": {"accuracy": .5}, "cases": []}
            with patch.dict(os.environ), \
                 patch.object(evaluation, "audit_sources", return_value={}), \
                 patch.object(evaluation, "audit_data", return_value={}), \
                 patch("unibrain_agent.full_extended.build_extended_model", return_value=(tiny, {}, {})), \
                 patch("unibrain_agent.full_connectivity.connect_model", side_effect=lambda model, **kwargs: model), \
                 patch("unibrain_agent.candidate_integration.bind_extended_adapters"), \
                 patch("unibrain_agent.replacement_training.configure_boundaries", return_value=[]), \
                 patch("unibrain_agent.full_replacement_worker.selected_device", return_value="0"), \
                 patch("unibrain_agent.full_replacement_worker.available_gpu", return_value={"scope": "CPU fixture"}), \
                 patch("torch.Tensor.cuda", lambda tensor: tensor), \
                 patch("torch.nn.Module.cuda", lambda model: model), \
                 patch("unibrain_agent.data.OriginalDataset", return_value=[0, 1]), \
                 patch("unibrain_agent.replacement_training.evaluate", side_effect=evaluate), \
                 patch("torch.optim.Adam", side_effect=AssertionError("Evaluation must not construct an optimizer")):
                result = evaluation.evaluate_checkpoint(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertTrue(result["full_split_evaluated"])
            saved = json.loads((args.output / "report.json").read_text())
            self.assertTrue(saved["strict_state_loaded"])
            self.assertEqual(saved["evaluation"]["metrics"]["accuracy"], .5)
            self.assertFalse(torch.cuda.is_initialized())

    def test_overflowed_json_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            for value in ("NaN", "Infinity", "1e999"):
                path.write_text('{"value":' + value + '}')
                with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                    evaluation.read_json(path)


if __name__ == "__main__":
    unittest.main()
