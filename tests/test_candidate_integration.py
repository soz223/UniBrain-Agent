import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from scripts.generate_integration_config import CANDIDATES, UNIMPLEMENTED, make_case
from unibrain_agent.candidate_integration import (GeneratedCore, bind_adapter, bind_extended_adapters, check_adapter, core_probe,
                                                 gradient_evidence, registration_geometry)
from unibrain_agent.integration_benchmark import case_blockers, source_context
from unibrain_agent.full_replacement import AdapterFailure


class TissueCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))

    def forward(self, image):
        logits = torch.cat((image * self.weight, -image * self.weight), dim=1)
        return [logits, F.avg_pool3d(logits, 2)]


class RegistrationCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))
        self.integrate = nn.Identity()
        self.fullsize = None

    def forward(self, moving, fixed):
        flow = (moving - fixed).repeat(1, 3, 1, 1, 1) * self.weight
        return moving + self.weight, fixed - self.weight, flow


class CandidateIntegrationTests(unittest.TestCase):
    def test_generated_adapter_is_the_executed_module(self):
        host = nn.Module()
        host.seg_net = TissueCore()
        image = torch.arange(64.).reshape(1, 1, 4, 4, 4)
        expected = host.seg_net(image)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.py"
            path.write_text("def adapt(core, image):\n    result = core(image)\n    return [result[0] * 2, result[1] * 2]\n")
            wrapper = bind_adapter(host, "nnunet", path)
            self.assertIs(host.seg_net, wrapper)
            self.assertTrue(torch.equal(host.seg_net(image)[0], expected * 2))
            evidence, _ = core_probe(wrapper, (image,))
            self.assertTrue(evidence["contract"])
            self.assertFalse(evidence["fidelity"])

    def test_detachment_passes_value_check_but_fails_gradient(self):
        code = "def adapt(core, moving, fixed):\n    result = core(moving, fixed)\n    return (result[0].detach(), result[1].detach(), result[2].detach())\n"
        wrapper = GeneratedCore(RegistrationCore(), code, "voxelmorph")
        image = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
        fixed = torch.zeros_like(image, requires_grad=False)
        evidence, _ = core_probe(wrapper, (image, fixed))
        self.assertTrue(evidence["fidelity"])
        loss = wrapper(image, fixed)[2].square().mean()
        self.assertFalse(gradient_evidence(loss, wrapper.parameters(), (image, fixed))["passed"])

    def test_good_registration_has_parameter_and_moving_input_gradients(self):
        wrapper = GeneratedCore(RegistrationCore(), "def adapt(core, moving, fixed):\n    return core(moving, fixed)\n", "voxelmorph")
        moving = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
        fixed = torch.zeros_like(moving, requires_grad=False)
        loss = wrapper(moving, fixed)[2].square().mean()
        evidence = gradient_evidence(loss, wrapper.parameters(), (moving, fixed))
        self.assertTrue(evidence["passed"])
        self.assertIs(wrapper.integrate, wrapper.core.integrate)
        self.assertIsNone(wrapper.fullsize)

    def test_orientation_error_is_measured(self):
        wrapper = GeneratedCore(TissueCore(), "def adapt(core, image):\n    result = core(image)\n    return [result[0].flip(2), result[1].flip(2)]\n", "nnunet")
        image = torch.linspace(-1, 1, 64).reshape(1, 1, 4, 4, 4)
        evidence, _ = core_probe(wrapper, (image,))
        self.assertTrue(evidence["contract"])
        self.assertFalse(evidence["spatial"])
        self.assertFalse(evidence["fidelity"])

    def test_output_structure_and_code_restrictions(self):
        wrapper = GeneratedCore(TissueCore(), "def adapt(core, image):\n    return core(image)[0]\n", "nnunet")
        evidence, _ = core_probe(wrapper, (torch.randn(1, 1, 4, 4, 4),))
        self.assertFalse(evidence["contract"])
        for code in ("import os\ndef adapt(core, image):\n    return core(image)",
                     "def adapt(core, image):\n    core.weight = image\n    return core(image)"):
            with self.assertRaises(ValueError):
                check_adapter(code, "nnunet")

    def test_independent_geometry_rejects_a_changed_grid(self):
        from unibrain_agent.full_candidates import compose_grids
        image = torch.linspace(0, 1, 512).reshape(1, 1, 8, 8, 8)
        affine = torch.eye(3, 4).unsqueeze(0)
        positive = torch.zeros(1, 3, 8, 8, 8)
        positive[:, 0] = .1
        forward, backward = compose_grids(image, affine, affine, positive, -positive)
        registration = SimpleNamespace(method="lapirn", core=SimpleNamespace(diff_transform=lambda v, g: v, grid_1=None),
                                       forward_grid=forward, inverse_grid=backward)
        model = SimpleNamespace(reg_net=registration)
        warped = F.grid_sample(image, forward, align_corners=True)
        output = ([image], [], [warped], [affine], [affine])
        reference = (positive, None, None, positive)
        self.assertTrue(registration_geometry(model, output, reference)["passed"])
        registration.forward_grid = forward.flip(-1)
        self.assertFalse(registration_geometry(model, output, reference)["passed"])

    def test_config_builder_covers_fourteen_and_blocks_missing_plans(self):
        self.assertEqual(len(CANDIDATES), 14)
        self.assertEqual(len(UNIMPLEMENTED), 0)
        with tempfile.TemporaryDirectory() as temporary:
            for method in CANDIDATES:
                case = make_case(method, temporary, temporary, temporary, steps=1)
                self.assertIn("{adapter}", case["evaluator"]["command"])
                self.assertEqual(case["max_optimizer_updates"], 1)
                self.assertEqual(case["implementation"], "full_generated_adapter")
                if method == "nnunet":
                    self.assertTrue(any("plans file" in value for value in case_blockers(case)))

    def test_gradient_check_retains_native_frozen_parameters(self):
        core = TissueCore()
        core.register_parameter("frozen_center", nn.Parameter(torch.ones(4), requires_grad=False))
        image = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
        loss = core(image)[0].square().mean()
        evidence = gradient_evidence(loss, core.parameters(), (image, True))
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["total_tensors"], 1)

    def test_plan_context_excludes_subject_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = root / "private_plans.json"
            plans.write_text(json.dumps({"subject_ids": ["PRIVATE_SUBJECT"], "dataset_name": "PRIVATE_DATASET",
                "configurations": {"3d_fullres": {"patch_size": [96] * 3, "architecture": {
                    "network_class_name": "PlainConvUNet", "arch_kwargs": {"features_per_stage": [32, 64], "subject_id": "PRIVATE_SUBJECT"}}}}}))
            case = make_case("nnunet", root, root, root, plans=plans)
            self.assertNotIn(str(plans), case["source_files"])
            context = json.dumps(case["structured_evidence"])
            self.assertNotIn("PRIVATE_SUBJECT", context)
            self.assertNotIn("PRIVATE_DATASET", context)
            self.assertIn("features_per_stage", context)

    def test_source_context_rejects_imaging_and_subject_tables(self):
        for filename in ("image.npy", "subjects.csv", "image.nii.gz"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "subject-table"):
                source_context({"source_files": [filename]})

    def test_extended_image_and_graph_wrappers_execute_generated_functions(self):
        for method, target, signature in (("deepn4", "preprocessor", "image"),
                                           ("fastsurfer", "par_net", "image"),
                                           ("fbnetgen", "graph_gen", "features")):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                host = nn.Module()
                setattr(host, target, nn.Linear(4, 4))
                host.candidate_modules = (target,)
                adapter = Path(temporary) / "adapter.py"
                adapter.write_text("def adapt(core, " + signature + "):\n    return core(" + signature + ") * 2\n")
                raw = getattr(host, target)
                image = torch.randn(2, 4)
                expected = raw(image)
                wrappers = bind_extended_adapters(host, method, adapter)
                self.assertTrue(torch.equal(getattr(host, target)(image), expected * 2))
                self.assertIs(wrappers[target].core, raw)

    def test_ernet_requires_and_binds_both_generated_functions(self):
        code = ("def adapt_extraction(core, image, if_train):\n    return core(image)\n"
                "def adapt_registration(core, fixed, moving):\n    return core(moving) + fixed\n")
        check_adapter(code, "ernet")
        with self.assertRaises(ValueError):
            check_adapter("def adapt_extraction(core, image, if_train):\n    return core(image)\n", "ernet")
        with tempfile.TemporaryDirectory() as temporary:
            host = nn.Module()
            host.ext_net, host.reg_net = nn.Linear(4, 4), nn.Linear(4, 4)
            host.reg_net.forward_grid = torch.ones(1, 2, 2, 2, 3)
            host.candidate_modules = ("ext_net", "reg_net")
            adapter = Path(temporary) / "adapter.py"
            adapter.write_text(code)
            wrappers = bind_extended_adapters(host, "ernet", adapter)
            image, fixed = torch.randn(2, 4, requires_grad=True), torch.zeros(2, 4)
            extracted = host.ext_net(image, True)
            result = host.reg_net(fixed, extracted)
            self.assertEqual(result.shape, image.shape)
            self.assertIs(host.reg_net.forward_grid, wrappers["reg_net"].core.forward_grid)
            for wrapper in wrappers.values():
                evidence = gradient_evidence(result.square().mean(), wrapper.parameters(), wrapper.last_inputs)
                self.assertTrue(evidence["passed"])

    def test_evaluator_preserves_update_count_when_adapter_fails_later(self):
        from scripts.evaluate_integration import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            adapter = folder / "adapter.py"
            adapter.write_text("def adapt(core, moving, fixed):\n    return core(moving, fixed)\n")
            args = SimpleNamespace(method="voxelmorph", source=folder, data=folder, adapter=adapter,
                                   output=folder / "evaluation.json", seed=17, gpu=0, steps=2, plans=None)

            def fail_after_update(arguments, report):
                report.update(runs=True, optimizer_updates=1)
                report["checks"]["gradient"] = False
                raise AdapterFailure("Measured failure after one update")

            with patch("unibrain_agent.candidate_integration_worker.validate", side_effect=fail_after_update):
                result = evaluate(args)
            self.assertEqual(result["status"], "adapter_failed")
            self.assertEqual(result["optimizer_updates"], 1)

    def test_invalid_code_is_rejected_before_the_gpu_worker(self):
        from scripts.evaluate_integration import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            adapter = folder / "adapter.py"
            adapter.write_text("def adapt(:\n")
            args = SimpleNamespace(method="voxelmorph", source=folder, data=folder, adapter=adapter,
                                   output=folder / "evaluation.json", seed=17, gpu=0, steps=2, plans=None)
            with patch("unibrain_agent.candidate_integration_worker.validate") as worker:
                result = evaluate(args)
            worker.assert_not_called()
            self.assertEqual(result["status"], "adapter_failed")
            self.assertIs(result["checks"]["contract"], False)

    def test_synthstrip_evaluator_preserves_worker_data_hash(self):
        from scripts.evaluate_integration import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            adapter = folder / "adapter.py"
            adapter.write_text("def adapt(core, image, if_train):\n    return [image], [image]\n")
            args = SimpleNamespace(method="synthstrip", source=folder, data=folder, adapter=adapter,
                                   output=folder / "evaluation.json", seed=17, gpu=0, steps=1, plans=None)

            def validate(arguments, report):
                report.update(status="passed", data={"sha256": "measured-training-array-hash"},
                              updates=[{"changed_parameter_tensors": {"ext_net": 1}}],
                              gradient_check={"registration_to_extraction_norm": 1., "input_gradient_norm": 1.})

            with patch("unibrain_agent.full_replacement_worker.validate", side_effect=validate), \
                 patch("unibrain_agent.candidate_integration_worker.source_hashes", return_value={}), \
                 patch("scripts.evaluate_integration.core_agreement", return_value={"passed": True, "spatial_passed": True}):
                report = evaluate(args)
            self.assertEqual(report["data"]["sha256"], "measured-training-array-hash")
            self.assertEqual(report["status"], "passed")


if __name__ == "__main__":
    unittest.main()
