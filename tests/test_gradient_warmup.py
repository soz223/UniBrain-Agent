import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from unibrain_agent.replacement_training import (affine_initialization_evidence, affine_encoder_evidence, check_gradient_paths,
    classification_gradient_paths, empty_gradient_warmup, gradient_statistics, record_warmup_update,
    require_confirmed_warmup, total_candidate_gradients)


class AffineFixture(nn.Module):
    def __init__(self, both=False):
        super().__init__()
        self.ext_net = nn.Linear(2, 2)
        self.reg_net = nn.Module()
        self.reg_net.affine = nn.Module()
        self.reg_net.affine.fc_loc = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU(.2), nn.Linear(128, 12))
        self.candidate_modules = ("ext_net", "reg_net") if both else ("ext_net",)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.eye(3, 4).flatten())

    @property
    def head(self):
        return self.reg_net.affine.fc_loc[2]

    def measure(self, detached=False, dead=False, direct_input=True, partial=False):
        self.zero_grad(set_to_none=True)
        image = torch.tensor([[.3, -.7], [1.2, .8]], requires_grad=True)
        extracted = nn.functional.linear(image, self.ext_net.weight, self.ext_net.bias.detach()) if partial else self.ext_net(image)
        theta = self.reg_net.affine.fc_loc(extracted.detach() if detached else extracted)
        score = theta[:, 3] * (0. if dead else 1.)
        score = score + (.1 * image[:, 0] if direct_input else 0.)
        classification = (score - 1.).square().mean()
        total = classification + self.ext_net(image).square().mean() + .2 * theta.square().mean()
        paths = classification_gradient_paths(self, classification, "synthstrip", True, image)
        total.backward()
        return {"protocol_version": 3, "classification_to_candidates": paths["candidates"],
                "classification_to_input": paths["input"], "input_required": True,
                "total_to_input": gradient_statistics([image.grad]),
                "total_to_candidates": total_candidate_gradients(self, "synthstrip")}


class GradientWarmupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.source_mock = patch("unibrain_agent.replacement_training.affine_encoder_evidence", return_value={
            "class_name": "encoder", "repository": "CPU fixture", "commit": "fixture", "source_file": "model.py",
            "source_sha256": "a" * 64, "head_object_verified": True, "forward_code_sha256": "b" * 64,
            "constructor_code_sha256": "c" * 64})
        self.source_mock.start()
        self.addCleanup(self.source_mock.stop)

    def test_zero_affine_head_allows_one_update_then_confirms_real_gradient(self):
        for both in (False, True):
            model = AffineFixture(both)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            state = empty_gradient_warmup()
            first = model.measure()
            before = copy.deepcopy(model.state_dict())
            check_gradient_paths(model, first, 0, 2, state)
            self.assertEqual(state["status"], "pending")
            self.assertFalse(first["classification_to_candidates"]["ext_net"]["nonzero"])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name])
            optimizer.step()
            record_warmup_update(model, state, first)
            self.assertTrue(first["warmup"]["initialization"]["weight_nonzero_after_update"])
            second = model.measure()
            check_gradient_paths(model, second, 1, 2, state)
            self.assertEqual(state["status"], "confirmed")
            self.assertEqual(state["confirmed_step"], 2)
            self.assertTrue(second["classification_to_candidates"]["ext_net"]["nonzero"])
            self.assertEqual(first["warmup"]["initialization"], second["warmup"]["initialization"])
            self.assertEqual(first["warmup"]["status"], "pending")
            optimizer.step()
            record_warmup_update(model, state, second)
            require_confirmed_warmup(state)

    def test_detached_missing_and_nonfinite_paths_cannot_warm_up(self):
        for problem in ("detached", "missing", "nan", "classification_input", "total_input", "total_candidate"):
            model = AffineFixture()
            if problem == "missing":
                model.ext_net.unused = nn.Parameter(torch.ones(1))
            record = model.measure(detached=problem == "detached")
            if problem == "nan":
                record["classification_to_candidates"]["ext_net"]["finite"] = False
            if problem in ("classification_input", "total_input", "total_candidate"):
                path = {"classification_input": record["classification_to_input"],
                        "total_input": record["total_to_input"],
                        "total_candidate": record["total_to_candidates"]["ext_net"]}[problem]
                path["nonzero"] = False
            state = empty_gradient_warmup()
            with self.assertRaisesRegex(ValueError, "Required gradient paths"):
                check_gradient_paths(model, record, 0, 2, state)
            self.assertEqual(state["status"], "not_needed")

    def test_persistently_dead_path_fails_second_check(self):
        model, state = AffineFixture(), empty_gradient_warmup()
        first = model.measure(dead=True)
        check_gradient_paths(model, first, 0, 2, state)
        torch.optim.SGD(model.parameters(), lr=.01).step()
        record_warmup_update(model, state, first)
        second = model.measure(dead=True)
        with self.assertRaisesRegex(ValueError, "warmup_unconfirmed"):
            check_gradient_paths(model, second, 1, 2, state)
        with self.assertRaisesRegex(ValueError, "pending"):
            require_confirmed_warmup(state)

    def test_single_update_budget_and_late_zero_gradient_are_rejected(self):
        for step, budget in ((0, 1), (1, 3)):
            model, state = AffineFixture(), empty_gradient_warmup()
            with self.assertRaisesRegex(ValueError, "Required gradient paths"):
                check_gradient_paths(model, model.measure(), step, budget, state)
            self.assertEqual(state["status"], "not_needed")

    def test_zero_head_requires_identity_bias_and_real_trainable_update(self):
        for change in ("bias", "frozen", "zero_gradient"):
            model = AffineFixture()
            if change == "bias":
                with torch.no_grad():
                    model.head.bias.zero_()
            if change == "frozen":
                model.head.requires_grad_(False)
            record = model.measure()
            if change == "zero_gradient":
                model.head.weight.grad.zero_()
            self.assertIsNone(affine_initialization_evidence(model))
            with self.assertRaisesRegex(ValueError, "Required gradient paths"):
                check_gradient_paths(model, record, 0, 2, empty_gradient_warmup())

    def test_pending_cannot_finish_or_claim_an_unperformed_update(self):
        model, state = AffineFixture(), empty_gradient_warmup()
        record = model.measure()
        check_gradient_paths(model, record, 0, 2, state)
        with self.assertRaisesRegex(ValueError, "pending"):
            require_confirmed_warmup(state)
        with self.assertRaisesRegex(ValueError, "did not change"):
            record_warmup_update(model, state, record)

    def test_generated_registration_reports_its_actual_registered_head(self):
        from unibrain_agent.candidate_integration import GeneratedCore
        model = AffineFixture()
        code = ("def adapt_extraction(core, image, if_train):\n    return core(image, if_train)\n"
                "def adapt_registration(core, fixed, moving):\n    return core(fixed, moving)\n")
        model.reg_net = GeneratedCore(model.reg_net, code, "ernet", "adapt_registration")
        model.measure()
        evidence = affine_initialization_evidence(model)
        self.assertEqual(evidence["head_path"], "reg_net.core.affine.fc_loc.2")
        self.assertIs(model.get_submodule(evidence["head_path"]), model.head)

    def test_partial_detach_after_warmup_cannot_confirm_or_remain_confirmed(self):
        for already_confirmed in (False, True):
            model, state = AffineFixture(), empty_gradient_warmup()
            first = model.measure()
            check_gradient_paths(model, first, 0, 3, state)
            torch.optim.SGD(model.parameters(), lr=.01).step()
            record_warmup_update(model, state, first)
            if already_confirmed:
                check_gradient_paths(model, model.measure(), 1, 3, state)
            record = model.measure(partial=True)
            self.assertEqual(record["classification_to_candidates"]["ext_net"]["observed_tensors"], 1)
            self.assertTrue(record["total_to_candidates"]["ext_net"]["nonzero"])
            with self.assertRaisesRegex(ValueError, "incomplete_warmup_path"):
                check_gradient_paths(model, record, 2 if already_confirmed else 1, 3, state)


class AffineSourceTests(unittest.TestCase):
    def test_shape_compatible_untrusted_encoder_is_rejected(self):
        model = AffineFixture()
        self.assertIsNone(affine_encoder_evidence(model.reg_net.affine, model.head))

    def test_pinned_encoder_binds_actual_head_and_rejects_modified_forward(self):
        from unibrain_agent.paths import SOURCES
        from unibrain_agent.sources import load
        if not (SOURCES / "reg_ernet/model.py").is_file():
            self.skipTest("Pinned ERNet source required")
        scope = load("reg_ernet/model.py", {"conv_block", "encoder"})
        encoder = scope["encoder"]()
        head = encoder.fc_loc[2]
        evidence = affine_encoder_evidence(encoder, head)
        self.assertTrue(evidence["head_object_verified"])
        self.assertEqual(evidence["class_name"], "encoder")
        self.assertIsNone(affine_encoder_evidence(encoder, nn.Linear(128, 12)))
        type(encoder).forward = lambda self, image: self.fc_loc(image)
        self.assertIsNone(affine_encoder_evidence(encoder, head))


if __name__ == "__main__":
    unittest.main()
