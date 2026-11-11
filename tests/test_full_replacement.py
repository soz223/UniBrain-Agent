import unittest
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from unibrain_agent.full_replacement import check_code, inject_hard_mask, run_loop
from unibrain_agent.full_replacement_worker import selected_device


CODE = """def adapt(core, image, if_train):
    distance = core(F.pad(image, (16, 16, 16, 16, 16, 16)))
    mask = torch.sigmoid(-distance[:, :, 16:-16, 16:-16, 16:-16])
    if not if_train:
        mask = (mask >= 0.5).to(image.dtype)
    return [image * mask], [mask]
"""


class FullReplacementTests(unittest.TestCase):
    def loop_with_reports(self, reports):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            candidate = root / "freesurfer_synthstrip/mri_synthstrip/mri_synthstrip"
            candidate.parent.mkdir(parents=True)
            candidate.write_text("# Test source\n", encoding="utf-8")
            stack.enter_context(patch("unibrain_agent.native.verify_original", return_value=[]))
            stack.enter_context(patch("unibrain_agent.sources.verify", return_value=[]))
            stack.enter_context(patch("unibrain_agent.paths.SOURCES", root))
            stack.enter_context(patch("unibrain_agent.upstream.synthstrip_core", return_value=SimpleNamespace(evidence=[])))
            stack.enter_context(patch("unibrain_agent.providers.provider_settings",
                                      return_value=SimpleNamespace(snapshot=lambda _: {"provider": "codex"})))
            invoke = stack.enter_context(patch("unibrain_agent.providers.invoke",
                                               return_value=({"code": CODE, "rationale": "Test proposal"}, None)))
            pending = iter(reports)

            def worker(command, **kwargs):
                self.assertEqual(kwargs["cwd"], Path(__file__).resolve().parents[1])
                output = Path(command[command.index("--output") + 1])
                report = next(pending)
                (output / "validation.json").write_text(json.dumps(report), encoding="utf-8")
                return SimpleNamespace(returncode=0 if report["status"] == "passed" else 1)

            stack.enter_context(patch("unibrain_agent.full_replacement.subprocess.run", side_effect=worker))
            result = run_loop(root / "run", root / "source", root / "data", Path("python"), "codex")
            return result, invoke.call_args_list

    def test_measured_failure_is_returned_to_repair_agent(self):
        result, calls = self.loop_with_reports([
            {"status": "adapter_failed", "error": "Measured missing gradient", "updates": []},
            {"status": "passed", "updates": [{"step": 1}, {"step": 2}]}])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(calls), 2)
        self.assertIn("Measured missing gradient", calls[1].args[1])

    def test_no_training_retry_after_an_update(self):
        result, calls = self.loop_with_reports([
            {"status": "adapter_failed", "updates": [{"step": 1}], "error": "Late failure"}])
        self.assertEqual(result["status"], "training_failed")
        self.assertEqual(len(calls), 1)

    def test_busy_gpu_does_not_trigger_code_repair(self):
        result, calls = self.loop_with_reports([{"status": "infrastructure_failed", "error": "GPU busy"}])
        self.assertEqual(result["status"], "infrastructure_failed")
        self.assertEqual(len(calls), 1)

    def test_respects_scheduler_visible_devices(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,4"}):
            self.assertEqual(selected_device(0), "2")
            self.assertEqual(selected_device(1), "4")
            with self.assertRaises(ValueError):
                selected_device(2)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaises(ValueError):
                selected_device(0)

    def test_hard_mask_fixture_really_disconnects_parameters(self):
        for code, connected in ((CODE, True), (inject_hard_mask(CODE), False)):
            scope = {"torch": torch, "F": F}
            exec(compile(check_code(code), "adapter.py", "exec"), scope)
            core = torch.nn.Conv3d(1, 1, 1)
            image = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
            extracted, masks = scope["adapt"](core, image, True)
            gradient = torch.autograd.grad(extracted[0].sum(), core.weight, allow_unused=True)[0]
            self.assertEqual(gradient is not None, connected)
            self.assertEqual(masks[0].shape, image.shape)

    def test_refuse_filesystem_and_host_mutation(self):
        for code in ("import os\n" + CODE, CODE.replace("mask = torch.sigmoid", "core.weight = torch.sigmoid"),
                     CODE.replace("torch.sigmoid", "__import__"), CODE.replace("torch.sigmoid", "torch.load")):
            with self.assertRaises(ValueError):
                check_code(code)

    def test_inference_mask_preserves_interface(self):
        scope = {"torch": torch, "F": F}
        exec(compile(check_code(CODE), "adapter.py", "exec"), scope)
        image = torch.randn(1, 1, 4, 4, 4)
        extracted, masks = scope["adapt"](torch.nn.Identity(), image, False)
        self.assertTrue(torch.all((masks[0] == 0) | (masks[0] == 1)))
        self.assertTrue(torch.equal(extracted[0], image * masks[0]))


if __name__ == "__main__":
    unittest.main()
