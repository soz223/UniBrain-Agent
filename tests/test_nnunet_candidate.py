import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from unibrain_agent.full_candidates import planned_unet, lapirn
from unibrain_agent.paths import SOURCES


class NnunetCandidateTests(unittest.TestCase):
    @unittest.skipUnless((SOURCES / "reg_lapirn/Code/miccai2020_model_stage.py").is_file(), "Pinned LapIRN source required")
    def test_lapirn_device_buffers_do_not_change_official_checkpoint_keys(self):
        model = lapirn()
        state = model.state_dict()
        self.assertFalse(any(key.endswith("grid_1") for key in state))
        self.assertEqual(len([name for name, _ in model.named_buffers() if name.endswith("grid_1")]), 3)
        restored = lapirn()
        result = restored.load_state_dict(state, strict=True)
        self.assertFalse(result.missing_keys)
        self.assertFalse(result.unexpected_keys)

    @unittest.skipUnless((SOURCES / "dynamic-network-architectures/dynamic_network_architectures/architectures/unet.py").is_file(), "Pinned architecture source required")
    def test_full_width_core_96_cubed_with_planner_operator_names(self):
        # This is a structural fixture, not a dataset-generated research plan.
        kwargs = {"n_stages": 5, "features_per_stage": [32, 64, 128, 256, 320],
                  "conv_op": "torch.nn.modules.conv.Conv3d", "kernel_sizes": [[3, 3, 3]] * 5,
                  "strides": [[1, 1, 1]] + [[2, 2, 2]] * 4, "n_conv_per_stage": [2] * 5,
                  "n_conv_per_stage_decoder": [2] * 4, "conv_bias": True,
                  "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d", "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                  "dropout_op": None, "dropout_op_kwargs": None,
                  "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True}}
        fixture = {"configurations": {"3d_fullres": {"patch_size": [96, 96, 96], "architecture": {
            "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": kwargs, "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structural_fixture.json"
            path.write_text(json.dumps(fixture))
            torch.set_num_threads(2)
            torch.manual_seed(7)
            model = planned_unet(path)
            self.assertGreater(sum(p.numel() for p in model.parameters()), 10_000_000)
            image = torch.randn(1, 1, 96, 96, 96, requires_grad=True)
            output = model(image)
            self.assertEqual([tuple(value.shape) for value in output], [(1, 4, size, size, size) for size in (96, 48, 24, 12)])
            loss = sum(F.cross_entropy(value, torch.zeros_like(value[:, 0], dtype=torch.long)) for value in output)
            loss.backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertGreater(image.grad.abs().sum().item(), 0)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
            changed = next(p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad))
            before = changed.detach().clone()
            torch.optim.SGD(model.parameters(), lr=1e-4).step()
            self.assertFalse(torch.equal(changed, before))


if __name__ == "__main__":
    unittest.main()
