import unittest

import torch
from torch import nn
from torch.nn import functional as F

from unibrain_agent.full_candidates import compose_grids, voxelmorph, lapirn, FullRegistration, planned_unet
from unibrain_agent.paths import SOURCES


class IdentityAffine(nn.Module):
    def forward(self, fixed, moving):
        theta = torch.eye(3, 4, device=moving.device)[None]
        return [moving] * 5, [theta] * 5, [theta] * 5


class FullCandidateTests(unittest.TestCase):
    def test_identity_composition_and_translation_units(self):
        image = torch.randn(1, 1, 8, 8, 8)
        theta = torch.eye(3, 4)[None]
        positive = torch.zeros(1, 3, 8, 8, 8)
        positive[:, 0] = 2 / 7
        grid, inverse = compose_grids(image, theta, theta, positive, -positive)
        identity = F.affine_grid(theta, image.shape, align_corners=True)
        self.assertTrue(torch.allclose(grid[..., 0] - identity[..., 0], torch.full_like(grid[..., 0], 2 / 7)))
        self.assertTrue(torch.allclose(inverse[..., 0] - identity[..., 0], torch.full_like(grid[..., 0], -2 / 7)))

    def test_unet_requires_actual_plan(self):
        with self.assertRaisesRegex(ValueError, "plans"):
            planned_unet(None)

    @unittest.skipUnless((SOURCES / "reg_voxelmorph_master/voxelmorph/torch/networks.py").exists(), "Pinned sources required")
    def test_full_voxelmorph_forward_and_backward(self):
        self.check_registration("voxelmorph", 301411)

    @unittest.skipUnless((SOURCES / "reg_lapirn/Code/miccai2020_model_stage.py").exists(), "Pinned sources required")
    def test_full_lapirn_forward_and_backward(self):
        self.check_registration("lapirn", 923748)

    def check_registration(self, method, count):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model = FullRegistration(method, IdentityAffine())
        self.assertEqual(sum(p.numel() for p in model.core.parameters()), count)
        moving = torch.randn(1, 1, 96, 96, 96, requires_grad=True)
        fixed = torch.randn_like(moving)
        warped, _, _ = model(fixed, moving)
        self.assertEqual(warped[-1].shape, moving.shape)
        self.assertEqual(model.inverse_grid.shape, (1, 96, 96, 96, 3))
        loss = (warped[-1] - fixed).square().mean()
        loss.backward()
        gradients = [p.grad for p in model.core.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertGreater(moving.grad.abs().sum().item(), 0)
        before = [p.detach().clone() for p in model.core.parameters()]
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.core.parameters())))


if __name__ == "__main__":
    unittest.main()
