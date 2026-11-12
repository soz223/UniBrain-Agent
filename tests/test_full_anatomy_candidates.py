import copy
import unittest

import torch
from torch import nn

from unibrain_agent.full_anatomy_candidates import build_anatomy_modules, memory_forward, FullFastSurferParcellation, FullERNetRegistration
from unibrain_agent.paths import SOURCES


class AnatomyCandidateContracts(unittest.TestCase):
    def test_activation_recomputation_preserves_gradients_and_batchnorm_statistics(self):
        torch.manual_seed(5)
        native = nn.Sequential(nn.Conv3d(1, 3, 3, padding=1), nn.BatchNorm3d(3), nn.ReLU(), nn.Conv3d(3, 2, 1))
        replayed = copy.deepcopy(native)
        first = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        expected, actual = native(first), memory_forward(replayed, second)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(first.grad, second.grad)
        for direct, recomputed in zip(native.parameters(), replayed.parameters()):
            torch.testing.assert_close(direct.grad, recomputed.grad)
        for direct, recomputed in zip(native.buffers(), replayed.buffers()):
            torch.testing.assert_close(direct, recomputed)

    @unittest.skipUnless((SOURCES / "FastSurfer/FastSurferCNN/models/networks.py").is_file(), "Pinned FastSurfer source required")
    def test_all_96_slices_and_neighbor_boundaries_are_preserved(self):
        class CenterSlice(nn.Module):
            def __init__(self):
                super().__init__()
                self.observed = []

            def forward(self, image):
                self.observed.extend(image[:, 3, 0, 0].tolist())
                return image[:, 3:4]

        candidate = FullFastSurferParcellation(slice_batch=7)
        core = CenterSlice()
        candidate.core = core
        image = torch.arange(96).float().reshape(1, 1, 96, 1, 1).expand(1, 1, 96, 96, 96)
        with torch.no_grad():
            result = candidate(image)
        self.assertEqual(core.observed, list(range(96)))
        torch.testing.assert_close(result, image)

    @unittest.skipUnless((SOURCES / "nobrainer/nobrainer/models/highresnet.py").is_file(), "Pinned nobrainer source required")
    def test_full_highres_widths_and_block_counts(self):
        modules, architecture = build_anatomy_modules("nobrainer")
        core = modules["par_net"].core
        self.assertEqual(core.init_conv.out_channels, 16)
        self.assertEqual(core.stage2_proj.out_channels, 32)
        self.assertEqual(core.stage3_proj.out_channels, 64)
        self.assertEqual([len(core.stage1), len(core.stage2), len(core.stage3)], [3, 3, 3])
        self.assertEqual(core.classifier[-1].out_channels, 117)
        self.assertFalse(architecture["compact"])

    @unittest.skipUnless((SOURCES / "reg_ernet/model.py").is_file(), "Pinned ERNet source required")
    def test_full_ernet_counts_and_heads(self):
        modules, architecture = build_anatomy_modules("ernet")
        self.assertEqual(modules["ext_net"].stage, 5)
        self.assertEqual(modules["ext_net"].gamma, 10)
        self.assertEqual(modules["reg_net"].stage, 5)
        self.assertEqual(modules["reg_net"].affine.fc_loc[0].in_features, 512 * 3 ** 3)
        self.assertEqual(modules["reg_net"].affine.fc_loc[0].out_features, 128)
        self.assertEqual(set(architecture["parameters"]), {"ext_net", "reg_net"})

    @unittest.skipUnless((SOURCES / "reg_ernet/model.py").is_file(), "Pinned ERNet source required")
    def test_nonidentity_affines_compose_and_invert_in_original_order(self):
        class KnownAffine(nn.Module):
            def forward(self, value):
                return value.new_tensor([[[.99, 0., 0., .02], [0., 1., 0., -.01], [0., 0., 1.01, 0.]]])
        module = FullERNetRegistration()
        module.affine = KnownAffine()
        image = torch.zeros(1, 1, 96, 96, 96)
        with torch.no_grad():
            _, affines, inverses = module(image, image)
        matrix = torch.cat((module.affine(image), torch.tensor([[[0., 0., 0., 1.]]])), dim=1)
        expected = torch.linalg.matrix_power(matrix, 5)
        torch.testing.assert_close(affines[-1], expected[:, :3])
        torch.testing.assert_close(inverses[-1], torch.linalg.inv(expected)[:, :3])


if __name__ == "__main__":
    unittest.main()
