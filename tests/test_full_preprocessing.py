import gc
import unittest

import torch

from unibrain_agent.full_preprocessing import REPOSITORIES, build_preprocessor
from unibrain_agent.paths import SOURCES


class FullPreprocessingTests(unittest.TestCase):
    @unittest.skipUnless(all((SOURCES / name).exists() for name in REPOSITORIES.values()), "Pinned sources required")
    def test_original_parameter_counts_and_shape_contract(self):
        counts = {"redwgan": 2326273, "deepn4": 19328418, "deepnormalize": 4770625, "harmonizing_flows": 816594}
        for method, count in counts.items():
            model = build_preprocessor(method)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            with self.assertRaisesRegex(ValueError, "96,96,96"):
                model(torch.zeros(1, 1, 16, 16, 16))
            del model

    @unittest.skipUnless(all((SOURCES / name).exists() for name in REPOSITORIES.values()), "Pinned sources required")
    def test_original_widths_full_volume_gradients_and_updates(self):
        torch.set_num_threads(2)
        for method in REPOSITORIES:
            with self.subTest(method=method):
                torch.manual_seed(17)
                model = build_preprocessor(method)
                image = torch.rand(1, 1, 96, 96, 96, requires_grad=True)
                result = model(image)
                self.assertEqual(tuple(result.shape), tuple(image.shape))
                self.assertTrue(torch.isfinite(result).all())
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(image.grad).all())
                self.assertGreater(image.grad.abs().sum().item(), 0)
                gradients = [parameter.grad for parameter in model.parameters()]
                self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
                self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0)
                if method == "harmonizing_flows":
                    self.assertTrue((image.grad.abs().sum(dim=(0, 1, 3, 4)) > 0).all())
                probe = next(parameter for parameter in model.parameters() if parameter.grad.abs().sum() > 0)
                before = probe.detach().clone()
                torch.optim.SGD(model.parameters(), lr=1e-3).step()
                self.assertFalse(torch.equal(before, probe))
                self.assertFalse(model.architecture["compact"])
                del result, image, model, gradients, probe, before
                gc.collect()

    def test_unknown_component_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_preprocessor("unknown")


if __name__ == "__main__":
    unittest.main()
