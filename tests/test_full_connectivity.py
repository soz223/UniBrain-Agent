import unittest

import torch
from torch.nn import functional as F

from unibrain_agent.full_connectivity import transfer


class ConnectivityTests(unittest.TestCase):
    def test_transfer_strategies_and_nearest_cut(self):
        labels = torch.zeros(1, 1, 4, 4, 4)
        labels[..., 2:] = 1
        for strategy, nearest in (("soft", False), ("straight_through", False), ("gumbel_softmax", False), ("soft", True)):
            torch.manual_seed(4)
            theta = torch.eye(3, 4)[None].clone().requires_grad_()
            grid = F.affine_grid(theta, labels.shape, align_corners=True) + .1
            probabilities = transfer(labels, torch.arange(2), grid, strategy, True, nearest)
            self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones_like(labels[:, 0])))
            gradient = torch.autograd.grad(probabilities[:, 1].sum(), theta)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(bool(gradient.abs().sum() > 0), not nearest)


if __name__ == "__main__":
    unittest.main()
