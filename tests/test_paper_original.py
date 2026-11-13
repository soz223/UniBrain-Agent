import unittest

import torch
from torch.nn import functional as F

from unibrain_agent.paper_original import perturb_graph
from unibrain_agent.full_connectivity import transfer


class PaperOriginalTests(unittest.TestCase):
    def test_edge_augmentation_preserves_retained_weights_and_gradients(self):
        torch.manual_seed(17)
        graph = torch.linspace(-.8, .8, 100).reshape(1, 10, 10).requires_grad_()
        output = perturb_graph(graph, .1)
        unchanged = (output == graph) & (graph > 0)
        self.assertEqual(int(unchanged.sum()), 45)
        self.assertEqual(int((output == 1).sum()), 5)
        self.assertTrue(bool((output >= 0).all()))
        output.sum().backward()
        self.assertTrue(torch.equal(graph.grad[unchanged], torch.ones(45)))
        self.assertEqual(int(graph.grad.count_nonzero()), 45)

    def test_template_target_has_affine_gradient(self):
        labels = torch.zeros(1, 1, 4, 4, 4)
        labels[:, :, 1:3] = 1
        affine = torch.eye(3, 4).unsqueeze(0)
        affine[:, 2, 3] = .1
        affine.requires_grad_()
        grid = F.affine_grid(affine, labels.shape, align_corners=True)
        target = transfer(labels, torch.arange(4), grid)
        logits = torch.randn_like(target, requires_grad=True)
        F.cross_entropy(logits, target).backward()
        self.assertTrue(torch.allclose(target.sum(1), torch.ones_like(labels[:, 0])))
        self.assertTrue(bool(affine.grad.isfinite().all()))
        self.assertGreater(float(affine.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
