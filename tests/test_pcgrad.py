import unittest

import torch

from unibrain_agent.pcgrad import project_weights
from unibrain_agent.taskforce import TASKS


class PCGradTest(unittest.TestCase):
    def test_matches_explicit_projections_with_unused_and_conflicting_gradients(self):
        parameter = torch.ones(3, dtype=torch.float64, requires_grad=True)
        matrix = parameter.new_tensor([[2, 1, 0], [-3, 1, 1], [0, 0, 0], [1, -3, 2]])
        losses = dict(zip(TASKS, matrix @ parameter))
        base = dict(zip(TASKS, (1., .1, 1., 1.)))
        vectors = matrix * parameter.new_tensor(list(base.values()))[:, None]
        for step in (0, 11):
            order = torch.Generator().manual_seed((17 << 32) + step)
            expected = torch.zeros_like(parameter)
            for i, vector in enumerate(vectors):
                projected = vector.clone()
                for j in torch.randperm(len(TASKS), generator=order):
                    other = vectors[j]
                    if i != j and other.dot(other) > 0 and projected.dot(other) < 0:
                        projected -= projected.dot(other) / other.dot(other) * other
                expected += projected
            weights = project_weights(losses, base, [parameter], 17, step)
            loss = sum(weights[name] * losses[name] for name in TASKS)
            actual, = torch.autograd.grad(loss, parameter, retain_graph=True)
            torch.testing.assert_close(actual, expected)
