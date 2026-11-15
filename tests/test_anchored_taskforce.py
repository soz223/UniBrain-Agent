import copy
import tempfile
import unittest
from pathlib import Path

import torch

from unibrain_agent.anchored_taskforce import AnchoredTaskForce, matched_forward, shield_coefficients
from unibrain_agent.replacement_training import rng_state
from unibrain_agent.taskforce import TASKS
from unibrain_agent.training_stability import recover


def losses(x, offset=0.):
    return {task: (x - (i + offset)).square().sum() + .01 for i, task in enumerate(TASKS)}


class AnchoredTests(unittest.TestCase):
    def test_reference_coefficients_and_conflict_shield(self):
        gradients = torch.tensor([[1., 0.], [-3., .1], [.2, 1.], [0., -1.]], dtype=torch.float64)
        gram = gradients @ gradients.T
        base = torch.tensor([1., .1, 1., 1.], dtype=torch.float64)
        applied, _ = shield_coefficients(base, base * torch.tensor([.9, 1.1, .9, 1.1]), gram, .1)
        reference = gram @ base
        floor = torch.where(reference > 0, reference * .5, reference)
        self.assertTrue(torch.all(gram @ applied >= floor - 1e-10))
        self.assertLessEqual(float((gradients.T @ (applied - base)).norm()), .1 * float((gradients.T @ base).norm()) + 1e-10)

    def test_reward_is_matched_and_policy_updates(self):
        team = AnchoredTaskForce((1., .1, 1., 1.), batch_size=2)
        x = torch.nn.Parameter(torch.tensor([.5, .6]))
        optimizer = torch.optim.SGD([x], lr=.01)
        before_policy = copy.deepcopy(team.critics[0].state_dict())
        for step in range(8):
            offset = step * 10.
            optimizer.zero_grad()
            before = losses(x, offset)
            total, record = team.combine(before, [x])
            total.backward()
            optimizer.step()
            after = losses(x, offset)
            record = team.after_model_step(after)
            expected = sum(weight * float(before[task].detach().log1p() - after[task].detach().log1p())
                           for task, weight in zip(TASKS, team.base)) / sum(team.base)
            self.assertAlmostEqual(record['reward'], max(-1., min(1., expected)), places=6)
            for task, base in zip(TASKS, team.base):
                self.assertGreaterEqual(record['weights'][task], base * .9 - 1e-8)
                self.assertLessEqual(record['weights'][task], base * 1.1 + 1e-8)
        self.assertGreater(team.updates, 0)
        self.assertTrue(any(not torch.equal(before_policy[k], v) for k, v in team.critics[0].state_dict().items()))
        restored = AnchoredTaskForce(team.base, batch_size=2)
        restored.load_state_dict(team.state_dict())
        self.assertEqual(restored.matched_transition[2], team.matched_transition[2])

    def test_matched_forward_replays_noise_without_advancing_buffers_or_rng(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.5))
        model.train()
        x = torch.randn(4, 3)
        before = rng_state()
        first = model(x)
        buffers = [value.clone() for value in model.buffers()]
        current = torch.get_rng_state().clone()
        second = matched_forward(model, before, lambda: model(x))
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(current, torch.get_rng_state()))
        for buffer, value in zip(model.buffers(), buffers):
            torch.testing.assert_close(buffer, value)

    def test_recovery_resets_policy_and_radius_without_lowering_lr(self):
        team = AnchoredTaskForce((1., .1, 1., 1.))
        with torch.no_grad():
            team.actors[0][-1].bias.fill_(4.)
        with tempfile.TemporaryDirectory() as root:
            source, destination = Path(root) / 'stable.pt', Path(root) / 'recovery.pt'
            state = {'checkpoint_kind': 'resume', 'identity': {'taskforce_gradient_normalization': 'anchored', 'seed': 17},
                     'model': {'x': torch.tensor([3.])}, 'optimizer': {'param_groups': [{'lr': 1e-5}], 'state': {}},
                     'progress': {'steps': 10}, 'coordination': team.state_dict()}
            torch.save(state, source)
            self.assertEqual(recover(source, destination, 1), [1e-5])
            result = torch.load(destination, weights_only=True)
            self.assertEqual(result['coordination']['trust_radius'], .05)
            self.assertEqual(float(result['coordination']['actors'][0]['4.bias']), 0.)
            torch.testing.assert_close(result['model']['x'], state['model']['x'])
