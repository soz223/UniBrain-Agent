import copy
import unittest

import torch

from unibrain_agent.taskforce import TASKS
from unibrain_agent.taskforce_paper import PaperTaskForce, full_gradient_gram, make_team, policy_parameters


def task_losses(parameter, step=0):
    return {task: (parameter - (.1 * step + i * .2)).square().sum() + .01 for i, task in enumerate(TASKS)}


def advance(team, parameter, optimizer, step):
    optimizer.zero_grad()
    total, _ = team.combine(task_losses(parameter, step), [parameter])
    total.backward()
    optimizer.step()
    return team.after_model_step()


class PaperTaskForceTests(unittest.TestCase):
    def test_gram_includes_every_parameter_and_preserves_model_gradients(self):
        x = torch.nn.Parameter(torch.tensor([1., 2.]))
        y = torch.nn.Parameter(torch.tensor([3.]))
        unused = torch.nn.Parameter(torch.ones(3))
        losses = dict(zip(TASKS, [x.square().sum(), (x.sum() + y.sum()).square(), y.square().sum(), torch.tensor(2.)]))
        gradients = torch.tensor([[2., 4., 0.], [12., 12., 12.], [0., 0., 6.], [0., 0., 0.]], dtype=torch.float64)
        x.grad = torch.ones_like(x)
        gram = full_gradient_gram(losses, [x, y, unused])
        torch.testing.assert_close(gram, gradients @ gradients.T)
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        self.assertIsNone(y.grad)

    def test_applied_update_is_the_convex_normalized_gradient(self):
        x = torch.nn.Parameter(torch.tensor([.5, -.7]))
        losses = task_losses(x)
        gradients = torch.stack([2 * (x.detach() - i * .2) for i in range(4)])
        team = make_team((1., .1, 1., 1.))
        total, record = team.combine(losses, [x])
        shares = torch.tensor([record['actions'][task] for task in TASKS])
        expected = (shares[:, None] * gradients / gradients.norm(dim=1)[:, None]).sum(0)
        before = x.detach().clone()
        total.backward()
        torch.testing.assert_close(x.grad, expected)
        torch.optim.SGD([x], lr=.02).step()
        torch.testing.assert_close(x.detach(), before - .02 * expected)
        self.assertAlmostEqual(float(shares.sum()), 1., places=6)
        self.assertTrue((shares >= 0).all())
        self.assertEqual(record['gradient_parameter_elements'], 2)
        self.assertEqual(team.updates, 0)
        self.assertTrue(team.after_model_step()['model_step_completed'])

    def test_decentralized_actor_does_not_read_other_observations(self):
        team = PaperTaskForce()
        for actor in team.actors:
            torch.nn.init.constant_(actor[-1].weight, .1)
        observation = torch.ones(4, 5)
        before = team.action(observation, explore=False)
        observation[2] += 2
        after = team.action(observation, explore=False)
        torch.testing.assert_close(before[[0, 1, 3]], after[[0, 1, 3]])
        self.assertNotEqual(float(before[2]), float(after[2]))

    def test_reward_and_replay_use_the_previous_raw_action(self):
        team = PaperTaskForce(batch_size=1)
        x = torch.nn.Parameter(torch.tensor([.5, -.7]))
        optimizer = torch.optim.SGD([x], lr=.02)
        first = advance(team, x, optimizer, 0)
        losses = task_losses(x, 1)
        total, record = team.combine(losses, [x])
        previous_loss = torch.tensor(list(first['task_losses'].values()))
        current_loss = torch.tensor([float(losses[t].detach()) for t in TASKS])
        shares = torch.tensor(list(first['actions'].values()), dtype=torch.float64)
        penalty = shares @ torch.tensor(first['gram'], dtype=torch.float64) @ shares
        self.assertAlmostEqual(record['reward'], float((previous_loss.log1p() - current_loss.log1p()).sum() - .001 * penalty))
        torch.testing.assert_close(team.replay[0][1], torch.tensor(list(first['action_logits'].values())))
        self.assertEqual(team.updates, 0)
        with self.assertRaisesRegex(RuntimeError, 'follow'):
            team.update()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        result = team.after_model_step()
        self.assertEqual(result['policy_updates'], 1)
        self.assertEqual(result['actor_optimizer_steps'], 4)
        self.assertEqual(result['model_steps'], 2)

    def test_all_actors_and_centralized_critics_learn_and_resume_exactly(self):
        team = PaperTaskForce(seed=4, batch_size=2)
        initial = copy.deepcopy(team.state_dict())
        x = torch.nn.Parameter(torch.tensor([.5, -.7]))
        optimizer = torch.optim.SGD([x], lr=.02)
        critic_inputs = []
        handle = team.critics[0].register_forward_pre_hook(lambda module, args: critic_inputs.append(args[0].detach().clone()))
        for step in range(5):
            advance(team, x, optimizer, step)
        handle.remove()
        self.assertTrue(all(value.shape == (2, 24) for value in critic_inputs))
        state = team.state_dict()
        for group in ('actors', 'critics', 'target_actors', 'target_critics'):
            for before, after in zip(initial[group], state[group]):
                self.assertTrue(any(not torch.equal(before[key], after[key]) for key in before))
        restored = PaperTaskForce(seed=99, batch_size=2)
        restored.load_state_dict(state)
        y = torch.nn.Parameter(x.detach().clone())
        other_optimizer = torch.optim.SGD([y], lr=.02)
        for step in range(5, 8):
            self.assertEqual(advance(team, x, optimizer, step), advance(restored, y, other_optimizer, step))
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_default_paper_settings_and_full_parameter_selection(self):
        team = PaperTaskForce()
        self.assertEqual((team.batch_size, team.replay.maxlen), (64, 100000))
        self.assertEqual(team.actors[0][0].out_features, 64)
        self.assertEqual(team.actor_optimizers[0].param_groups[0]['lr'], 5e-4)
        self.assertEqual(team.critic_optimizers[0].param_groups[0]['lr'], 5e-4)
        team.action(torch.ones(4, 5))
        self.assertAlmostEqual(team.noise_scale, .3)
        team.action_steps = 10000
        team.action(torch.ones(4, 5))
        self.assertAlmostEqual(team.noise_scale, .05)
        model = torch.nn.Linear(100, 10)
        parameters, names = policy_parameters(model, probe_budget=1)
        self.assertEqual(sum(p.numel() for p in parameters), 1010)
        self.assertEqual(names, ['weight', 'bias'])


if __name__ == '__main__':
    unittest.main()
