import copy
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from unibrain_agent import replacement_training
from unibrain_agent.anchored_taskforce import AnchoredTaskForce
from unibrain_agent.taskforce import TASKS
from unibrain_agent.training_stability import recover
from unibrain_agent.transfer_reward import PREVIOUS, reference_index, reference_losses, step_with_reference, upgrade_reward_identity


class TransferRewardTests(unittest.TestCase):
    def test_shadow_update_matches_independent_adam_and_preserves_agent_state(self):
        from unibrain_agent.transfer_reward import step_against_reference

        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        model.weight.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected, expected_opt = copy.deepcopy((model, optimizer))
        baseline, baseline_opt = copy.deepcopy((model, optimizer))
        expected.weight.sum().backward()
        expected_opt.step()
        (-baseline.weight.sum()).backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
        baseline_opt.step()
        model.weight.sum().backward()
        def measure(network, sample, templates):
            return network.weight.detach().clone()
        with patch('unibrain_agent.transfer_reward.reference_losses', side_effect=measure):
            base_score, agent_score = step_against_reference(model, optimizer, -model.weight.sum(), None, None)
        torch.testing.assert_close(base_score, baseline.weight)
        torch.testing.assert_close(agent_score, expected.weight)
        for key, value in expected_opt.state[expected.weight].items():
            torch.testing.assert_close(optimizer.state[model.weight][key], value)

    def test_zero_parameter_step_has_zero_reference_improvement_after_batchnorm_update(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(3)

            def forward(self, ref, image, tissue, atlas, if_train):
                value = self.bn(image).square().mean()
                return (value, None, [value])

        model = Model().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        model(None, torch.randn(4, 3) + 10, None, None, True)[0].backward()
        buffers = [v.clone() for v in model.buffers()]
        with patch.dict('sys.modules', {'model': SimpleNamespace(GCC=lambda: lambda x, y: x)}), \
             patch.object(replacement_training, 'original_losses', side_effect=lambda o, b, l: {t: o[0] for t in TASKS}):
            before, after = step_with_reference(model, optimizer,
                (torch.ones(4, 3), None, None, None, None), (None, None, None))
        for task in TASKS:
            torch.testing.assert_close(before[task], after[task], rtol=0, atol=0)
        for old, new in zip(buffers, model.buffers()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertTrue(model.training)

    def test_reference_forward_preserves_modes_buffers_and_random_state(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(3)
                self.register_buffer('counter', torch.zeros(1))

            def forward(self, ref, image, tissue, atlas, if_train):
                assert not self.training and not if_train
                self.counter.add_(1)
                value = self.bn(image).square().mean() + torch.rand(())
                return (value, None, [value])

        model = Model().train()
        model.bn.eval()
        image = torch.ones(2, 3)
        before = torch.get_rng_state().clone()
        with patch.dict('sys.modules', {'model': SimpleNamespace(GCC=lambda: lambda x, y: x)}), \
             patch.object(replacement_training, 'original_losses', side_effect=lambda o, b, l: {t: o[0] for t in TASKS}):
            values = reference_losses(model, (image, None, None, None, None), (None, None, None))
        self.assertTrue(model.training)
        self.assertFalse(model.bn.training)
        self.assertEqual(model.counter.item(), 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(not value.requires_grad for value in values.values()))

    def test_reference_is_distinct_balanced_and_reproducible(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        for step in range(30):
            index = reference_index(labels, step % 6, step, 17)
            self.assertNotEqual(index, step % 6)
            self.assertEqual(labels[index], step % 2)
            self.assertEqual(index, reference_index(labels, step % 6, step, 17))

    def test_reward_penalizes_harm_to_other_subject_despite_self_improvement(self):
        team = AnchoredTaskForce((1., .1, 1., 1.), batch_size=2)
        team.reward_mode = 'cross_subject'
        x = torch.nn.Parameter(torch.tensor(2.))
        before = {t: x.square() for t in TASKS}
        team.combine(before, [x])
        record = team.after_model_step({t: torch.tensor(1.) for t in TASKS},
                                       {t: torch.tensor(1.) for t in TASKS},
                                       {t: torch.tensor(2.) for t in TASKS})
        self.assertGreater(record['reward_terms']['same_subject_improvement'], 0)
        self.assertLess(record['reward'], 0)
        self.assertEqual(record['reward'], record['reward_terms']['cross_subject_improvement'])
        self.assertFalse(record['reward_terms']['validation_used'])
        restored = AnchoredTaskForce(team.base, batch_size=2)
        with self.assertRaises(ValueError):
            restored.load_state_dict(team.state_dict())
        restored.reward_mode = 'cross_subject'
        restored.load_state_dict(team.state_dict())
        self.assertEqual(restored.matched_transition[2], team.matched_transition[2])

    def test_migration_rejects_data_changes(self):
        previous = {'taskforce_gradient_normalization': 'anchored', 'data_hashes': {'train': 'fixed'},
                    'implementation_hashes': dict(PREVIOUS)}
        current = copy.deepcopy(previous)
        current['coordination_reward'] = 'cross_subject'
        current['implementation_hashes'].update({k: 'new' for k in (*PREVIOUS, 'transfer_reward.py')})
        upgrade_reward_identity(previous, current)
        current['data_hashes']['train'] = 'changed'
        with self.assertRaises(ValueError):
            upgrade_reward_identity(previous, current)

    def test_recovery_retains_reward_mode_and_learning_rate(self):
        team = AnchoredTaskForce((1., .1, 1., 1.))
        team.reward_mode = 'cross_subject'
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder) / 'stable.pt', Path(folder) / 'recovered.pt'
            torch.save({'checkpoint_kind': 'resume',
                        'identity': {'taskforce_gradient_normalization': 'anchored', 'seed': 17},
                        'optimizer': {'param_groups': [{'lr': 1e-5}], 'state': {}},
                        'progress': {'steps': 42}, 'coordination': team.state_dict()}, source)
            self.assertEqual(recover(source, dest, 1), [1e-5])
            state = torch.load(dest, weights_only=True)
            self.assertEqual(state['coordination']['reward_mode'], 'cross_subject')
            self.assertEqual(state['coordination']['trust_radius'], .05)
