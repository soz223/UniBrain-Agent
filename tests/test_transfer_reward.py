import copy
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from unibrain_agent.anchored_taskforce import AnchoredTaskForce
from unibrain_agent.taskforce import TASKS
from unibrain_agent.training_stability import recover
from unibrain_agent.transfer_reward import PREVIOUS, reference_index, reference_losses, upgrade_reward_identity


class TransferRewardTests(unittest.TestCase):
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
             patch('unibrain_agent.replacement_training.original_losses', side_effect=lambda o, b, l: {t: o[0] for t in TASKS}):
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
