import copy
import unittest

import torch

from unibrain_agent.normalization_migration import PAPER, RELEASES, verify_migration
from unibrain_agent.registration_normalization import configure_registration


class RegistrationNormalizationTests(unittest.TestCase):
    def test_modes_agree_and_gradients_update_without_buffer_drift(self):
        torch.manual_seed(19)
        layer = torch.nn.BatchNorm3d(3)
        state = copy.deepcopy(layer.state_dict())
        parameters = list(layer.parameters())
        configure_registration(layer, 'per_image')
        image = torch.randn(1, 3, 5, 5, 5, requires_grad=True)
        layer.train()
        training = layer(image)
        layer.eval()
        torch.testing.assert_close(training, layer(image), rtol=0, atol=0)
        optimizer = torch.optim.SGD(layer.parameters(), lr=.01)
        training.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        optimizer.step()
        self.assertFalse(torch.equal(layer.weight, state['weight']))
        for name in ('running_mean', 'running_var', 'num_batches_tracked'):
            torch.testing.assert_close(layer.state_dict()[name], state[name], rtol=0, atol=0)
        self.assertIs(next(layer.parameters()), parameters[0])
        with self.assertRaises(ValueError):
            layer(torch.zeros(2, 3, 5, 5, 5))
        layer.load_state_dict(state, strict=True)

    def test_migration_rejects_data_and_unknown_source_changes(self):
        previous = {'paper_original': True, 'paper_implementation': PAPER, 'dataset': 'hcp',
                    'coordination_reward': 'cross_subject',
                    'implementation_hashes': {k: sorted(v)[0] for k, v in RELEASES.items()}}
        current = copy.deepcopy(previous)
        current.update(paper_implementation='new', registration_normalization='per_image',
                       coordination_reward='classification_advantage')
        current['implementation_hashes'].update({k: 'new' for k in RELEASES})
        current['implementation_hashes'].update({'registration_normalization.py': 'new',
                                                 'normalization_migration.py': 'new'})
        verify_migration(previous, current)
        with self.assertRaises(ValueError):
            verify_migration(previous, {**current, 'dataset': 'abide'})
        invalid = copy.deepcopy(previous)
        invalid['implementation_hashes']['replacement_training.py'] = 'unknown'
        with self.assertRaises(ValueError):
            verify_migration(invalid, current)


if __name__ == '__main__':
    unittest.main()
