import unittest

import torch

from unibrain_agent.objective import observe_objective, preview_objective


class ObjectiveTests(unittest.TestCase):
    def test_preview_declares_optional_penalties_without_claiming_measured_connections(self):
        from unibrain_agent.catalog import DEFAULTS
        selection = {**DEFAULTS, 'registration': 'M06', 'denoising': 'M14', 'parcellation': 'M09', 'graph': 'continuous'}
        result = preview_objective(selection)
        weights = {term['id']: term['weight'] for term in result['terms']}
        self.assertEqual(weights['regularization'], .001)
        self.assertEqual(weights['parcellation'], .2)
        self.assertNotIn('denoising', weights)
        self.assertIn('denoising', [module['id'] for module in result['modules']])
        self.assertEqual(result['connections'], [])
        self.assertEqual(result['status'], 'declared')

    def test_executed_weights_and_detached_path(self):
        model = torch.nn.ModuleDict({'denoising': torch.nn.Linear(2, 2), 'classifier': torch.nn.Linear(2, 1)})
        image = torch.ones(1, 2)
        output = model['denoising'](image)
        losses = {'diagnosis': model['classifier'](output.detach()).square().mean(), 'smoothness': output.square().mean()}
        weights = {'diagnosis': .7, 'smoothness': .03}
        total = sum(losses[name] * weight for name, weight in weights.items())
        packet = observe_objective(model, losses, weights, total, 1)
        edges = {(edge['loss'], edge['module']) for edge in packet['connections']}
        self.assertNotIn(('diagnosis', 'denoising'), edges)
        self.assertIn(('diagnosis', 'classifier'), edges)
        self.assertIn(('smoothness', 'denoising'), edges)
        self.assertAlmostEqual(packet['total'], packet['reconstructed_total'], places=5)
        total.backward()
        self.assertIsNotNone(model['denoising'].weight.grad)

    def test_unreported_loss_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        loss = model(torch.ones(1, 2)).square().mean()
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            observe_objective(model, {'classification': loss}, {'classification': 1.}, loss + 1, 1)


if __name__ == '__main__':
    unittest.main()
