import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from unibrain_agent.objective import observe_objective, preview_objective


class ObjectiveTests(unittest.TestCase):
    def test_compact_components_reconstruct_the_actual_training_objective(self):
        from unibrain_agent.pipeline import loss_terms
        model = torch.nn.Conv3d(1, 4, 1)
        image = torch.ones(1, 1, 3, 3, 3)
        logits = model(image)
        output = {'mask': logits[:, :1].sigmoid(), 'warped': logits[:, :1], 'tissue_logits': logits,
                  'tissue_target': torch.zeros_like(image), 'logits': logits.mean((2, 3, 4)),
                  'parcellation_loss': logits.square().mean(), 'regularization': logits.abs().mean()}
        sample = {'raw_img_brain_mask_np': torch.ones_like(image), 'label': torch.tensor([0])}
        core = SimpleNamespace(GCC=lambda: torch.nn.MSELoss())
        with patch('unibrain_agent.pipeline.unibrain_core', return_value=core):
            losses = loss_terms(output, sample, {'image': image}, 'bce_dice')
        weights = {'extraction': 1.2, 'registration': .3, 'segmentation': .8, 'classification': 1.1}
        total = sum(weights[name] * value for name, value in losses.items())
        packet = observe_objective(model, losses, weights, total, 1)
        terms = {term['id']: term for term in packet['terms']}
        self.assertAlmostEqual(terms['regularization']['weight'], .003)
        self.assertAlmostEqual(terms['parcellation']['weight'], .16)
        self.assertEqual(terms['extraction']['definition'], 'BCE + soft Dice loss')
        total.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())

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
