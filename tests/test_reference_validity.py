import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unibrain_agent.reference_validity import attach_validity, validity_for, initial_state_digest


class ReferenceValidityTests(unittest.TestCase):
    def test_finite_probe_imbalance_records_warning_and_allows_training(self):
        from unibrain_agent.training_stability import qualify_probe
        for epoch, loss, fraction in ((0, .69, 1.), (3, .69, 0.), (100, 6., 1.)):
            with self.subTest(epoch=epoch):
                progress = {}
                metrics = {'classification_loss': loss, 'predicted_positive_fraction': fraction}
                self.assertFalse(qualify_probe(metrics, progress, epoch, 3))
                state = progress['stability_bootstrap']
                self.assertFalse(state['qualified'])
                self.assertEqual(state['last_warning'], {'epoch': epoch, **metrics})

    def test_qualified_probe_retains_later_warnings(self):
        from unibrain_agent.training_stability import qualify_probe
        progress = {}
        balanced = {'classification_loss': .6, 'predicted_positive_fraction': .5}
        imbalanced = {'classification_loss': .69, 'predicted_positive_fraction': 1.}
        self.assertTrue(qualify_probe(balanced, progress, 1, 3))
        self.assertFalse(qualify_probe(imbalanced, progress, 4, 3))
        self.assertTrue(progress['stability_bootstrap']['qualified'])
        warning = {'epoch': 4, **imbalanced}
        self.assertEqual(progress['stability_bootstrap']['last_warning'], warning)
        self.assertTrue(qualify_probe(balanced, progress, 5, 3))
        self.assertEqual(progress['stability_bootstrap']['last_warning'], warning)

    def test_invalid_probe_still_blocks_training(self):
        from unibrain_agent.training_stability import qualify_probe
        invalid = [
            {}, {'classification_loss': .6},
            *({'classification_loss': loss, 'predicted_positive_fraction': .5}
              for loss in (float('nan'), float('inf'), -float('inf'))),
            *({'classification_loss': .6, 'predicted_positive_fraction': fraction}
              for fraction in (-.1, 1.1, float('nan'), float('inf'))),
        ]
        for metrics in invalid:
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(ValueError, 'invalid training probe'):
                    qualify_probe(metrics, {}, 100, 3)

    def test_initialization_digest_detects_parameter_changes(self):
        model = torch.nn.Linear(3, 2)
        before = initial_state_digest(model)
        self.assertEqual(before, initial_state_digest(model))
        with torch.no_grad():
            model.bias.add_(1)
        self.assertNotEqual(before, initial_state_digest(model))

    def test_subset_uses_original_subject_mask(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            masks = np.zeros((2, 96, 96, 96), dtype=bool)
            masks[0, 0, 0, 0] = True
            masks[1, 1, 1, 1] = True
            np.save(path / 'tissue_valid_train.npy', masks)
            attach_validity({'train': dataset}, path)
            subset = torch.utils.data.Subset(dataset, [1, 0])
            mask = validity_for(subset, 0, 'cpu')
            self.assertTrue(mask[0, 0, 1, 1, 1])
            self.assertFalse(mask[0, 0, 0, 0, 0])
            prediction = torch.ones((1, 96, 96, 96), dtype=torch.long)
            self.assertEqual(int((prediction * mask[:, 0]).count_nonzero()), 1)
            del dataset.tissue_validity

    def test_unannotated_case_is_rejected(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(1))
        dataset.tissue_validity = np.zeros((1, 96, 96, 96), dtype=bool)
        with self.assertRaisesRegex(ValueError, 'no annotated'):
            validity_for(dataset, 0, 'cpu')
        self.assertIsNone(validity_for(torch.utils.data.TensorDataset(torch.zeros(1)), 0, 'cpu'))
