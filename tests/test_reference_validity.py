import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unibrain_agent.reference_validity import attach_validity, validity_for, initial_state_digest


class ReferenceValidityTests(unittest.TestCase):
    def test_fresh_bootstrap_does_not_disable_established_collapse_checks(self):
        from unibrain_agent.training_stability import qualify_probe
        progress = {}
        initial = {'classification_loss': .69, 'predicted_positive_fraction': 1.}
        self.assertFalse(qualify_probe(initial, progress, 0, 3))
        with self.assertRaisesRegex(ValueError, 'collapse'):
            qualify_probe(initial, {}, 3, 3)
        with self.assertRaisesRegex(ValueError, 'collapse'):
            qualify_probe({'classification_loss': 6., 'predicted_positive_fraction': 1.}, {}, 0, 3)
        self.assertTrue(qualify_probe({'classification_loss': .6, 'predicted_positive_fraction': .5}, progress, 1, 3))
        with self.assertRaisesRegex(ValueError, 'collapse'):
            qualify_probe(initial, progress, 1, 3)

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
