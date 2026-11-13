import signal
import unittest
from unittest.mock import patch

import numpy as np

from unibrain_agent.training_control import TrainingBudget, normalize_template, training_probe_indices


class TrainingControlTests(unittest.TestCase):
    def test_deadline_requests_a_boundary_stop_and_restores_signal_handlers(self):
        before = signal.getsignal(signal.SIGTERM)
        with patch('unibrain_agent.training_control.time.monotonic', return_value=10):
            with TrainingBudget(20) as budget:
                self.assertIsNone(budget.stop_reason)
                with patch('unibrain_agent.training_control.time.monotonic', return_value=30):
                    self.assertEqual(budget.stop_reason, 'wall_time_budget')
                budget.request_stop(signal.SIGTERM, None)
                self.assertEqual(budget.stop_reason, 'termination_signal')
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        for seconds in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                TrainingBudget(seconds)

    def test_template_scaling_preserves_source_and_geometry(self):
        source = np.array([0., 24., 48., 96.], dtype=np.float32).reshape(2, 2)
        before = source.copy()
        normalized, metadata = normalize_template(source, 'minmax')
        np.testing.assert_array_equal(source, before)
        np.testing.assert_allclose(normalized, source / 96)
        self.assertEqual(normalized.shape, source.shape)
        self.assertEqual(metadata['source_max'], 96.)
        self.assertEqual(metadata['output_max'], 1.)
        for invalid in (np.zeros(3), np.array([0., np.nan])):
            with self.assertRaises(ValueError):
                normalize_template(invalid, 'minmax')

    def test_probe_is_fixed_balanced_unique_and_handles_rare_classes(self):
        labels = np.array([0] * 9 + [1] * 3)
        indices = training_probe_indices(labels, 6, 17)
        self.assertEqual(indices, training_probe_indices(labels, 6, 17))
        self.assertEqual(len(set(indices)), 6)
        self.assertEqual(int(labels[indices].sum()), 3)
        self.assertEqual(training_probe_indices(labels, 100, 17), list(range(12)))
        self.assertEqual(training_probe_indices(labels, 0, 17), [])
        with self.assertRaises(ValueError):
            training_probe_indices(labels, 1, 17)
