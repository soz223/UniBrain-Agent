import math
import unittest
from unittest.mock import patch

from unibrain_agent.training_monitor import TrainingMonitor


class ExperimentTests(unittest.TestCase):
    def monitor(self):
        return TrainingMonitor({str(i): {'path': str(i), 'experiment': {
            'id': 'experiment', 'label': 'Matched study', 'expected_folds': 3, 'fold': i}}
            for i in range(1, 4)})

    def test_groups_runs_and_excludes_incomplete_folds(self):
        monitor = self.monitor()
        self.assertEqual(len(monitor.experiments()), 1)
        def packet(key):
            return {'reported_status': 'complete' if key != '3' else 'budget_exhausted',
                    'best_validation': {'auc': {'1': .8, '2': .9, '3': 1.0}[key]}}
        with patch.object(monitor, 'read', side_effect=packet):
            summary = monitor.experiment('experiment')
        auc = summary['summary']['validation']['auc']
        self.assertEqual(auc['n'], 2)
        self.assertAlmostEqual(auc['mean'], .85)
        self.assertAlmostEqual(auc['std'], math.sqrt(.005))
        self.assertIsNone(summary['summary']['test']['auc']['mean'])

    def test_single_fold_has_no_estimated_variance(self):
        monitor = self.monitor()
        with patch.object(monitor, 'read', side_effect=lambda key: {
            'reported_status': 'complete' if key == '1' else 'running', 'best_validation': {'auc': .9}}):
            auc = monitor.experiment('experiment')['summary']['validation']['auc']
        self.assertEqual(auc['n'], 1)
        self.assertIsNone(auc['std'])
        self.assertIsNone(auc['variance'])
