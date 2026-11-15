import unittest

from unibrain_agent.generalization_audit import diagnose_cases, summarize


class GeneralizationAuditTests(unittest.TestCase):
    def test_full_cohort_exposes_unrepresentative_probe(self):
        train = [{'id': str(i), 'site': 'A', 'label': i % 2,
                  'probability': (.9 if i % 2 else .1) if i < 4 else (.1 if i % 2 else .9)}
                 for i in range(12)]
        result = diagnose_cases(train, train[4:], ['0', '1', '2', '3'])
        self.assertEqual(result['summaries']['train_probe']['auc'], 1)
        self.assertEqual(result['summaries']['train_complement']['auc'], 0)
        self.assertLess(result['summaries']['train_full']['auc'], .5)
        self.assertFalse(result['test_evaluated'])

    def test_site_baseline_does_not_fit_validation_labels(self):
        train = [{'id': str(i), 'site': 'A' if i < 4 else 'B', 'label': int(i < 4), 'probability': .5}
                 for i in range(8)]
        val = [{**c, 'label': 1 - c['label']} for c in train]
        result = diagnose_cases(train, val, [])
        self.assertEqual(result['summaries']['site_only_validation']['auc'], 0)
        self.assertIsNone(result['sites']['A']['validation']['auc'])

    def test_calibration_and_confident_errors_are_reported(self):
        result = summarize([{'label': 0, 'probability': .99}, {'label': 1, 'probability': .01}])
        self.assertEqual(result['confident_error_fraction'], 1)
        self.assertGreater(result['classification_loss'], 4)
        self.assertEqual(result['auc_ci95'], [0, 0])
