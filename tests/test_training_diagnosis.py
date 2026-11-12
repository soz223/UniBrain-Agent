import tempfile
import unittest

import numpy as np

from unibrain_agent.agents import Agent
from unibrain_agent.roles import run_team
from unibrain_agent.segmentation_audit import label_metrics
from unibrain_agent.training_diagnosis import diagnostic_context, qualify, review


class DiagnosisTests(unittest.TestCase):
    def test_metric_and_permutation_are_distinct(self):
        reference = np.array([0, 1, 2, 3, 1, 2, 3, 0]).reshape(2, 2, 2)
        prediction = np.array([0, 2, 3, 1])[reference]
        result = label_metrics(prediction, reference)
        self.assertLess(result['foreground_macro_dice'], .001)
        self.assertEqual(result['best_permuted_dice'], 1.)
        self.assertTrue(result['permutation_is_diagnostic_only'])
        self.assertEqual(label_metrics(reference, reference)['foreground_macro_dice'], 1.)

    def test_invalid_labels_and_shapes_fail(self):
        with self.assertRaises(ValueError):
            label_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            label_metrics(np.full((2, 2, 2), .5), np.zeros((2, 2, 2)))

    def test_empty_classes_remain_in_the_mean(self):
        result = label_metrics(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
        self.assertEqual(result['foreground_macro_dice'], 1.)

    def test_detached_target_isolates_the_target_gradient_path(self):
        import torch
        from unibrain_agent.segmentation_audit import gradient_summary
        torch.manual_seed(3)
        student, teacher = torch.nn.Linear(2, 4), torch.nn.Linear(2, 4)
        image = torch.ones(1, 2)
        logits, target = student(image), teacher(image).softmax(1)
        loss = torch.nn.functional.cross_entropy(logits, target)
        self.assertGreater(gradient_summary(loss, teacher)['nonzero_tensors'], 0)
        detached = torch.nn.functional.cross_entropy(logits, target.detach())
        self.assertEqual(gradient_summary(detached, teacher)['connected_tensors'], 0)
        self.assertGreater(gradient_summary(detached, student)['nonzero_tensors'], 0)

    def test_rule_fixtures_do_not_claim_llm_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = Agent('rules', folder, lambda *a, **kw: None)
            result = qualify(agent)
            self.assertTrue(result['passed'])
            self.assertEqual(result['backend'], 'rules')
            self.assertEqual(len(result['cases']), 6)

    def test_prompt_receives_no_subject_identifiers_or_paths(self):
        context = diagnostic_context({'cases': [{'id': 'private', 'path': '/private/data', 'segmentation_loss': .02}]})
        self.assertNotIn('private', str(context))
        self.assertEqual(context['evidence'][0]['segmentation_loss'], .02)

    def test_diagnostic_gate_blocks_weight_changes(self):
        context = {'diagnostic': {'task_norms': {key: 1. for key in ('extraction', 'registration', 'segmentation', 'classification')}},
                   'training_diagnosis': {'evidence': [{'id': 'probe', 'metric_absolute_error': .5}]}}
        with tempfile.TemporaryDirectory() as folder:
            agent = Agent('rules', folder, lambda *a, **kw: None)
            result = run_team(agent, context, 'coordinator')
        self.assertEqual(result['action'], 'request_measurement')

    def test_unknown_evidence_is_rejected(self):
        class UnsupportedAgent:
            def decide(self, *args, **kwargs):
                return {'evidence_ids': ['invented']}
        with self.assertRaisesRegex(ValueError, 'unavailable evidence'):
            review(UnsupportedAgent(), {'evidence': []})


if __name__ == '__main__':
    unittest.main()
