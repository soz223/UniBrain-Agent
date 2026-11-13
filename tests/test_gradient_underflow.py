import copy
import unittest

import torch

from unibrain_agent.replacement_training import (
    PRE_UNDERFLOW_AUDIT_TRAINER, classification_gradient_paths, classification_path_valid,
    resume_identity_matches,
    gradient_statistics, probe_total_underflow, track_gradient_health,
)


class GradientUnderflowTests(unittest.TestCase):
    def measure(self, mode="underflow"):
        model = torch.nn.Module()
        model.ext_net = torch.nn.Linear(1, 1, bias=False)
        model.candidate_modules = ("ext_net",)
        image = torch.ones(1, 1, requires_grad=True)
        feature = model.ext_net(image).sum()
        if mode == "detached":
            feature = feature.detach()
        coefficient = 0 if mode == "dead" else 1e-10
        logits = torch.stack((feature * coefficient + image.sum(), image.sum() * 0 + 90)).reshape(1, 2)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        return model, image, loss

    def test_probe_recovers_underflow_without_changing_optimizer_gradient(self):
        model, image, loss = self.measure()
        expected = torch.autograd.grad(loss, model.ext_net.weight, retain_graph=True)[0]
        paths = classification_gradient_paths(model, loss, "original", True, image)
        value = paths['candidates']['ext_net']
        self.assertFalse(value['nonzero'])
        self.assertTrue(value['underflow_probe']['nonzero'])
        self.assertTrue(classification_path_valid(value))
        self.assertIsNone(model.ext_net.weight.grad)
        loss.backward()
        torch.testing.assert_close(model.ext_net.weight.grad, expected, rtol=0, atol=0)

    def test_disconnected_and_structurally_zero_paths_still_fail(self):
        for mode in ("detached", "dead"):
            model, image, loss = self.measure(mode)
            value = classification_gradient_paths(model, loss, "original", True, image)['candidates']['ext_net']
            self.assertFalse(classification_path_valid(value))

    def test_resume_upgrade_allows_only_the_known_audit_change(self):
        previous = {'seed': 17, 'implementation_hashes': {'replacement_training.py': PRE_UNDERFLOW_AUDIT_TRAINER, 'paper_original.py': 'same'}}
        current = copy.deepcopy(previous)
        current['implementation_hashes']['replacement_training.py'] = 'new'
        with self.assertRaises(ValueError):
            resume_identity_matches(previous, current)
        self.assertFalse(resume_identity_matches(previous, current, True))
        current['seed'] = 19
        with self.assertRaises(ValueError):
            resume_identity_matches(previous, current, True)

    def test_weighted_total_probe_preserves_actual_gradients(self):
        model = torch.nn.Module()
        model.feature_net = torch.nn.Linear(1, 1, bias=False)
        model.candidate_modules = ('feature_net',)
        image = torch.ones(1, 1, requires_grad=True)
        logits = torch.stack((model.feature_net(image).sum(), image.sum() * 0 + 90)).reshape(1, 2)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        paths = classification_gradient_paths(model, loss, 'original', True, image)
        total = loss * 1e-10
        total.backward(retain_graph=True)
        before = model.feature_net.weight.grad.clone()
        record = {'classification_to_candidates': paths['candidates'],
                  'classification_to_input': paths['input'],
                  'total_to_candidates': {'feature_net': gradient_statistics([model.feature_net.weight.grad])}}
        probe_total_underflow(model, total, loss, record)
        self.assertFalse(record['total_to_candidates']['feature_net']['nonzero'])
        self.assertTrue(classification_path_valid(record['total_to_candidates']['feature_net']))
        torch.testing.assert_close(model.feature_net.weight.grad, before, atol=0, rtol=0)
        health = {}
        for step in range(8):
            track_gradient_health(record, health, step)
        with self.assertRaisesRegex(ValueError, 'Persistent numerical underflow'):
            track_gradient_health(record, health, 9)

    def test_total_detach_is_not_rescued(self):
        model, image, loss = self.measure()
        model.feature_net = model.ext_net
        model.candidate_modules = ('feature_net',)
        paths = classification_gradient_paths(model, loss, 'original', True, image)
        total = model.feature_net.weight.sum() * 0 + image.sum()
        total.backward(retain_graph=True)
        record = {'classification_to_candidates': paths['candidates'],
                  'total_to_candidates': {'feature_net': gradient_statistics([model.feature_net.weight.grad])}}
        probe_total_underflow(model, total, loss, record)
        self.assertFalse(classification_path_valid(record['total_to_candidates']['feature_net']))
