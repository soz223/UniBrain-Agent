import unittest

import torch
from torch import nn
from torch.nn import functional as F

from unibrain_agent.agents import PLAN_SCHEMA, balance_weights, validate_decision
from unibrain_agent.pipeline import TASKS, atlas_probabilities
from unibrain_agent.validation import gate_candidate, gradients


class GradientTests(unittest.TestCase):
    def test_soft_atlas_restores_grid_gradient(self):
        labels = torch.zeros(1, 1, 4, 4, 4)
        labels[:, :, 2:] = 1
        for soft in (False, True):
            theta = torch.eye(3, 4).unsqueeze(0).requires_grad_()
            grid = F.affine_grid(theta, labels.shape, align_corners=True)
            probabilities = atlas_probabilities(labels, grid, soft)
            self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(1, 4, 4, 4)))
            if soft:
                gradient = torch.autograd.grad(probabilities[:, 1, 1:3].sum(), theta)[0]
                self.assertGreater(float(gradient.abs().sum()), 0)
            else:
                self.assertFalse(probabilities.requires_grad)

    def test_disconnected_and_zero_are_different(self):
        model = nn.Module()
        model.ext_net = nn.Linear(2, 2)
        model.reg_net = nn.Linear(2, 2)
        model.seg_net = nn.Linear(2, 2)
        model.par_net = nn.Linear(2, 2).requires_grad_(False)
        model.feature_net = nn.Identity()
        model.graph_gen = nn.Identity()
        model.gnn_pred = nn.Linear(2, 1)
        features = model.ext_net(torch.ones(1, 2))
        registered = model.reg_net(features)
        segmented = model.seg_net(features)
        edges = registered.square()
        losses = {"extraction": features.sum(), "registration": registered.sum(),
                  "segmentation": segmented.sum() * 0, "classification": model.gnn_pred(edges.detach()).sum()}
        audit = gradients(model, losses, {"raw_edges": edges})
        self.assertEqual(audit["matrix"]["classification"]["reg_net"]["state"], "disconnected")
        self.assertEqual(audit["matrix"]["segmentation"]["seg_net"]["state"], "zero")
        self.assertEqual(audit["matrix"]["extraction"]["par_net"]["state"], "inactive")
        self.assertFalse(audit["classification_to_graph_weights"]["connected"])

    def test_weights_preserve_budget_and_bounds(self):
        weights = dict(zip(TASKS, [1.0, 0.1, 1.0, 1.0]))
        norms = dict(zip(TASKS, [1e-6, 10000., 0., 1.]))
        updated = balance_weights(weights, norms)
        self.assertAlmostEqual(sum(updated.values()), sum(weights.values()))
        for key in weights:
            self.assertGreaterEqual(updated[key], 0.5 * weights[key])
            self.assertLessEqual(updated[key], 2 * weights[key])

    def test_action_schema_rejects_unregistered_modules(self):
        plan = {"extraction": "shell_command", "soft_atlas": True, "continuous_edges": True,
                "extraction_loss": "bce", "rationale": "test", "evidence_ids": [], "limitations": []}
        with self.assertRaises(ValueError):
            validate_decision(plan, PLAN_SCHEMA)

    def test_gate_cannot_keep_a_broken_connection(self):
        baseline = {"validation": {"reference_loss": 1.0}, "training_seconds": 1.0}
        candidate = {"validation": {"reference_loss": 0.9}, "training_seconds": 1.1,
                     "spatial": {"passed": True}, "updated_parameters": {"ext_net": 2},
                     "gradients": {"matrix": {"classification": {"reg_net": {"state": "active"}}},
                                   "classification_to_graph_weights": {"connected": True, "norm": 0.1}}}
        self.assertEqual(gate_candidate(baseline, candidate)["decision"], "keep")
        candidate["gradients"]["classification_to_graph_weights"]["connected"] = False
        self.assertEqual(gate_candidate(baseline, candidate)["decision"], "revert")
        candidate["gradients"]["classification_to_graph_weights"]["connected"] = True
        candidate["training_seconds"] = 5.0
        self.assertEqual(gate_candidate(baseline, candidate)["decision"], "revert")


if __name__ == "__main__":
    unittest.main()
