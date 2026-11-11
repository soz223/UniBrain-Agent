import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from unibrain_agent.taskforce import GradientTeam, TASKS, select_probes
from unibrain_agent.live import RunMonitor
from unibrain_agent.builds import create_build, generated_code
from unibrain_agent.catalog import DEFAULTS


class GradientTeamTests(unittest.TestCase):
    def test_prepared_original_code_pins_policy_choice(self):
        with tempfile.TemporaryDirectory() as folder, patch("unibrain_agent.builds.BUILDS", Path(folder)):
            record = create_build(DEFAULTS, "rules", coordination="taskforce")
            self.assertEqual(record["coordination"], "taskforce")
            self.assertIn('coordination=\'taskforce\'', generated_code(DEFAULTS, record["coordination"]))

    def test_live_monitor_retains_policy_decision(self):
        monitor = RunMonitor()
        monitor.reset(1)
        monitor.record({"kind": "training", "message": "Updated", "step": 1,
                        "weighted_loss": 1.0, "losses": {"extraction": 1.0},
                        "weights": {"extraction": 1.1},
                        "gradient_policy": {"policy_updates": 1, "weights": {"extraction": 1.1}}})
        self.assertEqual(monitor.snapshot()["history"][0]["gradient_policy"]["policy_updates"], 1)

    def test_probes_bound_gradient_measurement(self):
        model = torch.nn.Module()
        for name in ("ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred"):
            setattr(model, name, torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)))
        parameters, names = select_probes(model, per_module=100)
        self.assertEqual({name.split(".")[0] for name in names},
                         {"ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred"})
        self.assertLessEqual(sum(parameter.numel() for parameter in parameters), 500)

    def test_policy_changes_backpropagation_and_learns(self):
        torch.manual_seed(4)
        parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25, 0.75, 0.1]))
        optimizer = torch.optim.SGD([parameter], lr=0.02)
        team = GradientTeam(seed=4, batch_size=2)
        decisions = []
        for step in range(5):
            optimizer.zero_grad()
            losses = {name: (parameter[i] - 0.1 * (step + 1)) ** 2 + 0.01
                      for i, name in enumerate(TASKS)}
            total, decision = team.combine(losses, [parameter])
            self.assertAlmostEqual(sum(decision["weights"][name] / team.base[i]
                                       for i, name in enumerate(TASKS)), 4.0, places=5)
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertTrue(torch.isfinite(parameter.grad).all())
            expected = torch.tensor([2 * (float(parameter[i].detach()) - 0.1 * (step + 1)) *
                                     decision["weights"][name] for i, name in enumerate(TASKS)])
            torch.testing.assert_close(parameter.grad, expected)
            optimizer.step()
            decisions.append(decision)
        self.assertGreater(team.updates, 0)
        self.assertNotEqual(decisions[0]["weights"], decisions[-1]["weights"])
        self.assertEqual(decisions[-1]["gram"], list(map(list, zip(*decisions[-1]["gram"]))))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pt"
            team.save(path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["policy_updates"], team.updates)
            self.assertEqual(tuple(saved["tasks"]), TASKS)


if __name__ == "__main__":
    unittest.main()
