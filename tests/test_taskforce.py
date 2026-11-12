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

    def test_resume_replays_next_decision_and_optimizer_update(self):
        parameter = torch.nn.Parameter(torch.tensor([.2, .3, .4, .5]))
        team = GradientTeam(seed=71, batch_size=2)
        for step in range(4):
            losses = {name: (parameter[index] - .01 * step).square() for index, name in enumerate(TASKS)}
            team.combine(losses, [parameter])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "team.pt"
            team.save(path)
            restored = GradientTeam(seed=99, batch_size=2)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        losses = {name: (parameter[index] - .07).square() for index, name in enumerate(TASKS)}
        expected, decision = team.combine(losses, [parameter])
        actual, replayed = restored.combine(losses, [parameter])
        self.assertEqual(decision, replayed)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(team.updates, restored.updates)
        for group in ("actors", "critics", "target_actors", "target_critics"):
            for original, resumed in zip(getattr(team, group), getattr(restored, group)):
                for left, right in zip(original.parameters(), resumed.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_frozen_loss_and_extended_probes(self):
        model = torch.nn.Module()
        model.preprocessor = torch.nn.Linear(3, 3)
        model.par_net = torch.nn.Linear(3, 3)
        model.graph_gen = torch.nn.Linear(3, 3)
        model.par_net.requires_grad_(False)
        probes, names = select_probes(model, modules=["preprocessor", "par_net", "graph_gen"])
        self.assertEqual({name.split('.')[0] for name in names}, {"preprocessor", "graph_gen"})
        losses = {name: sum(parameter.square().sum() for parameter in probes) for name in TASKS}
        losses["segmentation"] = torch.tensor(1.)
        _, gram, _ = GradientTeam().observe(losses, probes)
        self.assertTrue(torch.equal(gram[2], torch.zeros(4)))

    def test_initialization_preserves_global_rng(self):
        state = torch.get_rng_state()
        GradientTeam(seed=112)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
