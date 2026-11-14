import copy
import tempfile
import unittest
import sys
from pathlib import Path

import torch

from unibrain_agent.taskforce import TASKS
from unibrain_agent.taskforce_paper import PaperTaskForce
from unibrain_agent.training_stability import (
    PREVIOUS_POLICY, PREVIOUS_TRAINERS, check_training_loss, recover, stable_probe, upgrade_identity,
)
from unibrain_agent.training_supervisor import diagnose, supervise


class StabilityTests(unittest.TestCase):
    def test_supervisor_rolls_back_and_restarts_after_instability(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            state = {"checkpoint_kind": "resume", "identity": {"taskforce_gradient_normalization": "bounded_l2"},
                     "optimizer": {"param_groups": [{"lr": 1e-6}]}, "progress": {}, "coordination": {}}
            torch.save(state, folder / "stable.pt")
            script = folder / "worker.py"
            script.write_text('''import json, sys
from pathlib import Path
p = Path(__file__).parent
if "--resume" not in sys.argv:
    (p / "report.json").write_text(json.dumps({"status": "failed", "error": "Stability guard: training probe classification collapse"}))
    raise SystemExit(1)
import torch
s = torch.load(sys.argv[sys.argv.index("--resume") + 1], weights_only=True)
assert s["optimizer"]["param_groups"][0]["lr"] == 5e-7
(p / "report.json").write_text(json.dumps({"status": "complete"}))
''')
            result = supervise({"output": root, "cwd": str(Path(__file__).resolve().parents[1]),
                                "stability_recovery": True,
                                "command": [sys.executable, str(script), "--verify-gradients", "--output", root]})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["stability_recoveries"], 1)
            self.assertEqual(result["restarts"], 1)

    def test_tiny_gradient_is_not_amplified_and_policy_still_learns(self):
        x = torch.nn.Parameter(torch.tensor([1.]))
        team = PaperTaskForce(batch_size=2, normalization="bounded_l2", action_bound=1.)
        optimizer = torch.optim.Adam([x], lr=1e-3)
        before = copy.deepcopy(team.actors[0].state_dict())
        for _ in range(8):
            optimizer.zero_grad()
            losses = {name: (x - i).square().sum() * (1e-15 if i == 3 else 1.) for i, name in enumerate(TASKS)}
            total, record = team.combine(losses, [x])
            self.assertTrue(all(0 < weight <= 1 for weight in record["weights"].values()))
            total.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            optimizer.step()
            team.after_model_step()
        self.assertGreater(team.updates, 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in team.actors[0].state_dict().items()))
        restored = PaperTaskForce(batch_size=2, normalization="bounded_l2", action_bound=1.)
        restored.load_state_dict(team.state_dict())
        self.assertEqual(restored.updates, team.updates)

    def test_collapse_and_underflow_trigger_diagnosis(self):
        with self.assertRaisesRegex(ValueError, "Stability guard"):
            check_training_loss(torch.tensor(132.3))
        self.assertFalse(stable_probe({"classification_loss": .69, "predicted_positive_fraction": 0.}))
        self.assertTrue(stable_probe({"classification_loss": 1., "predicted_positive_fraction": .5}))
        with tempfile.TemporaryDirectory() as root:
            diagnosis = diagnose(Path(root), {"error": "Persistent numerical underflow requires diagnosis: ext_net"}, {})
            self.assertEqual(diagnosis["kind"], "training_instability")

    def test_recovery_preserves_weights_and_adam_state(self):
        with tempfile.TemporaryDirectory() as root:
            source, dest = Path(root) / "stable.pt", Path(root) / "recovery.pt"
            state = {"checkpoint_kind": "resume", "identity": {"taskforce_gradient_normalization": "bounded_l2"},
                     "model": {"x": torch.tensor([2.])}, "optimizer": {"param_groups": [{"lr": 1e-6}], "state": {0: {"exp_avg": torch.tensor([.2])}}},
                     "progress": {"steps": 100}, "coordination": {"replay": [1], "previous": 1}}
            torch.save(state, source)
            self.assertEqual(recover(source, dest, 1), [5e-7])
            result = torch.load(dest, weights_only=True)
            torch.testing.assert_close(result["model"]["x"], state["model"]["x"])
            torch.testing.assert_close(result["optimizer"]["state"][0]["exp_avg"], torch.tensor([.2]))
            self.assertEqual(result["progress"]["steps"], 100)
            self.assertEqual(result["coordination"]["replay"], [])
            self.assertEqual(torch.load(source, weights_only=True)["optimizer"]["param_groups"][0]["lr"], 1e-6)
            with self.assertRaises(ValueError):
                recover(source, dest, 4)

    def test_migration_rejects_other_changes(self):
        previous = {"coordination": "taskforce", "taskforce_gradient_normalization": "l2", "data_hashes": "original",
                    "implementation_hashes": {"replacement_training.py": next(iter(PREVIOUS_TRAINERS)), "taskforce_paper.py": PREVIOUS_POLICY}}
        current = copy.deepcopy(previous)
        current["taskforce_gradient_normalization"] = "bounded_l2"
        current["implementation_hashes"].update({name: "new" for name in ("replacement_training.py", "taskforce_paper.py", "training_stability.py")})
        upgrade_identity(previous, current)
        current["data_hashes"] = "different"
        with self.assertRaises(ValueError):
            upgrade_identity(previous, current)
