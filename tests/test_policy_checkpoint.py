import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from unibrain_agent.policy_checkpoint import inspect_policy, source_digest
from unibrain_agent.taskforce import GradientTeam, TASKS


class PolicyCheckpointTests(unittest.TestCase):
    def fixture(self, root):
        seed, batch = 19, 2
        team = GradientTeam(seed=seed, batch_size=batch)
        parameter = torch.nn.Parameter(torch.tensor([.2, .3, .4, .5]))
        optimizer = torch.optim.SGD([parameter], lr=.01)
        for step in range(5):
            optimizer.zero_grad()
            losses = {task: (parameter[index] - .1 * step).square() + .01 for index, task in enumerate(TASKS)}
            loss, _ = team.combine(losses, [parameter])
            loss.backward()
            optimizer.step()
        config = {"seed": seed, "coordination_batch_size": batch, "coordination": "taskforce",
                  "method": "synthstrip", "setting": "joint", "dataset": "fixture", "fold": 1}
        state = {"checkpoint_kind": "resume", "coordination": team.state_dict(), "progress": {"steps": 5, "pretrain_steps": 0},
                 "identity": {**config, "implementation_hashes": {"taskforce.py": source_digest()}}}
        report = {"status": "complete", "config": config, "seed": seed, "method": "synthstrip", "setting": "joint", "steps": 5,
                  "coordination": {"mode": "taskforce", "base_weights": dict(zip(TASKS, team.base)), "policy_updates": team.updates,
                                   "actor_optimizer_steps": 4 * team.updates, "critic_optimizer_steps": 4 * team.updates,
                                   "last_decision": team.last_record}}
        checkpoint, report_path = root / "last.pt", root / "report.json"
        torch.save(state, checkpoint)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return checkpoint, report_path, state, report

    def test_real_policy_updates_change_each_actor_and_critic(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report, _, _ = self.fixture(Path(directory))
            result = inspect_policy(checkpoint, report)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["policy_updates"], 3)
        for group in ("actors", "critics"):
            self.assertTrue(all(value["changed_tensors"] > 0 and value["delta_l1"] > 0 for value in result["deltas"][group].values()))
        self.assertNotIn(directory, json.dumps(result))
        self.assertEqual(len(result["checkpoint_sha256"]), 64)

    def test_positive_counts_with_unchanged_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report, state, _ = self.fixture(Path(directory))
            initial = GradientTeam(seed=19, batch_size=2).state_dict()
            for group in ("actors", "critics"):
                state["coordination"][group] = copy.deepcopy(initial[group])
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "actor or critic is unchanged"):
                inspect_policy(checkpoint, report)

    def test_counter_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report_path, _, report = self.fixture(Path(directory))
            report["coordination"]["actor_optimizer_steps"] += 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counts disagree"):
                inspect_policy(checkpoint, report_path)

    def test_nonfinite_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report, state, _ = self.fixture(Path(directory))
            state["coordination"]["actors"][0]["0.weight"][0, 0] = float("nan")
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                inspect_policy(checkpoint, report)

    def test_seed_and_source_identity_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report, state, _ = self.fixture(Path(directory))
            state["identity"]["seed"] = 20
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "identities disagree"):
                inspect_policy(checkpoint, report)
            state["identity"]["seed"] = 19
            state["identity"]["implementation_hashes"]["taskforce.py"] = "incorrect"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "differs from the checkpoint source"):
                inspect_policy(checkpoint, report)


if __name__ == "__main__":
    unittest.main()
