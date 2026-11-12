import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from unibrain_agent.live import RunMonitor
from unibrain_agent.runner import trace_modules
from unibrain_agent.server import create_server


class LiveTests(unittest.TestCase):
    def test_objective_is_latest_only_and_cleared_on_reset(self):
        monitor = RunMonitor()
        monitor.record({'kind': 'training', 'message': 'Update', 'step': 1, 'weighted_loss': 1.,
                        'losses': {}, 'weights': {}, 'objective': {'step': 1, 'terms': []}})
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot['objective']['step'], 1)
        self.assertNotIn('objective', snapshot['history'][0])
        self.assertNotIn('objective', snapshot['events'][0])
        monitor.reset(2)
        self.assertIsNone(monitor.snapshot()['objective'])

    def test_observing_modules_preserves_outputs_and_gradients(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(stages={})
        model.ext_net = torch.nn.Linear(3, 2)
        model.bias_net = torch.nn.Identity()
        sample = torch.ones(1, 3, requires_grad=True)
        expected = model.ext_net(sample)
        gradient = torch.autograd.grad(expected.sum(), model.ext_net.weight)[0]
        events = []
        with trace_modules(model, lambda kind, message, **extra: events.append(dict(kind=kind, **extra))):
            actual = model.ext_net(sample)
            actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.ext_net.weight.grad, gradient)
        self.assertEqual([e["action"] for e in events], ["started", "completed"])
        self.assertGreaterEqual(events[-1]["duration"], 0)
        self.assertFalse(model.ext_net._forward_hooks)
        self.assertFalse(model.ext_net._forward_pre_hooks)
        self.assertFalse(model.bias_net._forward_hooks)

    def test_live_records_are_isolated_and_reset_for_each_run(self):
        monitor = RunMonitor()
        monitor.reset(5)
        monitor.record({"kind": "training", "message": "Step 1", "step": 1, "weighted_loss": 2.1,
                        "losses": {"classification": .4}, "weights": {"classification": 1}, "subject_id": "private"})
        snapshot = monitor.snapshot()
        self.assertNotIn("private", json.dumps(snapshot))
        snapshot["history"][0]["weighted_loss"] = -1
        self.assertEqual(monitor.snapshot()["history"][0]["weighted_loss"], 2.1)
        monitor.record({"kind": "failed", "message": "Expected test failure"})
        self.assertEqual(monitor.snapshot()["phase"], "failed")
        monitor.reset(2)
        self.assertEqual(monitor.snapshot()["history"], [])
        self.assertEqual(monitor.snapshot()["events"], [])

    def test_long_run_keeps_recent_history_and_total_progress(self):
        monitor = RunMonitor()
        monitor.reset(700)
        for step in range(1, 701):
            monitor.record({"kind": "training", "message": "Training", "step": step,
                            "weighted_loss": 1.0, "losses": {}, "weights": {}})
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["completed_steps"], 700)
        self.assertEqual(len(snapshot["history"]), 500)
        self.assertEqual(snapshot["history"][0]["step"], 201)

    def test_api_streams_updates_and_rejects_duplicate_runs(self):
        started, release = threading.Event(), threading.Event()
        server = create_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def post():
            return urlopen(Request(base + "/api/run", data=b'{"backend":"rules","steps":2}',
                                   headers={"Content-Type": "application/json"}))

        try:
            with tempfile.TemporaryDirectory() as directory:
                def fake_run(**kwargs):
                    self.assertFalse(kwargs["comparison"])
                    folder = Path(directory)
                    kwargs["on_event"]({"kind": "phase", "message": "Training", "phase": "training"}, folder)
                    kwargs["on_event"]({"kind": "training", "message": "Update 1", "step": 1,
                                        "weighted_loss": 1.2, "losses": {}, "weights": {}}, folder)
                    started.set()
                    release.wait(5)
                    kwargs["on_event"]({"kind": "complete", "message": "Complete"}, folder)
                    return {"run_dir": directory}

                with patch("unibrain_agent.runner.run", side_effect=fake_run):
                    with post() as response:
                        self.assertEqual(response.status, 202)
                    self.assertTrue(started.wait(3))
                    with urlopen(base + "/api/status") as response:
                        state = json.load(response)
                    self.assertTrue(state["running"])
                    self.assertEqual(state["live"]["phase"], "training")
                    self.assertEqual(state["live"]["history"][0]["weighted_loss"], 1.2)
                    self.assertGreater(state["live"]["resources"]["rss_mb"], 0)
                    with self.assertRaises(HTTPError) as failure:
                        post()
                    self.assertEqual(failure.exception.code, 409)
                    release.set()
                    for worker in threading.enumerate():
                        if worker.name.endswith("(execute)"):
                            worker.join(3)
                    self.assertTrue(server.state["completed"])
                    saved = json.loads((Path(directory) / "report.json").read_text())
                    self.assertEqual(saved["live"]["phase"], "complete")
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
