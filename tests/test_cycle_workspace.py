import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from unibrain_agent.access import AccessPolicy
from unibrain_agent.cycle_workspace import CycleWorkspace, save_json
from unibrain_agent.providers import ProviderSettings
from unibrain_agent.server import create_server


class CycleWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.data = self.root / "source", self.root / "data"
        self.source.mkdir()
        self.data.mkdir()
        self.workspace = CycleWorkspace(self.root / "jobs")
        self.runtime = {"original_source": str(self.source), "data": str(self.data),
                        "original_python": sys.executable, "device": "cuda:3"}
        self.provider = {"provider": "openai", "model": "configured-model", "api_key": "private-api-key"}

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, **options):
        return self.workspace.create({"task": "Check graph classification", **options}, self.runtime, None, self.provider)["id"]

    def test_full_command_and_process_credentials_are_separate(self):
        identifier = self.create(coordination="taskforce", mode="preflight")
        root = self.workspace.path(identifier)
        record = json.loads((root / "job.json").read_text())
        command = record["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith("run_research_cycle.py"))
        for flag, expected in (("--coordination", "taskforce"), ("--checkpoint-mode", "weights"),
                               ("--acceptance-steps", "1"), ("--train-steps", "2"), ("--gpu", "3")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--preflight", command)
        self.assertNotIn("private-api-key", json.dumps(record))

        def execute(argv, log, timeout, **kwargs):
            self.assertIsNone(timeout)
            self.assertEqual(log, root / "process.log")
            self.assertEqual(kwargs["env"]["OPENAI_API_KEY"], "private-api-key")
            self.assertEqual(kwargs["env"]["UNIBRAIN_AI_SETTINGS"], str((self.root / "ai.json").resolve()))
            cycle = Path(argv[argv.index("--output") + 1])
            cycle.mkdir()
            save_json(cycle / "report.json", {"status": "ready", "phase": "static_preflight"})
            return 0

        with patch("unibrain_agent.cycle_workspace.run_logged", side_effect=execute):
            self.workspace.execute(identifier, self.provider, self.root / "ai.json")
        public = self.workspace.read(identifier)
        self.assertEqual(public["status"], "ready")
        self.assertNotIn("private-api-key", json.dumps(public))
        self.assertIsNone(self.workspace.active)

    def test_live_progress_excludes_subject_rows_and_checks_accepted_code(self):
        identifier = self.create()
        cycle = self.workspace.path(identifier) / "cycle"
        training = cycle / "training"
        training.mkdir(parents=True)
        code = cycle / "accepted_adapter.py"
        code.write_text("def adapt(core, graph, features):\n    return core(graph, features)\n", encoding="utf-8")
        digest = hashlib.sha256(code.read_bytes()).hexdigest()
        save_json(cycle / "report.json", {"status": "running", "phase": "joint_training", "method": "bnt",
                                         "accepted_adapter": {"sha256": digest}, "private_ids": ["subject-private"]})
        save_json(training / "report.json", {"steps": 0, "predictions": [{"id": "subject-private"}],
                                              "coordination": {"mode": "taskforce", "policy_updates": 1}})
        (training / "updates.jsonl").write_text(json.dumps({"step": 7, "total_loss": 2.5, "id": "subject-private",
                                                             "gradient_policy": {"policy_updates": 3, "weights": {"classification": 1.1}}}) + "\n", encoding="utf-8")
        result = self.workspace.read(identifier)
        self.assertEqual(result["actual_updates"]["training"], 7)
        self.assertEqual(result["coordination"]["policy_updates"], 3)
        self.assertNotIn("subject-private", json.dumps(result))
        self.assertEqual(self.workspace.code(identifier)["sha256"], digest)
        code.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.workspace.code(identifier)
        with self.assertRaisesRegex(ValueError, "ID"):
            self.workspace.read("../outside")

    def test_invalid_requests_do_not_create_jobs(self):
        for body in ({"train_steps": True}, {"acceptance_steps": 3}, {"stage": "unknown"}, {"coordination": []}, {"mode": []},
                     {"output": "arbitrary"}, {"data": "missing-fold"}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.create(**body)
        self.assertFalse(self.workspace.folder.exists())

    def test_api_requires_account_connection_and_excludes_concurrent_baseline(self):
        settings = ProviderSettings(self.root / "ai.json")
        finished = threading.Event()
        launched = threading.Event()

        def execute(identifier, provider, ai_path):
            launched.set()
            finished.wait(5)
            root = self.workspace.path(identifier)
            record = json.loads((root / "job.json").read_text())
            record.update(status="ready")
            save_json(root / "job.json", record)
            self.workspace.active = None

        with patch("unibrain_agent.providers.provider_settings", return_value=settings), \
             patch("unibrain_agent.accounts.login_status", return_value={"authenticated": True}), \
             patch("unibrain_agent.cycle_workspace.CycleWorkspace", return_value=self.workspace), \
             patch("unibrain_agent.runtime.settings", return_value=self.runtime), \
             patch.object(self.workspace, "execute", side_effect=execute):
            server = create_server(0, access=AccessPolicy(token="", origins=[]))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def request(route, body=None, session=None):
                headers = {"Content-Type": "application/json"}
                if session:
                    headers["X-UniBrain-AI-Session"] = session
                with urlopen(Request(base + route, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers), timeout=5) as response:
                    return json.load(response)

            try:
                with self.assertRaises(HTTPError) as caught:
                    request("/api/research-cycles", {"task": "Test"})
                self.assertEqual(caught.exception.code, 403)
                session = request("/api/ai/connect", {"confirm": True})["session"]
                server.state["running"] = True
                with self.assertRaises(HTTPError) as caught:
                    request("/api/research-cycles", {"task": "Test"}, session)
                self.assertEqual(caught.exception.code, 409)
                server.state["running"] = False
                result = request("/api/research-cycles", {"task": "Test", "mode": "preflight"}, session)
                self.assertTrue(launched.wait(2))
                self.assertEqual(request("/api/status")["job"], "research_cycle")
                self.assertEqual(request("/api/research-cycles/" + result["id"])["status"], "running")
                with self.assertRaises(HTTPError) as caught:
                    request("/api/run", {"backend": "rules", "steps": 2})
                self.assertEqual(caught.exception.code, 409)
            finally:
                finished.set()
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
