import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from unibrain_agent.training_supervisor import retry_reason, supervise


class SupervisorTests(unittest.TestCase):
    def test_gradient_failures_never_retry_blindly(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'last.pt'
            checkpoint.touch()
            for error in ('Required gradient paths failed: feature_net:total', 'Nonfinite parameter gradient', 'CUDA out of memory'):
                self.assertIsNone(retry_reason(-9, {'status': 'failed', 'error': error}, checkpoint))
            self.assertEqual(retry_reason(-9, {'status': 'running'}, checkpoint), 'killed_worker_checkpoint_rollback')
            self.assertIsNone(retry_reason(-15, {}, checkpoint))
            checkpoint.unlink()
            self.assertIsNone(retry_reason(-9, {}, checkpoint))

    def test_existing_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'supervisor.lock').write_text('other worker')
            with self.assertRaises(FileExistsError):
                supervise({'output': directory, 'command': ['unused', '--verify-gradients', '--output', directory],
                           'wall_seconds': 30, 'cwd': directory})
            self.assertEqual((folder / 'supervisor.lock').read_text(), 'other worker')

    @unittest.skipIf(os.name == 'nt', 'POSIX kill return code')
    def test_killed_worker_resumes_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            script = folder / 'worker.py'
            script.write_text('''import json, os, signal, sys
from pathlib import Path
p = Path(__file__).parent
if not (p / "last.pt").exists():
    (p / "last.pt").write_text("valid fake checkpoint")
    (p / "report.json").write_text(json.dumps({"status": "running"}))
    os.kill(os.getpid(), signal.SIGKILL)
assert "--resume" in sys.argv
(p / "report.json").write_text(json.dumps({"status": "complete"}))
''')
            result = supervise({'output': directory, 'cwd': directory, 'wall_seconds': 30,
                                'command': [sys.executable, str(script), '--verify-gradients', '--output', directory]})
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['restarts'], 1)
            self.assertFalse((folder / 'supervisor.lock').exists())
