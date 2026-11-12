import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from unibrain_agent.owned_process import OWNER_KEY, linux_identity, run_capture, run_logged, stop_linux_owned


class OwnedProcessTests(unittest.TestCase):
    def test_timeout_stops_the_nested_child_before_it_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            marker, log = root / "unexpected.txt", root / "run.log"
            child = "import time; from pathlib import Path; time.sleep(2); Path(" + repr(str(marker)) + ").write_text('orphan')"
            parent = ("import subprocess,sys,time,json; child=subprocess.Popen([sys.executable,'-c'," + repr(child)
                      + "]); print(json.dumps({'child':child.pid}),flush=True); time.sleep(10)")
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged([sys.executable, "-c", parent], log, 1.5, root)
            self.assertIsInstance(json.loads(log.read_text())["child"], int)
            time.sleep(2.2)
            self.assertFalse(marker.exists())

    def test_exit_status_and_log_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            result = run_logged([sys.executable, "-c", "print('retained failure'); raise SystemExit(3)"], log, 5)
            self.assertEqual(result, 3)
            self.assertIn("retained failure", log.read_text())

    def test_capture_drains_large_output_before_child_reads_large_input(self):
        command = [sys.executable, "-c", "import os,sys; sys.stdout.write('x'*131072); sys.stdout.flush(); "
                   "data=sys.stdin.read(); sys.stderr.write(str(len(data))+':'+os.environ['OWNED_TEST']); sys.exit(3)"]
        result = run_capture(command, "y" * 262144, 10, env={**os.environ, "OWNED_TEST": "private fixture"})
        self.assertEqual(result.returncode, 3)
        self.assertEqual(len(result.stdout), 131072)
        self.assertEqual(result.stderr, "262144:private fixture")

    def test_outer_timeout_owns_an_inner_helper_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ready, marker = root / "ready", root / "unexpected"
            child = ("import time; from pathlib import Path; Path(" + repr(str(ready))
                     + ").write_text('ready'); time.sleep(2); Path(" + repr(str(marker)) + ").write_text('orphan')")
            parent = ("import sys; from unibrain_agent.owned_process import run_logged; run_logged("
                      + repr([sys.executable, "-c", child]) + ", " + repr(str(root / "child.log")) + ", None)")
            with self.assertRaises(subprocess.TimeoutExpired):
                run_capture([sys.executable, "-c", parent], "", 1.5, Path(__file__).resolve().parents[1])
            self.assertTrue(ready.exists())
            time.sleep(2.2)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process identity protocol")
    def test_linux_cleanup_ignores_pid_reuse(self):
        target = Path('/proc/12345678')
        with patch('unibrain_agent.owned_process.Path.iterdir', side_effect=[[target], []]), \
                patch('unibrain_agent.owned_process.linux_identity', side_effect=['old-start', 'new-start']), \
                patch('unibrain_agent.owned_process.os.kill') as kill:
            stop_linux_owned('owned-token')
        kill.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process environment protocol")
    def test_linux_nested_capture_preserves_ancestral_owner_and_private_environment(self):
        script = "import os; print(os.environ['" + OWNER_KEY + "'])"
        result = run_capture([sys.executable, '-c', script], '', 5, env={**os.environ, OWNER_KEY: 'ancestor'})
        tokens = result.stdout.strip().split(',')
        self.assertIn('ancestor', tokens)
        inherited = {value for value in os.environ.get(OWNER_KEY, '').split(',') if value}
        self.assertEqual(len(tokens), len(inherited | {'ancestor'}) + 1)
        self.assertNotIn('ancestor', os.environ.get(OWNER_KEY, '').split(','))


if __name__ == "__main__":
    unittest.main()
