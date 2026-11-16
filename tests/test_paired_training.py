import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unibrain_agent.paired_training import run


class PairedTrainingTests(unittest.TestCase):
    def execute(self, directory, mismatch=False, busy=False):
        seen = []
        jobs = [{'name': name, 'cwd': str(directory), 'output': str(directory / name),
                 'command': ['python', 'train.py', '--output', str(directory / name)]}
                for name in ('original', 'agent')]
        def write(command):
            folder = Path(command[command.index('--output') + 1])
            steps = int(command[command.index('--max-steps') + 1])
            seen.append((folder.name, steps))
            folder.mkdir(exist_ok=True)
            (folder / 'last.pt').touch()
            (folder / 'report.json').write_text(json.dumps({'status': 'budget_exhausted', 'steps': steps,
                'architecture': {'initial_state_sha256': folder.name if mismatch else 'same'}}))
        class Child:
            returncode = 0
            def __init__(self, command, **kwargs):
                write(command)
            def poll(self):
                return 0
        def supervise(spec):
            self.assertIn('--resume', spec['command'])
            self.assertTrue(spec['resume_supervisor_state'])
            write(spec['command'])
            return {'status': 'budget_exhausted'}
        with patch('unibrain_agent.paired_training.gpu_available', side_effect=[RuntimeError('busy'), None, None] if busy else None), \
             patch('unibrain_agent.paired_training.time.sleep'), \
             patch('unibrain_agent.paired_training.subprocess.Popen', Child), \
             patch('unibrain_agent.paired_training.supervise', supervise):
            run({'jobs': jobs, 'steps_per_epoch': 3, 'epochs': 2}, directory)
        return seen

    def test_waits_for_gpu_release_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            seen = self.execute(Path(folder), busy=True)
            self.assertEqual(seen[:2], [('original', 2), ('agent', 2)])
            self.assertEqual(json.loads((Path(folder) / 'queue.json').read_text())['status'], 'complete')

    def test_alternates_matched_budgets_and_resumes_each_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            seen = self.execute(Path(folder))
            self.assertEqual(seen, [('original', 2), ('agent', 2), ('original', 3), ('agent', 3), ('original', 6), ('agent', 6)])

    def test_mismatched_initialization_stops_before_long_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'identical initialized'):
                self.execute(Path(folder), mismatch=True)
            self.assertEqual(json.loads((Path(folder) / 'queue.json').read_text())['status'], 'needs_attention')
