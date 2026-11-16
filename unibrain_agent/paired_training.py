"""Alternate matched training budgets on one available GPU."""

import argparse
import copy
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .training_supervisor import atomic_json, gpu_available, option, read_json, supervise


def budgets(steps_per_epoch, epochs):
    if steps_per_epoch < 2 or epochs < 1:
        raise ValueError('Positive epochs and at least two training subjects are required')
    return sorted(set([2] + [steps_per_epoch * epoch for epoch in range(1, epochs + 1)]))


def run(spec, directory):
    directory.mkdir(parents=True, exist_ok=True)
    jobs = spec['jobs']
    if len(jobs) != 2 or len({job['output'] for job in jobs}) != 2:
        raise ValueError('Two distinct experiment directories are required')
    stopping = []
    handlers = {sig: signal.signal(sig, lambda number, frame: stopping.append(number))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for limit in budgets(spec['steps_per_epoch'], spec['epochs']):
            for job in jobs:
                if stopping:
                    atomic_json(directory / 'queue.json', {'status': 'stopped'})
                    return
                folder = Path(job['output'])
                report = read_json(folder / 'report.json')
                if report.get('steps', 0) >= limit and report.get('status') != 'failed':
                    continue
                current = copy.deepcopy(job)
                current['command'] = option(current['command'], '--max-steps', limit)
                atomic_json(directory / 'queue.json', {'status': 'running', 'active': job['name'], 'target_steps': limit})
                if (folder / 'last.pt').is_file():
                    current['command'] = option(current['command'], '--resume', folder / 'last.pt')
                    current['resume_supervisor_state'] = True
                    status = supervise(current)
                    if status['status'] == 'stopped':
                        atomic_json(directory / 'queue.json', {'status': 'stopped'})
                        return
                else:
                    if folder.exists():
                        raise ValueError('Existing experiment lacks a resume checkpoint: ' + job['name'])
                    gpu_available(job.get('environment', {}))
                    with (directory / (job['name'] + '_initial.log')).open('ab') as log:
                        child = subprocess.Popen(current['command'], cwd=job['cwd'],
                            env={**os.environ, **job.get('environment', {})}, stdout=log, stderr=subprocess.STDOUT)
                        signaled = False
                        while child.poll() is None:
                            if stopping and not signaled:
                                child.terminate()
                                signaled = True
                            time.sleep(1)
                        if stopping:
                            atomic_json(directory / 'queue.json', {'status': 'stopped'})
                            return
                        if child.returncode:
                            raise RuntimeError('Initial full-model check failed: ' + job['name'])
                report = read_json(folder / 'report.json')
                if report.get('status') not in ('budget_exhausted', 'complete') or report.get('steps', 0) < limit:
                    raise RuntimeError('Training did not reach its matched budget: ' + job['name'])
                state = read_json(folder / 'supervisor.json')
                state['status'] = 'complete' if report['status'] == 'complete' else 'waiting_for_turn'
                atomic_json(folder / 'supervisor.json', state)
            reports = [read_json(Path(job['output']) / 'report.json') for job in jobs]
            hashes = [r.get('architecture', {}).get('initial_state_sha256') for r in reports]
            if not all(hashes) or len(set(hashes)) != 1:
                raise RuntimeError('Paired runs do not have identical initialized model states')
            atomic_json(directory / 'paired_acceptance.json', {'status': 'passed', 'matched_steps': limit,
                'initial_state_sha256': hashes[0], 'test_evaluated': False})
        atomic_json(directory / 'queue.json', {'status': 'complete'})
    except Exception as error:
        atomic_json(directory / 'queue.json', {'status': 'needs_attention', 'error': str(error)})
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text()), args.manifest.parent)
