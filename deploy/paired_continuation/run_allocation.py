import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

manifest, batch = map(Path, sys.argv[1:3])
root = Path(__file__).parent
subprocess.run(['nvidia-smi'], check=True)
child = subprocess.Popen([sys.executable, '-u', str(root / 'launch_pair.py'), str(manifest)], start_new_session=True)
stopping = threading.Event()
rotation = threading.Event()


def checkpoint_and_stop(*args):
    stopping.set()
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def rotate_allocation():
    rotation.set()
    checkpoint_and_stop()


signal.signal(signal.SIGTERM, checkpoint_and_stop)
timer = threading.Timer(23 * 3600 + 45 * 60, rotate_allocation)
timer.daemon = True
timer.start()
code = child.wait()
timer.cancel()
state = json.loads((manifest.parent / 'queue.json').read_text())
if rotation.is_set() and state['status'] == 'stopped':
    submitted = subprocess.check_output(['sbatch', '--parsable', '--dependency=afterany:' + os.environ['SLURM_JOB_ID'], str(batch)], text=True).strip()
    with (manifest.parent / 'allocations.jsonl').open('a') as stream:
        stream.write(json.dumps({'previous_job': os.environ['SLURM_JOB_ID'], 'next_job': submitted,
                                'reason': 'Allocation budget reached; checkpoint continuation'}) + '\n')
    print('Continuation submitted:', submitted, flush=True)
elif state['status'] != 'complete':
    raise SystemExit(code or 1)
