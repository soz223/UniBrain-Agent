"""Publish UniBrain metrics to W&B from a separate monitoring process."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import wandb
from wandb_metrics import read_batch


def delta(command):
    return subprocess.check_output(['ssh', '-S', str(Path.home() / '.ssh/unibrain_delta/master.sock'),
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'szhao2@dt-login03.delta.ncsa.illinois.edu',
        shlex.join(command)], text=True, timeout=60)


def gpu_stats(spec):
    query = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
             '--format=csv,noheader,nounits']
    if spec.get('slurm_name'):
        jobs = delta(['squeue', '--me', '--name', spec['slurm_name'], '-t', 'RUNNING', '-h', '-o', '%A']).split()
        if len(jobs) != 1 or not jobs[0].isdigit():
            return {}
        output = delta(['srun', '--jobid='+jobs[0], '--overlap', '--immediate=3', '-N1', '-n1', '-c1',
                        '--gpus-per-task=1', 'bash', '-c', 'exec "$@" -i "${CUDA_VISIBLE_DEVICES%%,*}"', 'bash', *query])
    else:
        folder = Path(spec['path'])
        state = folder / 'supervisor.json'
        if not state.exists():
            return {}
        supervisor = json.loads(state.read_text())
        if supervisor.get('status') != 'running':
            return {}
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                            '--format=csv,noheader'], text=True)
        devices = [line.split(',')[1].strip() for line in processes.splitlines()
                   if line.split(',')[0].strip() == str(supervisor.get('worker_pid'))]
        if len(devices) != 1:
            return {}
        output = subprocess.check_output([*query, '-i', devices[0]], text=True)
    values = output.strip().split(',')
    if len(values) != 4:
        return {}
    return {'gpu/'+key:float(value) for key,value in zip(
        ('utilization_pct','memory_MiB','power_W','temperature_C'),values) if value.strip() not in ('N/A','[N/A]')}


def monitor(spec, state):
    cursor = json.loads(state.read_text()) if state.exists() else [0,0,0]
    run = None
    last_hardware = 0
    while True:
        if 'remote_reader' in spec:
            try:
                batch = json.loads(delta(['python3',spec['remote_reader'],spec['path'],*map(str,cursor)]))
            except (OSError, subprocess.SubprocessError):
                time.sleep(60)
                continue
        else:
            batch = read_batch(Path(spec['path']), *cursor)
        if batch['ready'] and run is None:
            run = wandb.init(entity='soz223-lehigh-university', project='unibrain', name=spec['name'],
                id=hashlib.sha256(spec['name'].encode()).hexdigest()[:16], resume='allow',
                config={'dataset':spec['dataset'],'fold':1,'method':spec['method']},
                settings=wandb.Settings(x_disable_stats=True,disable_git=True,save_code=False,console='off'))
            for prefix, axis in [('train','optimizer_step'),('agent','optimizer_step'),('weight','optimizer_step'),
                                 ('val','epoch'),('probe','epoch'),('gpu','sample_time')]:
                run.define_metric(prefix+'/*',step_metric=axis)
        if run:
            if not any(cursor) and 'remote_reader' not in spec:
                probes = Path(spec['path']) / 'early_auc.jsonl'
                if probes.exists():
                    run.define_metric('early_validation/*', step_metric='probe_step')
                    for line in probes.read_text().splitlines():
                        row = json.loads(line)
                        run.log({'probe_step': row['step'], 'early_validation/auc': row['auc'],
                                 'early_validation/accuracy': row['accuracy']})
            for event in batch['events']:
                run.log(event)
            cursor = batch['cursor']
            tmp = state.with_suffix('.tmp')
            tmp.write_text(json.dumps(cursor))
            tmp.replace(state)
            if not batch['more'] and batch['status']=='complete':
                run.finish()
                return
            if time.monotonic()-last_hardware>=120:
                try:
                    hw=gpu_stats(spec)
                    if hw:run.log({'sample_time':time.time(),**hw})
                except (OSError,ValueError,subprocess.SubprocessError):
                    pass
                last_hardware=time.monotonic()
        if not batch['more'] or not run:
            time.sleep(45)


if __name__=='__main__':
    specfile=Path(sys.argv[1])
    os.environ.setdefault('WANDB_DISABLE_CODE','true')
    monitor(json.loads(specfile.read_text()),specfile.with_suffix('.cursor'))
