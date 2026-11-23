"""Generate a restricted Linux user service for the classroom backend."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def quote(value):
    value = str(value)
    if any(character in value for character in '\n\r\x00'):
        raise ValueError('Unit values must be single line')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'


def unit_text(project, python, config, environment, state, assets):
    return f'''[Unit]
Description=UniBrain classroom inference
After=network-online.target

[Service]
Type=simple
WorkingDirectory={quote(project)}
EnvironmentFile={quote(environment)}
Environment=CUDA_VISIBLE_DEVICES=
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=OPENBLAS_NUM_THREADS=2
Environment=MPLCONFIGDIR={quote(state / 'matplotlib')}
Environment=XDG_CACHE_HOME={quote(state / 'cache')}
Environment=CODEX_HOME={quote(state / 'codex')}
ExecStart={quote(python)} -m unibrain_agent.classroom --config {quote(config)}
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
KillMode=control-group
UMask=0077
Nice=10
CPUQuota=200%
MemoryMax=12G
MemorySwapMax=0
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths={quote(state)}
ReadOnlyPaths={quote(assets)} {quote(project)} {quote(config)} {quote(environment)}

[Install]
WantedBy=default.target
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'env-file', 'project', 'python', 'assets'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--name', default='unibrain-classroom.service')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Install the user service on Linux')
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+\.service', args.name):
        parser.error('Use a simple service name')
    project, python, config, environment, assets = [path.resolve(strict=True) for path in
                                                  (args.project, args.python, args.config, args.env_file, args.assets)]
    if environment.stat().st_mode & 0o077:
        parser.error('The environment file must be private: chmod 600')
    settings = json.loads(config.read_text(encoding='utf-8'))
    if settings.get('host', '127.0.0.1') != '127.0.0.1':
        parser.error('The classroom service must bind to loopback')
    state = Path(settings['state']).resolve()
    command = settings.get('worker_command', [])
    for flag in ('--source', '--checkpoint', '--data', '--templates'):
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            parser.error('The fixed inference command needs all four asset paths')
        path = Path(command[command.index(flag) + 1]).resolve(strict=True)
        if path != assets and assets not in path.parents:
            parser.error('Every worker input must be an isolated asset copy')
    for readonly in (project, assets, config, environment):
        if readonly == state or state in readonly.parents or readonly in state.parents:
            parser.error('Keep demo state separate from code, configuration and assets')
    if not os.access(python, os.X_OK) or not (project / 'unibrain_agent/classroom.py').is_file():
        parser.error('Select an installed Python environment and the demo checkout')
    unit = Path.home() / '.config/systemd/user' / args.name
    if unit.exists():
        raise FileExistsError('Review the existing service before replacing it')
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ('cache', 'matplotlib', 'codex'):
        (state / name).mkdir(mode=0o700, exist_ok=True)
    unit.parent.mkdir(parents=True, exist_ok=True)
    with unit.open('x', encoding='utf-8') as stream:
        stream.write(unit_text(project, python, config, environment, state, assets))
    subprocess.run(['systemd-analyze', '--user', 'verify', str(unit)], check=True)
    if args.start:
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
        subprocess.run(['systemctl', '--user', 'enable', '--now', args.name], check=True)
        subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.name], check=True)
    print(json.dumps({'service': args.name, 'started': args.start, 'unit': str(unit)}))


if __name__ == '__main__':
    main()
