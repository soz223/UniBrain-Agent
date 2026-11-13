"""Mirror aggregate training records over an existing SSH login."""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def sync(host, entries, output, python='python3'):
    reader = Path(__file__).resolve().parents[1] / 'unibrain_agent' / 'training_monitor.py'
    code = reader.read_text(encoding='utf-8')
    code += '\nimport sys\nentries=json.loads(sys.argv[1])\nprint(json.dumps([snapshot(path, key) for key,path in entries.items()], allow_nan=False))\n'
    command = shlex.join([python, '-', json.dumps(entries)])
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
                            input=code, text=True, encoding='utf-8', capture_output=True, timeout=45, check=True)
    for packet in json.loads(result.stdout):
        folder = output / packet['id']
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / 'snapshot.tmp'
        temporary.write_text(json.dumps(packet, allow_nan=False), encoding='utf-8')
        temporary.replace(folder / 'snapshot.json')
        latest = packet['history'][-1] if packet['history'] else {}
        print(packet['id'], 'step', latest.get('step'), 'observed', packet['observed_at'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--runs', type=Path, required=True, help='JSON mapping of safe run IDs to remote directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=0)
    parser.add_argument('--python', default='python3')
    args = parser.parse_args()
    import re
    entries = json.loads(args.runs.read_text(encoding='utf-8'))
    if not entries or any(not re.fullmatch(r'[A-Za-z0-9_]+', key) for key in entries):
        parser.error('Run IDs must contain only letters, digits and underscores')
    if args.host.startswith('-') or args.interval < 0:
        parser.error('Invalid host or interval')
    while True:
        try:
            sync(args.host, entries, args.output, args.python)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print('Telemetry refresh failed:', type(error).__name__, flush=True)
            if not args.interval:
                raise
        if not args.interval:
            break
        time.sleep(max(10, args.interval))


if __name__ == '__main__':
    main()
