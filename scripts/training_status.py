"""Show progress from training artifacts without loading checkpoints."""

import argparse
import json
from pathlib import Path


def latest_update(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 262144))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if 'step' in value:
            return value
    return {}


def show(root):
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        report_path = folder / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        update = latest_update(folder / 'updates.jsonl')
        if not report and not update:
            continue
        print(folder.name)
        print(f"  Status: {report.get('status', 'initializing')} | Epoch: {update.get('epoch', '?')} | Step: {update.get('step', report.get('steps', 0))}")
        if update:
            print('  Latest losses: ' + json.dumps(update.get('losses', {})))
        history = report.get('history', [])
        if history:
            print('  Latest validation: ' + json.dumps(history[-1]))
        if report.get('error'):
            print('  Error: ' + report['error'])
        print('  Files: ' + str(folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    show(parser.parse_args().root)
