"""Reject code or dependency drift against a shared release manifest."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def fingerprint(root, source):
    files = {}
    for folder in ('unibrain_agent', 'scripts', 'configs', 'catalog'):
        for path in (root / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml'}:
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    upstream = {path.name: hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                for path in source.glob('*.py')}
    if not files or 'model.py' not in upstream:
        raise ValueError('Missing code or upstream model')
    packages = ('numpy', 'scipy', 'scikit-learn', 'torch', 'torch-geometric')
    return {'code': files, 'upstream_lf': upstream,
            'packages': {name: importlib.metadata.version(name) for name in packages}}


def verify(actual, expected):
    differences = []
    for section in ('code', 'upstream_lf', 'packages'):
        current, reference = actual[section], expected[section]
        differences.extend(f'{section}/{name}' for name in sorted(current.keys() | reference.keys())
                           if current.get(name) != reference.get(name))
    if differences:
        raise ValueError('Release mismatch: ' + ', '.join(differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    current = fingerprint(args.root, args.source)
    if args.create:
        with args.manifest.open('x', encoding='utf-8') as stream:
            json.dump(current, stream, indent=2, sort_keys=True)
    else:
        verify(current, json.loads(args.manifest.read_text(encoding='utf-8')))
    print(json.dumps({'status': 'passed', 'code_files': len(current['code']),
                      'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
