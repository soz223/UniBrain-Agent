"""Export and train matched model pairs, one fold at a time on one GPU."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_cohort_fold import export_fold
from unibrain_agent.cohort_provenance import validate_cohort
from unibrain_agent.paired_training import run
from unibrain_agent.training_supervisor import atomic_json


def execute(path):
    spec = json.loads(path.read_text())
    root = path.parent
    for item in spec['folds']:
        manifest = Path(item)
        pair = json.loads(manifest.read_text())
        data = manifest.parent / 'data'
        atomic_json(root / 'cross_validation.json', {'status': 'preparing', 'fold': pair['fold']})
        if not (data / 'cohort.json').exists():
            export_fold(Path(spec['cohort']), pair['fold'], data, full=True, dataset='hcp',
                        split_root=Path(spec['cohort']))
        audit = validate_cohort(data, 'hcp', pair['fold'], required_scope='full_fold', allow_missing=False)
        if not audit['family_disjoint_verified']:
            raise ValueError('Family-independent cohort required')
        if audit['split_counts']['train'] != pair['steps_per_epoch']:
            raise ValueError('Epoch budget does not match training cohort')
        atomic_json(root / 'cross_validation.json', {'status': 'training', 'fold': pair['fold'], 'epochs': pair['epochs']})
        run(pair, manifest.parent)
        state = json.loads((manifest.parent / 'queue.json').read_text())
        if state['status'] != 'complete':
            atomic_json(root / 'cross_validation.json', {'status': state['status'], 'fold': pair['fold']})
            return
    atomic_json(root / 'cross_validation.json', {'status': 'complete', 'folds': len(spec['folds'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    execute(args.manifest)
