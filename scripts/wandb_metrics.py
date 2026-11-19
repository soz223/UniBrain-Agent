"""Read an allowlisted batch of UniBrain scalars, locally or over SSH."""
import json
import math
import sys
from pathlib import Path

TASKS = ('classification', 'extraction', 'registration', 'segmentation', 'parcellation', 'regularization')
METRICS = ('auc', 'accuracy', 'balanced_accuracy', 'macro_f1', 'sensitivity', 'specificity',
           'brain_dice', 'brain_jaccard', 'tissue_dice', 'tissue_jaccard', 'atlas_dice',
           'atlas_jaccard', 'registration_mi', 'registration_cc', 'reference_loss',
           *(task + '_loss' for task in TASKS))


def scalars(values, keys, prefix):
    return {prefix + key: values[key] for key in keys
            if type(values.get(key)) in (int, float) and math.isfinite(values[key])}


def read_batch(folder, offset, previous_step, previous_epoch):
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    events = []
    path = folder / 'updates.jsonl'
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(offset if path.stat().st_size >= offset else 0)
            while len(events) < 1024:
                line = stream.readline()
                if not line.endswith(b'\n'):
                    break
                offset = stream.tell()
                row = json.loads(line)
                step = row.get('step', 0)
                if step <= previous_step or not row.get('losses'):
                    continue
                event = {'optimizer_step': step, **scalars(row['losses'], TASKS, 'train/')}
                event.update(scalars(row, ('total_loss',), 'train/'))
                policy = row.get('gradient_policy') or {}
                event.update(scalars(policy, ('reward', 'accepted_action_fraction', 'trust_radius'), 'agent/'))
                event.update(scalars(policy.get('weights', {}), TASKS, 'weight/'))
                events.append(event)
                previous_step = step
    for row in report.get('history', []):
        epoch = row['epoch']
        if epoch > previous_epoch:
            events.append({'epoch': epoch, **scalars(row.get('validation', {}), METRICS, 'val/'),
                           **scalars(row.get('training_probe', {}), METRICS, 'probe/')})
            previous_epoch = epoch
    return {'events': events, 'cursor': [offset, previous_step, previous_epoch],
            'status': report.get('status'), 'ready': bool(report),
            'more': path.exists() and offset < path.stat().st_size}


if __name__ == '__main__':
    print(json.dumps(read_batch(Path(sys.argv[1]), *map(int, sys.argv[2:5]))))
