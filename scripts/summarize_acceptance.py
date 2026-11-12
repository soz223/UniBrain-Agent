"""Revalidate completed cohort checks and export a public evidence table."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_acceptance_matrix import METHODS, validate_result
from unibrain_agent.data import digest


def summarize(matrix):
    matrix = Path(matrix).resolve()
    record = json.loads(matrix.read_text(encoding='utf-8'))
    rows = []
    for job in record['jobs']:
        dataset, method = job['dataset'], job['method']
        if dataset not in ('abide', 'adhd') or method not in METHODS:
            raise ValueError('Unexpected dataset or method')
        row = {key: job[key] for key in ('dataset', 'method', 'fold', 'seed', 'status')}
        row['optimizer_updates'] = 0
        if job['status'] == 'passed':
            path = matrix.parent / f'{dataset}_{method}' / 'report.json'
            measured = validate_result(job, path, record['optimizer_update_budget_per_job'])
            for key in ('report_sha256', 'updates_sha256', 'checkpoint_sha256'):
                if measured[key] != job[key]:
                    raise ValueError('Recorded acceptance evidence changed: ' + key)
            report = json.loads(path.read_text(encoding='utf-8'))
            row.update({key: measured[key] for key in ('optimizer_updates', 'changed_modules', 'trajectory',
                       'coordination', 'gradient_verification', 'report_sha256', 'updates_sha256', 'checkpoint_sha256')})
            row['parameters'] = report['architecture']['parameters']
            row['input_shape'] = [96, 96, 96]
            row['split_counts'] = report['audit']['splits']
            row['train_seconds'] = report['train_seconds']
            row['compact'] = False
        rows.append(row)
    return {'scope': record['scope'], 'matrix_sha256': digest(matrix), 'jobs': rows,
            'passed': sum(row['status'] == 'passed' for row in rows), 'total': len(rows),
            'boundary': 'Checks use subsets of the original folds. They do not establish cohort convergence, diagnostic benefit or Agent effectiveness.'}


def table(record):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(('Dataset', 'Method', 'Fold', 'Seed', 'Status', 'Updates', 'Parameters', 'Validation trend'))
    for row in record['jobs']:
        writer.writerow((row['dataset'], row['method'], row['fold'], row['seed'], row['status'],
                         row['optimizer_updates'], row.get('parameters', ''),
                         row.get('trajectory', {}).get('observed_trend', 'not_measured')))
    return stream.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evidence directory; existing reports are retained')
    result = summarize(args.matrix)
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'summary.csv').write_text(table(result), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'total': result['total']}))
