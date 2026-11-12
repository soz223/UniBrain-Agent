"""Revalidate completed cohort checks and export a public evidence table."""

import argparse
import csv
import io
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_acceptance_matrix import METHODS, validate_result
from unibrain_agent.data import digest


def summarize(matrix):
    matrix = Path(matrix).resolve()
    record = json.loads(matrix.read_text(encoding='utf-8'))
    rows = []
    identities = set()
    for job in record['jobs']:
        dataset, method = job['dataset'], job['method']
        if dataset not in ('abide', 'adhd') or method not in METHODS:
            raise ValueError('Unexpected dataset or method')
        identity = (dataset, method, job['fold'], job['seed'])
        if identity in identities:
            raise ValueError('Duplicate job in one acceptance matrix')
        identities.add(identity)
        row = {key: job[key] for key in ('dataset', 'method', 'fold', 'seed', 'status')}
        coordination = job.get('coordination', 'none')
        row.update(optimizer_updates=0, setting=job.get('setting'),
                   coordination_mode=coordination.get('mode', 'none') if isinstance(coordination, dict) else coordination,
                   data_scope=(job.get('data_audit') or {}).get('scope'),
                   data_fingerprint=canonical_hash((job.get('data_audit') or {}).get('data_hashes', {})),
                   strict_current_gradient_verified=False, gradient_boundary_scope='not_verified',
                   matrix_sha256=digest(matrix))
        if job['status'] == 'passed':
            path = matrix.parent / f'{dataset}_{method}' / 'report.json'
            measured = validate_result(job, path, record['optimizer_update_budget_per_job'])
            for key in ('report_sha256', 'updates_sha256', 'checkpoint_sha256'):
                if measured[key] != job[key]:
                    raise ValueError('Recorded acceptance evidence changed: ' + key)
            report = json.loads(path.read_text(encoding='utf-8'))
            row.update({key: measured[key] for key in ('optimizer_updates', 'changed_modules', 'trajectory',
                       'coordination', 'gradient_verification', 'report_sha256', 'updates_sha256', 'checkpoint_sha256',
                       'strict_current_gradient_verified', 'gradient_boundary_scope', 'checkpoint_policy')})
            row['parameters'] = report['architecture']['parameters']
            row['input_shape'] = [96, 96, 96]
            row['split_counts'] = report['audit']['splits']
            row['train_seconds'] = report['train_seconds']
            row['compact'] = False
        rows.append(row)
    return {'scope': record['scope'], 'matrix_sha256': digest(matrix), 'jobs': rows,
            'passed': sum(row['status'] == 'passed' for row in rows), 'total': len(rows),
            'strict_passed': sum(row['status'] == 'passed' and row['strict_current_gradient_verified'] for row in rows),
            'legacy_passed': sum(row['status'] == 'passed' and not row['strict_current_gradient_verified'] for row in rows),
            'boundary': 'Checks use subsets of the original folds. They do not establish cohort convergence, diagnostic benefit or Agent effectiveness.'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize_cycle(path):
    from unibrain_agent.integration_benchmark import verdict
    path = Path(path).resolve()
    cycle = json.loads(path.read_text(encoding='utf-8'))
    training_path = path.parent / 'training/report.json'
    training = json.loads(training_path.read_text(encoding='utf-8'))
    if cycle.get('status') != 'complete' or training != cycle.get('training'):
        raise ValueError('Cycle must retain its complete matching training report')
    config = training.get('config', {})
    method, steps = training.get('method'), training.get('steps')
    if (method not in METHODS or method != cycle.get('method') or config.get('dataset') not in ('abide', 'adhd')
            or config.get('fold') not in range(1, 6) or training.get('seed') != cycle.get('seed')):
        raise ValueError('Cycle method, dataset, fold or seed identity is invalid')
    if config.get('max_steps') != steps:
        raise ValueError('This acceptance exporter requires an exact bounded cycle; full cycles belong to the paper suite')
    accepted = cycle.get('accepted_adapter', {})
    if accepted.get('file') != 'accepted_adapter.py':
        raise ValueError('Cycle adapter is not retained in its declared location')
    adapter_hash = digest(path.parent / 'accepted_adapter.py')
    architecture = training.get('architecture', {})
    if (accepted.get('sha256') != adapter_hash or cycle.get('accepted_adapter_unchanged_after_training') is not True
            or architecture.get('generated_adapter_active_in_training') is not True
            or architecture.get('generated_adapter_sha256') != adapter_hash):
        raise ValueError('Cycle did not execute the retained unchanged adapter')
    integration = cycle.get('integration', {})
    matching = [call for call in integration.get('calls', []) if call.get('adapter_sha256') == adapter_hash
                and verdict(call.get('evaluation', {})) == 'passed']
    if integration.get('status') != 'passed' or integration.get('final_valid') is not True or not matching:
        raise ValueError('Cycle accepted code lacks successful independent validation')
    updates = cycle.get('actual_updates', {})
    accepted_steps = updates.get('acceptance')
    if (type(accepted_steps) is not int or not 1 <= accepted_steps <= cycle.get('budgets', {}).get('acceptance_updates', 0)
            or integration.get('optimizer_updates') != accepted_steps or updates.get('training') != steps):
        raise ValueError('Cycle acceptance and training update counts disagree')
    data_hashes = training.get('audit', {}).get('data_hashes', {})
    if not data_hashes or not any(call['evaluation'].get('data', {}).get('sha256') == data_hashes.get('adhd_train_sample.npy') for call in matching):
        raise ValueError('Cycle acceptance and training used different data')
    job = {'dataset': config['dataset'], 'method': method, 'fold': config['fold'], 'seed': training['seed'],
           'setting': training['setting'], 'data_audit': {'data_hashes': data_hashes}, 'verify_gradients': True,
           'gradient_protocol_version': 2, 'coordination': config.get('coordination', 'none'),
           'checkpoint_mode': config.get('checkpoint_mode'), 'phase': 'acceptance'}
    measured = validate_result(job, training_path, steps)
    return {'cycle_sha256': digest(path), 'dataset': config['dataset'], 'method': method, 'fold': config['fold'],
            'seed': training['seed'], 'status': 'passed', 'acceptance_updates': accepted_steps, 'training_updates': steps,
            'adapter_sha256': adapter_hash, 'training_report_sha256': measured['report_sha256'],
            'strict_current_gradient_verified': measured['strict_current_gradient_verified'],
            'scope': 'Agent cycle evidence, separate from candidate method coverage'}


def merge_summaries(records, cycles=()):
    observations, executions, groups = [], {}, {}
    for record in records:
        for row in record['jobs']:
            observations.append(row)
            key = tuple(row.get(name) for name in ('dataset', 'method', 'fold', 'seed', 'setting', 'coordination_mode'))
            groups.setdefault(key, []).append(row)
            if row['status'] == 'passed':
                identity = row['report_sha256']
                if identity in executions:
                    previous = executions[identity]
                    if any(previous.get(name) != row.get(name) for name in ('updates_sha256', 'checkpoint_sha256', 'dataset', 'method', 'fold', 'seed')):
                        raise ValueError('One execution hash has conflicting evidence')
                executions[identity] = row
    coverage = []
    for rows in groups.values():
        known = [row for row in rows if row.get('data_fingerprint') != canonical_hash({})]
        if len({(row.get('data_scope'), row.get('data_fingerprint')) for row in known}) > 1:
            raise ValueError('Cannot combine different data exports in one experiment cell')
        first = known[0] if known else rows[0]
        strict = {row['report_sha256'] for row in rows if row['status'] == 'passed' and row['strict_current_gradient_verified']}
        legacy = {row['report_sha256'] for row in rows if row['status'] == 'passed' and not row['strict_current_gradient_verified']}
        coverage.append({**{name: first.get(name) for name in ('dataset', 'method', 'fold', 'seed', 'setting', 'coordination_mode', 'data_scope', 'data_fingerprint')},
                         'status': 'strict_passed' if strict else 'legacy_only' if legacy else 'not_verified',
                         'strict_executions': len(strict), 'legacy_executions': len(legacy),
                         'observed_statuses': sorted({row['status'] for row in rows})})
    unique_cycles = {row['cycle_sha256']: row for row in cycles}
    all_training = set(executions) | {row['training_report_sha256'] for row in unique_cycles.values()}
    return {'scope': 'Merged bounded evidence with distinct execution and method coverage counts',
            'matrices': sorted({record['matrix_sha256'] for record in records}), 'jobs': observations,
            'executions': list(executions.values()), 'execution_count': len(executions),
            'execution_count_scope': 'Distinct revalidated passing training reports; other statuses remain in observations',
            'observed_status_counts': dict(Counter(row['status'] for row in observations)),
            'strict_execution_count': sum(row['strict_current_gradient_verified'] for row in executions.values()),
            'legacy_execution_count': sum(not row['strict_current_gradient_verified'] for row in executions.values()),
            'coverage': coverage, 'coverage_total': len(coverage),
            'strict_coverage_passed': sum(row['status'] == 'strict_passed' for row in coverage),
            'legacy_only_coverage': sum(row['status'] == 'legacy_only' for row in coverage),
            'cycles': list(unique_cycles.values()), 'cycle_count': len(unique_cycles),
            'unique_training_executions_including_cycles': len(all_training),
            'boundary': 'All observations are retained. Copied execution reports count once. Legacy runs cannot establish protocol 2 coverage. Agent cycles do not enlarge candidate coverage.'}


def table(record):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(('Dataset', 'Method', 'Fold', 'Seed', 'Status', 'Strict protocol 2', 'Gradient scope', 'Updates', 'Parameters',
                     'Validation trend', 'Execution SHA256', 'Matrix SHA256'))
    for row in record['jobs']:
        writer.writerow((row['dataset'], row['method'], row['fold'], row['seed'], row['status'],
                         row['strict_current_gradient_verified'], row['gradient_boundary_scope'],
                         row['optimizer_updates'], row.get('parameters', ''),
                         row.get('trajectory', {}).get('observed_trend', 'not_measured'), row.get('report_sha256', ''), row['matrix_sha256']))
    return stream.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, nargs='+', required=True)
    parser.add_argument('--cycle', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evidence directory; existing reports are retained')
    summaries = [summarize(path) for path in args.matrix]
    result = merge_summaries(summaries, [summarize_cycle(path) for path in args.cycle])
    if len(summaries) == 1 and not args.cycle:
        result = {**summaries[0], **result}
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'summary.csv').write_text(table(result), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('execution_count', 'strict_coverage_passed', 'coverage_total', 'cycle_count')}))
