"""Read bounded, aggregate training telemetry without loading model checkpoints."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


TASKS = ('extraction', 'registration', 'segmentation', 'classification')
MAX_ROWS = 128
MAX_BYTES = 4 * 1024 * 1024


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def numeric_map(value, keys=TASKS):
    return {key: number(value[key]) for key in keys if isinstance(value, dict) and key in value}


def matrix(value):
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        return None
    return [[number(item) for item in row] for row in value]


def policy_record(value):
    if not isinstance(value, dict):
        return None
    result = {key: numeric_map(value.get(key, {})) for key in
              ('weights', 'task_losses', 'base_weights', 'actions', 'gradient_norms', 'raw_gradient_norms', 'gradient_scales',
               'action_logits', 'actor_objectives', 'critic_td_losses')}
    for key in ('gram', 'raw_gram', 'cosine_similarity'):
        result[key] = matrix(value.get(key))
    result.update({key: number(value.get(key)) for key in
                   ('reward', 'policy_updates', 'actor_optimizer_steps', 'critic_optimizer_steps', 'replay_size',
                    'model_steps', 'noise_scale', 'gradient_parameter_tensors', 'gradient_parameter_elements', 'action_bound')})
    result['reward_terms'] = numeric_map(value.get('reward_terms'),
                                       ('training_loss_improvement', 'gradient_norm_penalty', 'gradient_penalty_weight'))
    choices = {'profile': ('paper_core',), 'gradient_scope': ('all_trainable_parameters', 'recorded_probe_parameters'),
               'gradient_normalization': ('l2', 'none'), 'critic_action_space': ('raw_logits',),
               'update_order': ('model_then_policy',), 'noise_type': ('ornstein_uhlenbeck',),
               'weight_rule': ('softmax_div_gradient_scale',)}
    for key, allowed in choices.items():
        result[key] = value.get(key) if value.get(key) in allowed else None
    result['gradient_scope'] = result['gradient_scope'] or 'recorded_probe_parameters'
    result['model_step_completed'] = value.get('model_step_completed') is True
    result['validation_used'] = False if (value.get('reward_terms') or {}).get('validation_used') is False else None
    observations = value.get('observations')
    if isinstance(observations, dict):
        result['observations'] = {task: [number(x) for x in observations[task]] for task in TASKS
                                  if isinstance(observations.get(task), list) and len(observations[task]) == 5}
    return result


def objective_record(value):
    if not isinstance(value, dict):
        return None
    result = {key: value[key] for key in ('status', 'step', 'total', 'reconstructed_total') if key in value}
    fields = {'stages': ('id', 'name', 'method', 'loss_ids', 'branch'),
              'terms': ('id', 'name', 'definition', 'weight', 'value', 'requires_grad'),
              'modules': ('id', 'name'), 'connections': ('loss', 'module', 'status')}
    for key, allowed in fields.items():
        result[key] = [{field: item[field] for field in allowed if field in item}
                       for item in value.get(key, [])[:64] if isinstance(item, dict)]
    return result


def paper_layout(packet):
    if packet is None:
        return None
    for stage in packet.get('stages', []):
        if stage['id'] == 'parcellation':
            stage['method'] = 'Inverse affine, trilinear atlas'
    packet['forward_edges'] = [['input', 'extraction'], ['input', 'segmentation'], ['input', 'graph'],
                               ['extraction', 'registration'], ['registration', 'parcellation'],
                               ['parcellation', 'graph'], ['graph', 'classification']]
    packet['layout_source'] = 'PaperUniBrain.forward; template inputs omitted from schematic'
    return packet


def tail_records(path):
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - MAX_BYTES)
        stream.seek(start)
        if start:
            stream.readline()
        lines = stream.read(MAX_BYTES).splitlines()
    rows = []
    for line in lines[-MAX_ROWS - 1:]:
        try:
            row = json.loads(line)
            if isinstance(row, dict) and number(row.get('step')) is not None:
                rows.append(row)
        except (ValueError, UnicodeDecodeError):
            continue
    return rows[-MAX_ROWS:]


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def health_record(folder):
    def read(name):
        try:
            return json.loads((folder / name).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
    health, supervisor = read('health.json'), read('supervisor.json')
    diagnosis = read('training_diagnosis.json')
    if not health and not supervisor:
        return None
    last = health.get('last_recovery', {})
    allowed = {'ext_net', 'reg_net', 'feature_net', 'gnn_pred', 'classification_to_input',
               'ext_net:total', 'reg_net:total', 'feature_net:total', 'gnn_pred:total'}
    return {'phase': health.get('phase') if health.get('phase') in ('training', 'validation') else None,
            'diagnosis': {'epoch': number(diagnosis.get('epoch')),
                          'warnings': [value for value in diagnosis.get('warnings', []) if value in
                                       ('training_validation_gap', 'five_validation_epochs_near_chance')],
                          'targets': {name: {**numeric_map(value, ('target', 'observed')), 'met': value.get('met')}
                                      for name, value in diagnosis.get('targets', {}).items() if name in
                                      ('brain_dice', 'brain_jaccard', 'tissue_dice', 'tissue_jaccard', 'atlas_dice',
                                       'atlas_jaccard', 'registration_mi', 'registration_cc', 'accuracy', 'auc')}},
            **numeric_map(health, ('step', 'epoch', 'recovery_count')),
            'last_recovery': {'step': number(last.get('step')), 'paths': [p for p in last.get('paths', []) if p in allowed],
                              'optimizer_modified': False if last.get('optimizer_modified') is False else None},
            'supervisor': {'status': supervisor.get('status') if supervisor.get('status') in
                          ('starting', 'running', 'waiting_for_gpu', 'checkpointing', 'recovering', 'complete', 'budget_exhausted', 'stopped', 'needs_attention') else None,
                          **numeric_map(supervisor, ('restarts', 'max_restarts', 'heartbeat_age_seconds')),
                          'updated_at': supervisor.get('updated_at')}}


def snapshot(folder, identifier, label=None):
    folder = Path(folder)
    mirror = folder / 'snapshot.json'
    if mirror.is_file():
        packet = json.loads(mirror.read_text(encoding='utf-8'))
        packet.update(id=identifier, label=label or identifier)
        return packet
    report_path, updates_path = folder / 'report.json', folder / 'updates.jsonl'
    try:
        report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
    except (ValueError, UnicodeDecodeError):
        report = {}
    rows = tail_records(updates_path)
    history = [{**numeric_map(row, ('step', 'epoch', 'total_loss', 'weighted_loss')),
                'losses': numeric_map(row.get('losses')), 'gradient_policy': policy_record(row.get('gradient_policy')),
                'objective': objective_record(row.get('objective'))} for row in rows]
    config = report.get('config', {})
    if config.get('paper_original'):
        for row in history:
            row['objective'] = paper_layout(row['objective'])
    metric_keys = ('accuracy', 'balanced_accuracy', 'auc', 'brain_dice', 'tissue_dice', 'atlas_dice', 'registration_dice',
                   'brain_jaccard', 'tissue_jaccard', 'atlas_jaccard', 'registration_mi', 'registration_cc',
                   'classification_loss', 'probability_std', 'predicted_positive_fraction')
    validation = [{**numeric_map(row, ('epoch',)), 'validation': numeric_map(row.get('validation'), metric_keys),
                   'training_probe': numeric_map(row.get('training_probe'), metric_keys)}
                  for row in report.get('history', [])[-1000:]]
    files = [path for path in (report_path, updates_path) if path.is_file()]
    return {'id': identifier, 'label': label or identifier, 'source': 'training_files',
            'health': health_record(folder),
            'reported_status': report.get('status', 'no_records'),
            'stop_reason': report.get('stop_reason') if report.get('stop_reason') in
                           ('wall_time_budget', 'step_budget', 'termination_signal', 'validation_patience') else None,
            'training_probe_size': number(report.get('training_probe', {}).get('size')),
            'dataset': config.get('dataset') if config.get('dataset') in ('abide', 'adhd', 'hcp') else None,
            'fold': number(config.get('fold')), 'epochs': number(config.get('epochs')),
            'paper_original': config.get('paper_original') is True,
            'coordination': 'taskforce' if config.get('coordination') == 'taskforce' else 'none',
            'observed_at': timestamp(max(path.stat().st_mtime for path in files)) if files else None,
            'captured_at': datetime.now(timezone.utc).isoformat(), 'history': history, 'validation': validation}


class TrainingMonitor:
    def __init__(self, entries=None, base=None):
        if entries is None:
            from .paths import LOCAL, CONFIG
            entries, base = LOCAL.get('training_runs', {}), CONFIG.parent
        self.entries = entries
        self.base = Path(base or '.')

    def listing(self):
        return [{'id': key, 'label': value.get('label', key) if isinstance(value, dict) else key}
                for key, value in self.entries.items() if re.fullmatch(r'[A-Za-z0-9_]+', key)]

    def read(self, identifier):
        if not re.fullmatch(r'[A-Za-z0-9_]+', identifier) or identifier not in self.entries:
            raise ValueError('Unknown monitored run')
        value = self.entries[identifier]
        path = Path(value['path'] if isinstance(value, dict) else value).expanduser()
        path = path if path.is_absolute() else self.base / path
        return snapshot(path.resolve(), identifier, value.get('label') if isinstance(value, dict) else None)
