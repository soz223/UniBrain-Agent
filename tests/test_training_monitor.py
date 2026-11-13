import json
import tempfile
import unittest
from pathlib import Path

from unibrain_agent.training_monitor import TrainingMonitor, MAX_ROWS, snapshot


def check_records_are_bounded_and_exclude_subjects_and_paths(tmp_path):
    (tmp_path / 'report.json').write_text(json.dumps({'status':'running', 'config':{'data':'private/path', 'dataset':'abide', 'fold':1}, 'history':[]}))
    with (tmp_path / 'updates.jsonl').open('w') as stream:
        for step in range(160):
            stream.write(json.dumps({'step':step, 'subject_id':'private_subject', 'losses':{'extraction':.2},
                'gradient_policy':{'reward_terms':None, 'weights':{'extraction':float('nan')}, 'private':'secret'}})+'\n')
        stream.write('{"step":')
    record = snapshot(tmp_path, 'run')
    assert len(record['history']) == MAX_ROWS
    assert record['history'][-1]['step'] == 159
    assert record['history'][-1]['gradient_policy']['weights']['extraction'] is None
    encoded = json.dumps(record, allow_nan=False)
    assert all(text not in encoded for text in ('private/path', 'private_subject', 'secret'))
    assert record['observed_at']


def check_only_server_registered_runs_are_readable(tmp_path):
    monitor = TrainingMonitor({'abide':{'path':'runs/abide', 'label':'ABIDE fold 1'}}, tmp_path)
    assert monitor.listing() == [{'id':'abide', 'label':'ABIDE fold 1'}]
    assert monitor.read('abide')['history'] == []
    for identifier in ('../abide', '/etc/passwd', 'unknown', 'abide%2f..'):
        with unittest.TestCase().assertRaises(ValueError):
            monitor.read(identifier)


def check_mirror_preserves_source_timestamp_and_relative_config(tmp_path):
    folder = tmp_path / 'mirror'
    folder.mkdir()
    packet = {'id':'remote', 'history':[], 'observed_at':'2026-11-12T00:00:00+00:00'}
    (folder / 'snapshot.json').write_text(json.dumps(packet))
    result = TrainingMonitor({'local':{'path':'mirror', 'label':'Remote snapshot'}}, tmp_path).read('local')
    assert result['observed_at'] == packet['observed_at']
    assert result['label'] == 'Remote snapshot'
    assert result['id'] == 'local'


class TrainingMonitorTests(unittest.TestCase):
    def test_training_probe_and_time_budget_are_distinct_from_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'report.json').write_text(json.dumps({
                'status': 'budget_exhausted', 'stop_reason': 'wall_time_budget',
                'training_probe': {'size': 64, 'indices': [1, 2]},
                'history': [{'epoch': 1, 'training_probe': {'auc': .8}, 'validation': {'auc': .5}}]}))
            record = snapshot(path, 'run')
        self.assertEqual(record['stop_reason'], 'wall_time_budget')
        self.assertEqual(record['training_probe_size'], 64)
        self.assertEqual(record['validation'][0]['training_probe']['auc'], .8)
        self.assertEqual(record['validation'][0]['validation']['auc'], .5)
        self.assertNotIn('indices', json.dumps(record))

    def test_core_profile_preserves_measurement_and_weight_semantics(self):
        from unibrain_agent.training_monitor import policy_record
        record = policy_record({'profile': 'paper_core', 'gradient_scope': 'all_trainable_parameters',
                                'gradient_normalization': 'l2', 'critic_action_space': 'raw_logits',
                                'weight_rule': 'softmax_div_gradient_scale', 'model_step_completed': True,
                                'gradient_parameter_elements': 300000000, 'gradient_scales': {'extraction': 5.},
                                'private': 'credential'})
        self.assertEqual(record['gradient_scope'], 'all_trainable_parameters')
        self.assertEqual(record['gradient_scales']['extraction'], 5.)
        self.assertTrue(record['model_step_completed'])
        self.assertNotIn('credential', json.dumps(record))

    def test_paper_profile_has_correct_atlas_and_segmentation_routes(self):
        from unibrain_agent.training_monitor import paper_layout
        result = paper_layout({'stages':[{'id':'parcellation','method':'Nearest atlas transfer'}]})
        self.assertIn('trilinear', result['stages'][0]['method'])
        self.assertIn(['input','segmentation'], result['forward_edges'])
        self.assertNotIn(['registration','segmentation'], result['forward_edges'])

    def test_bounded_private_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            check_records_are_bounded_and_exclude_subjects_and_paths(Path(folder))

    def test_registry_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            check_only_server_registered_runs_are_readable(Path(folder))

    def test_mirror_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            check_mirror_preserves_source_timestamp_and_relative_config(Path(folder))
