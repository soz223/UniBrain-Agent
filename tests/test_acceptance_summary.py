import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.summarize_acceptance import canonical_hash, merge_summaries, summarize, summarize_cycle, table
from tests import test_acceptance_matrix as fixtures


class AcceptanceSummaryTests(unittest.TestCase):
    def fixture(self, strict=True):
        fixture = fixtures.AcceptanceExecutionTests("test_budget_exhausted_is_valid_only_with_complete_bounded_evidence")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for job in fixture.jobs:
            job.update(verify_gradients=True)
            job['data_audit'] = {'scope': 'acceptance_subset_not_full_cohort', 'data_hashes': {'adhd_train_sample.npy': 'a' * 64}}
            if strict:
                job['gradient_protocol_version'] = 2
        def runner(command, log, timeout):
            result = fixture.gradient_runner(command, log, timeout)
            if '--preflight' in command:
                return result
            job = next(job for job in fixture.jobs if job['dataset'] == command[-1])
            folder = Path(job['output'])
            report = json.loads((folder / 'report.json').read_text())
            report.update(architecture={'parameters': 10}, train_seconds=.1)
            report['audit']['splits'] = {'train': 2, 'val': 2, 'test': 2}
            report['config'].update(coordination='none', checkpoint_mode='weights')
            report['checkpoint_policy'] = {'mode': 'weights', 'resume_allowed': False}
            report['gradient_verification']['protocol_version'] = 2 if strict else 1
            (folder / 'report.json').write_text(json.dumps(report), encoding='utf-8')
            events = [json.loads(line) for line in (folder / 'updates.jsonl').read_text().splitlines()]
            for event in events:
                verification = event['gradient_verification']
                verification['classification_to_candidates']['reg_net']['nonzero'] = True
                if strict:
                    verification.update(protocol_version=2, classification_to_input={'required': True, 'finite': True, 'nonzero': True})
            (folder / 'updates.jsonl').write_text('\n'.join(map(json.dumps, events)), encoding='utf-8')
            return result
        fixtures.matrix.execute(fixture.jobs, fixture.output, {'scope': 'CPU protocol fixture',
                                                             'optimizer_update_budget_per_job': 2}, 2, runner=runner)
        return fixture.output / 'matrix.json'

    def test_actual_revalidation_exports_strict_and_legacy_scopes(self):
        for strict in (True, False):
            report = summarize(self.fixture(strict))
            self.assertEqual(report['passed'], 2)
            self.assertEqual(report['strict_passed'], 2 if strict else 0)
            self.assertEqual(report['legacy_passed'], 0 if strict else 2)
            self.assertTrue(all(row['strict_current_gradient_verified'] is strict for row in report['jobs']))
            self.assertIn('Gradient contract verified', table(report))
            self.assertIn('protocol_2' if strict else 'legacy_boundary', table(report))

    def test_duplicate_jobs_in_one_matrix_are_rejected(self):
        path = self.fixture()
        record = json.loads(path.read_text())
        record['jobs'].append(record['jobs'][0])
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate job'):
            summarize(path)

    def test_protocol_three_warmup_is_preserved_in_summary_and_coverage(self):
        fixture = fixtures.ProtocolThreeAcceptanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        job, path, report, events = fixture.fixture()
        fixture.write(path, report, events)
        measured = fixtures.matrix.validate_result(job, path, 2)
        job.update(measured)
        matrix_path = path.parent.parent / 'matrix.json'
        matrix_path.write_text(json.dumps({'scope': 'CPU protocol fixture', 'optimizer_update_budget_per_job': 2, 'jobs': [job]}), encoding='utf-8')
        result = summarize(matrix_path)
        self.assertEqual(result['warmup_passed'], 1)
        self.assertEqual(result['jobs'][0]['gradient_protocol_version'], 3)
        self.assertFalse(result['jobs'][0]['initial_nonzero_gradient_verified'])
        merged = merge_summaries([result, copy.deepcopy(result)])
        self.assertEqual(merged['warmup_execution_count'], 1)
        self.assertEqual(merged['gradient_protocol_execution_counts'], {'3': 1})
        self.assertEqual(merged['coverage'][0]['verified_gradient_protocols'], [3])
        self.assertEqual(merged['coverage'][0]['warmup_executions'], 1)
        self.assertIn('protocol_3', table(result))

    def test_copied_reports_do_not_count_as_additional_executions(self):
        report = summarize(self.fixture())
        result = merge_summaries([report, copy.deepcopy(report)])
        self.assertEqual(result['execution_count'], 2)
        self.assertEqual(result['coverage_total'], 2)
        self.assertEqual(len(result['jobs']), 4)
        changed = copy.deepcopy(report)
        changed['jobs'][0]['data_fingerprint'] = 'different data'
        with self.assertRaisesRegex(ValueError, 'different data exports'):
            merge_summaries([report, changed])

    def test_six_previous_and_twenty_six_new_jobs_cover_twenty_eight_cells(self):
        methods = ['nnunet', 'fbnetgen', 'brainnetcnn', 'bnt', 'redwgan', 'deepn4', 'deepnormalize',
                   'harmonizing_flows', 'fastsurfer', 'nobrainer', 'ernet', 'synthstrip', 'voxelmorph', 'lapirn']
        old, new = [], []
        for dataset in ('abide', 'adhd'):
            for position, method in enumerate(methods):
                row = {'dataset': dataset, 'method': method, 'fold': 1, 'seed': 17, 'setting': 'joint',
                       'coordination_mode': 'none', 'data_scope': 'acceptance_subset_not_full_cohort',
                       'data_fingerprint': canonical_hash({'dataset': dataset}), 'status': 'not_run_after_transition',
                       'strict_current_gradient_verified': False}
                prior = {**row}
                if dataset == 'abide' and position < 6:
                    prior.update(status='passed', report_sha256='old' + method, updates_sha256='old' + method,
                                 checkpoint_sha256='old' + method, strict_current_gradient_verified=position >= 4)
                old.append(prior)
                if dataset != 'abide' or method not in ('redwgan', 'deepn4'):
                    new.append({**row, 'status': 'passed', 'report_sha256': dataset + method,
                                'updates_sha256': dataset + method, 'checkpoint_sha256': dataset + method,
                                'strict_current_gradient_verified': True})
        cycles = [{'cycle_sha256': dataset, 'training_report_sha256': 'cycle' + dataset} for dataset in ('abide', 'adhd')]
        result = merge_summaries([{'matrix_sha256': 'old', 'jobs': old}, {'matrix_sha256': 'new', 'jobs': new}], cycles)
        self.assertEqual(result['execution_count'], 32)
        self.assertEqual(result['legacy_execution_count'], 4)
        self.assertEqual(result['coverage_total'], 28)
        self.assertEqual(result['strict_coverage_passed'], 28)
        self.assertEqual(result['cycle_count'], 2)
        self.assertEqual(result['unique_training_executions_including_cycles'], 34)
        self.assertEqual(len(result['jobs']), 54)

    def test_cycle_is_separate_and_revalidates_its_training_and_adapter(self):
        path = self.fixture()
        folder = path.parent / 'abide_voxelmorph'
        cycle_root = path.parent / 'cycle'
        cycle_root.mkdir()
        training_root = cycle_root / 'training'
        training_root.mkdir()
        for name in ('report.json', 'updates.jsonl', 'last.pt'):
            (training_root / name).write_bytes((folder / name).read_bytes())
        adapter = cycle_root / 'accepted_adapter.py'
        adapter.write_text('# CPU protocol fixture', encoding='utf-8')
        sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
        training = json.loads((training_root / 'report.json').read_text())
        training['architecture'].update(generated_adapter_active_in_training=True, generated_adapter_sha256=sha)
        (training_root / 'report.json').write_text(json.dumps(training), encoding='utf-8')
        evaluation = {'status': 'passed', 'runs': True, 'optimizer_updates': 1,
                      'checks': {key: True for key in ('contract', 'gradient', 'spatial', 'fidelity', 'updates')},
                      'data': {'sha256': 'a' * 64}}
        cycle = {'status': 'complete', 'method': 'voxelmorph', 'seed': 17, 'training': training,
                 'accepted_adapter': {'file': adapter.name, 'sha256': sha}, 'accepted_adapter_unchanged_after_training': True,
                 'actual_updates': {'acceptance': 1, 'training': 2}, 'budgets': {'acceptance_updates': 1},
                 'integration': {'status': 'passed', 'final_valid': True, 'optimizer_updates': 1,
                                 'calls': [{'adapter_sha256': sha, 'evaluation': evaluation}]}}
        cycle_path = cycle_root / 'report.json'
        cycle_path.write_text(json.dumps(cycle), encoding='utf-8')
        record = summarize_cycle(cycle_path)
        self.assertEqual(record['training_updates'], 2)
        self.assertTrue(record['strict_current_gradient_verified'])
        merged = merge_summaries([summarize(path)], [record, record])
        self.assertEqual(merged['cycle_count'], 1)
        self.assertEqual(merged['coverage_total'], 2)
        adapter.write_text('# altered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unchanged adapter'):
            summarize_cycle(cycle_path)


if __name__ == '__main__':
    unittest.main()
