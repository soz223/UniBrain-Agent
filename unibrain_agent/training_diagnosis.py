"""Evidence-bound advisory review and small diagnostic qualification cases."""

from .paths import PROJECT

ISSUES = ('target_disagreement', 'invalid_target_mass', 'metric_disagreement',
          'geometry_warning', 'missing_segmentation_gradient', 'insufficient_evidence')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'summary': {'type': 'string'}, 'issues': {'type': 'array', 'items': {'type': 'string', 'enum': list(ISSUES)}},
    'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    'next_action': {'type': 'string', 'enum': ['inspect_supervision', 'inspect_geometry', 'inspect_metrics', 'inspect_gradients', 'request_measurement']},
    'uncertainty': {'type': 'string'}}, 'required': ['summary', 'issues', 'evidence_ids', 'next_action', 'uncertainty']}


def diagnostic_context(report):
    fields = ('segmentation_loss', 'subject_reference_ce', 'subject_foreground_ce',
              'prediction_to_reference', 'prediction_to_training_target', 'training_target_to_reference',
              'identity_template_to_reference', 'metric_absolute_error', 'target_channel_sum_min',
              'target_channel_sum_max', 'target_requires_grad', 'inverse_grid_outside_fraction',
              'affine_determinant', 'segmentation_gradient', 'detached_target_gradient',
              'physical_geometry', 'target_foreground_fraction', 'target_foreground_ce', 'target_background_ce',
              'axis_flip_dice_diagnostic_only')
    return {'scope': 'Read-only checkpoint diagnosis; no automatic training changes',
            'dataset': report.get('dataset'), 'fold': report.get('fold'),
            'checkpoint_sha256': report.get('checkpoint_sha256'),
            'operator_evidence': report.get('operator_evidence', {}),
            'physical_geometry': report.get('physical_geometry', 'Not verified'),
            'metric_contract': {'classes': [1, 2, 3], 'average': 'unweighted foreground class mean',
                                'smoothing': 1e-5, 'both_empty': 'Dice equals one and remains included in the mean'},
            'training_mode_probe': report.get('training_mode_probe'),
            'evidence': [{'id': f'probe_{index}', **{key: case[key] for key in fields if key in case}}
                         for index, case in enumerate(report.get('cases', []))],
            'saved_prediction_check': report.get('saved_prediction_check', {})}


def screening_issues(context):
    issues = set()
    rows = context.get('evidence', [])
    if not rows:
        return ['insufficient_evidence']
    for row in rows:
        target = row.get('prediction_to_training_target', {}).get('foreground_macro_dice')
        subject = row.get('prediction_to_reference', {}).get('foreground_macro_dice')
        if target is not None and subject is not None and target > .8 and subject < .4:
            issues.add('target_disagreement')
        if row.get('training_target_to_reference', {}).get('foreground_macro_dice', 1) < .4:
            issues.add('target_disagreement')
        if row.get('target_channel_sum_min', 1) < .999 or row.get('target_channel_sum_max', 1) > 1.001:
            issues.add('invalid_target_mass')
        if row.get('metric_absolute_error', 0) > 1e-5:
            issues.add('metric_disagreement')
        if row.get('inverse_grid_outside_fraction', 0) > .5:
            issues.add('geometry_warning')
        gradient = row.get('segmentation_gradient', {}).get('seg_net')
        if gradient is not None and (not gradient.get('finite') or not gradient.get('nonzero_tensors')):
            issues.add('missing_segmentation_gradient')
    return sorted(issues)


def review(agent, context):
    expected = screening_issues(context)
    result = agent.decide('training_diagnosis', context, SCHEMA,
                          {'summary': 'Deterministic screening only; no language model used.', 'issues': expected,
                           'evidence_ids': [row['id'] for row in context.get('evidence', [])],
                           'next_action': 'request_measurement', 'uncertainty': 'Screening flags are not causal proof.'},
                          instructions=(PROJECT / 'prompts/training_diagnosis.md').read_text(encoding='utf-8'))
    allowed = {row['id'] for row in context.get('evidence', [])}
    allowed.update(context.get('operator_evidence', {}))
    allowed.update(key for key in ('training_mode_probe', 'metric_contract', 'saved_prediction_check') if context.get(key))
    if not set(result['evidence_ids']) <= allowed:
        raise ValueError('Diagnosis cited unavailable evidence')
    result['screening_issues'] = expected
    result['screening_agreement'] = set(expected) <= set(result['issues'])
    result['automatic_changes_applied'] = False
    return result


def qualification_cases():
    cases = [({}, ['insufficient_evidence']),
             ({'prediction_to_training_target': {'foreground_macro_dice': .94},
               'prediction_to_reference': {'foreground_macro_dice': .22}}, ['target_disagreement']),
             ({'target_channel_sum_min': 0., 'target_channel_sum_max': 1.}, ['invalid_target_mass']),
             ({'metric_absolute_error': .2}, ['metric_disagreement']),
             ({'inverse_grid_outside_fraction': .8}, ['geometry_warning']),
             ({'segmentation_gradient': {'seg_net': {'finite': True, 'nonzero_tensors': 0}}}, ['missing_segmentation_gradient'])]
    return [({'scope': 'Synthetic qualification fixture; not a real experiment',
              'evidence': [{'id': 'fixture', **row}] if row else []}, expected) for row, expected in cases]


def qualify(agent):
    cases = []
    for context, expected in qualification_cases():
        result = review(agent, context)
        cases.append({'expected': expected, 'result': result, 'passed': set(result['issues']) == set(expected)})
    return {'passed': all(case['passed'] for case in cases), 'cases': cases,
            'scope': 'Six synthetic diagnostic cases; not evidence of general autonomous troubleshooting ability',
            'backend': agent.backend}
