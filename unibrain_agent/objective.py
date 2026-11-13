"""Describe the executed objective from tensors, coefficients and autograd edges."""

import math

LABELS = {'extraction': ('Brain extraction', 'BCE', 'Subject brain mask'),
          'registration': ('Image registration', 'GCC', 'Fixed MRI template'),
          'segmentation': ('Tissue segmentation', 'CE', 'Tissue reference'),
          'classification': ('Diagnosis classification', 'CE', 'Subject class label'),
          'parcellation': ('AAL parcellation', 'CE', 'Subject AAL reference'),
          'regularization': ('Deformation regularization', 'Smoothness', 'Deformation field')}


STAGE_ORDER = ('denoising', 'bias', 'extraction', 'normalization', 'registration',
               'parcellation', 'graph', 'segmentation', 'classification', 'surface')
METHOD_STAGES = {
    'synthstrip': {'extraction': 'M02'}, 'voxelmorph': {'registration': 'M06'},
    'lapirn': {'registration': 'M07'}, 'nnunet': {'segmentation': 'M08'},
    'redwgan': {'denoising': 'M14'}, 'deepn4': {'bias': 'M01'},
    'ernet': {'extraction': 'M03', 'registration': 'M03'},
    'deepnormalize': {'normalization': 'M04'}, 'harmonizing_flows': {'normalization': 'M05'},
    'fastsurfer': {'parcellation': 'M09'}, 'nobrainer': {'parcellation': 'M10'},
    'fbnetgen': {'graph': 'M11'}, 'brainnetcnn': {'classification': 'M12'},
    'bnt': {'classification': 'M13'},
}


def with_stage_layout(packet, selection=None, method=None):
    """Associate objective terms with their pipeline stages, including lossless stages."""
    from .catalog import DEFAULTS, catalog
    selection = selection or packet.get('selection')
    if selection is None:
        selection = {**DEFAULTS, **METHOD_STAGES.get(method or packet.get('method'), {})}
    stages = {stage['id']: stage for stage in catalog()['stages']}
    layout = []
    for identifier in STAGE_ORDER:
        stage = stages[identifier]
        option = next(option for option in stage['options'] if option['id'] == selection[identifier])
        if option['id'] == 'none':
            continue
        losses = [term['id'] for term in packet['terms']
                  if ('registration' if term['id'] == 'regularization' else term['id']) == identifier]
        layout.append(dict(id=identifier, name=stage['name'], method=option.get('method') or option['label'],
                           loss_ids=losses, branch=identifier in {'segmentation', 'surface'},
                           note='' if losses else 'N/A: no independent loss. See measured downstream connections below.'))
    assigned = {identifier for stage in layout for identifier in stage['loss_ids']}
    result = {**packet, 'selection': dict(selection), 'stages': layout,
              'unassigned_loss_ids': [term['id'] for term in packet['terms'] if term['id'] not in assigned]}
    if packet.get('paper_original'):
        from .training_monitor import paper_layout
        result = paper_layout(result)
    return result


def preview_objective(selection, coordination='single', build=None):
    from .catalog import DEFAULTS, catalog, validate_selection
    selection = validate_selection(selection)
    original = selection == DEFAULTS
    terms = []
    for name, weight in [('extraction', 1.), ('registration', .1), ('segmentation', 1.), ('classification', 1.)]:
        label, definition, target = LABELS[name]
        if name == 'segmentation':
            target = 'Warped template tissue labels' if original else 'Subject tissue reference'
        terms.append(dict(id=name, name=label, definition=definition, supervision=target, weight=weight))
    if not original:
        if build and build.get('selection') == selection and build.get('coordination', 'single') == coordination:
            if build.get('plan', {}).get('extraction_loss') == 'bce_dice':
                terms[0]['definition'] = 'BCE + soft Dice loss'
        if selection['registration'] in {'M06', 'M07'}:
            terms.append(dict(id='regularization', name=LABELS['regularization'][0], definition='Deformation smoothness', supervision='Deformation field', weight=.001, weight_group='registration', relative_weight=.01))
        if selection['parcellation'] in {'M09', 'M10'}:
            terms.append(dict(id='parcellation', name=LABELS['parcellation'][0], definition='CE', supervision='Subject AAL reference', weight=.2, weight_group='segmentation', relative_weight=.2))
        if selection['segmentation'] == 'M08':
            terms[2]['definition'] = 'CE + 0.5 CE per auxiliary head'
    modules = []
    for stage in catalog()['stages']:
        option = next(option for option in stage['options'] if option['id'] == selection[stage['id']])
        if option['id'] != 'none':
            modules.append({'id': stage['id'], 'name': option.get('method') or option['label'].split(' (')[0]})
    return with_stage_layout(dict(version=1, status='declared', profile='Original full UniBrain' if original else 'Compact adapter preview',
                selection=selection, coordination=coordination, terms=terms, modules=modules, connections=[],
                connection_evidence='Configured objective preview. Actual connections and coefficients appear after an executed forward pass.',
                explanation='Optional preprocessing learns through downstream objectives unless an explicit loss is declared.'))


def observe_objective(model, losses, weights, total, step):
    definitions = getattr(losses, 'objective_definitions', {})
    if hasattr(losses, 'objective_parts'):
        parts = losses.objective_parts
        weights = {name: weights[group] * scale for name, (_, group, scale) in parts.items()}
        losses = {name: value for name, (value, _, _) in parts.items()}
    parameters = {id(parameter): name for name, parameter in model.named_parameters()}
    modules = {name.split('.')[0] for name in parameters.values()}
    terms, connections = [], []
    reconstructed = 0.
    for name, loss in losses.items():
        weight = float(weights[name])
        value = float(loss.detach())
        reconstructed += weight * value
        pending = [loss.grad_fn] if loss.grad_fn else []
        visited, reached, operations = set(), set(), set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            operations.add(type(node).__name__)
            variable = getattr(node, 'variable', None)
            parameter = parameters.get(id(variable))
            if parameter:
                reached.add(parameter.split('.')[0])
            pending.extend(child for child, _ in node.next_functions if child is not None)
        label, definition, supervision = LABELS.get(name, (name.replace('_', ' ').title(), 'Runtime tensor', 'See generated code'))
        definition = definitions.get(name, definition)
        terms.append({'id': name, 'name': label, 'definition': definition, 'supervision': supervision, 'weight': weight,
                      'value': value, 'weighted_value': weight * value,
                      'operations': sorted(operations), 'requires_grad': bool(loss.requires_grad)})
        connections.extend({'loss': name, 'module': module, 'status': 'autograd_path' if weight else 'zero_weight'} for module in sorted(reached))
    actual = float(total.detach())
    if not math.isfinite(actual) or not math.isclose(actual, reconstructed, rel_tol=2e-5, abs_tol=2e-5):
        raise ValueError('Objective manifest does not reconstruct the executed total')
    packet = {'version': 1, 'status': 'observed', 'step': step, 'terms': terms,
            'paper_original': getattr(model, 'paper_original', False),
            'profile': 'Executed full UniBrain',
            'modules': [{'id': module, 'name': type(getattr(model, module, model)).__name__ + ' (' + module + ')'} for module in sorted(modules)],
            'connections': connections, 'total': actual,
            'reconstructed_total': reconstructed,
            'connection_evidence': 'Autograd paths; nonzero gradients are reported separately.'}

    return with_stage_layout(packet, getattr(getattr(model, 'config', None), 'stages', None))
