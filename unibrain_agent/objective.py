"""Describe the executed objective from tensors, coefficients and autograd edges."""

import math

LABELS = {'extraction': ('Brain extraction', 'BCE', 'Subject brain mask'),
          'registration': ('Image registration', 'GCC', 'Fixed MRI template'),
          'segmentation': ('Tissue segmentation', 'CE', 'Tissue reference'),
          'classification': ('Diagnosis classification', 'CE', 'Subject class label'),
          'parcellation': ('AAL parcellation', 'CE', 'Subject AAL reference'),
          'regularization': ('Deformation regularization', 'Smoothness', 'Deformation field')}


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
    return dict(version=1, status='declared', profile='Original full UniBrain' if original else 'Compact adapter preview',
                selection=selection, coordination=coordination, terms=terms, modules=modules, connections=[],
                connection_evidence='Configured objective preview. Actual connections and coefficients appear after an executed forward pass.',
                explanation='Optional preprocessing learns through downstream objectives unless an explicit loss is declared.')


def observe_objective(model, losses, weights, total, step):
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
        terms.append({'id': name, 'name': label, 'definition': definition, 'supervision': supervision, 'weight': weight,
                      'value': value, 'weighted_value': weight * value,
                      'operations': sorted(operations), 'requires_grad': bool(loss.requires_grad)})
        connections.extend({'loss': name, 'module': module, 'status': 'autograd_path' if weight else 'zero_weight'} for module in sorted(reached))
    actual = float(total.detach())
    if not math.isfinite(actual) or not math.isclose(actual, reconstructed, rel_tol=2e-5, abs_tol=2e-5):
        raise ValueError('Objective manifest does not reconstruct the executed total')
    return {'version': 1, 'status': 'observed', 'step': step, 'terms': terms,
            'profile': 'Executed full UniBrain',
            'modules': [{'id': module, 'name': type(getattr(model, module, model)).__name__ + ' (' + module + ')'} for module in sorted(modules)],
            'connections': connections, 'total': actual,
            'reconstructed_total': reconstructed,
            'connection_evidence': 'Autograd paths; nonzero gradients are reported separately.'}
