"""Full architecture library extensions with explicit host connections."""

import torch
from torch import nn

from .full_model import build_full_model
from .full_connectivity import connect_model
from .data import digest

REPOSITORIES = {'redwgan': 'denoise_redwgan', 'deepn4': 'DeepN4', 'deepnormalize': 'deepNormalize',
                'harmonizing_flows': 'HarmonizingFlows', 'ernet': 'reg_ernet', 'fastsurfer': 'FastSurfer',
                'nobrainer': 'nobrainer', 'fbnetgen': 'FBNETGEN', 'brainnetcnn': 'FBNETGEN', 'bnt': 'BrainNetworkTransformer'}
TARGETS = {'redwgan': 'preprocessor', 'deepn4': 'preprocessor', 'deepnormalize': 'preprocessor',
           'harmonizing_flows': 'preprocessor', 'ernet': 'ext_net', 'fastsurfer': 'par_net',
           'nobrainer': 'par_net', 'fbnetgen': 'graph_gen', 'brainnetcnn': 'gnn_pred', 'bnt': 'gnn_pred'}


def build_extended_model(source, method, host_checkpoint=None, weights=None):
    if method not in TARGETS:
        raise ValueError('Unknown extended candidate')
    host, config, architecture = build_full_model(source, host_checkpoint=host_checkpoint)
    if TARGETS[method] == 'preprocessor':
        from .full_preprocessing import build_preprocessor
        component = build_preprocessor(method)
        modules = {'preprocessor': component}
        metadata = component.architecture
    elif method in ('ernet', 'fastsurfer', 'nobrainer'):
        from .full_anatomy_candidates import build_anatomy_modules
        modules, metadata = build_anatomy_modules(method)
    else:
        from .full_graph_candidates import build_graph_candidate
        component, metadata = build_graph_candidate(method)
        modules = {TARGETS[method]: component}
    if weights:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        state = state.get('model_state_dict', state)
        if len(modules) == 1:
            next(iter(modules.values())).load_state_dict(state, strict=True)
        else:
            nn.ModuleDict(modules).load_state_dict(state, strict=True)
    for name, module in modules.items():
        setattr(host, name, module)
    host.candidate_modules = tuple(modules)
    host.learned_parcellation = method in ('fastsurfer', 'nobrainer')
    model = connect_model(host)
    architecture.update(method=method, parameters=sum(p.numel() for p in model.parameters()), compact=False,
                        input_shape=[96, 96, 96], candidate=metadata, candidate_modules=list(modules),
                        weights_sha256=digest(weights) if weights else None,
                        connection='Soft atlas transfer and positive continuous graph weights',
                        spatial_augmentation='Disabled for learned parcellation to keep subject AAL supervision aligned; original image/brain augmentation otherwise',
                        parcellation_connection='Equal mixture of predicted and transferred atlas probabilities; explicit supervised AAL loss' if host.learned_parcellation else None)
    for name in modules:
        architecture['unchanged_modules'].pop(name, None)
    return model, config, architecture
