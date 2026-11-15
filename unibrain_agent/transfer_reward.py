"""Training-only cross-subject feedback for the anchored coordinator."""

import copy

import numpy as np


PREVIOUS = {
    'replacement_training.py': '535578b0e538a58e306d584b555dcc439c9f8035ffd4e56f9bbd12d07bf8ff48',
    'anchored_taskforce.py': 'caf7e3d72578c1cb673604198bab1d41b95a56f026648d32ef37cbdfa58c2ee8',
    'training_stability.py': 'dbbd5f0a0a8a6794c787d0e7cabda0216580cfeeb02c277c43706a4e1598d5c5',
}


def upgrade_reward_identity(previous, current):
    if (previous.get('taskforce_gradient_normalization') != 'anchored'
            or previous.get('coordination_reward', 'same_subject') != 'same_subject'
            or current.get('coordination_reward') != 'cross_subject'
            or any(previous['implementation_hashes'].get(k) != v for k, v in PREVIOUS.items())):
        raise ValueError('Unsupported cross-subject reward migration')
    migrated = copy.deepcopy(previous)
    migrated['coordination_reward'] = 'cross_subject'
    for name in (*PREVIOUS, 'transfer_reward.py'):
        migrated['implementation_hashes'][name] = current['implementation_hashes'][name]
    if migrated != current:
        raise ValueError('Reward migration cannot change data, architecture or optimizer settings')


def reference_index(labels, current, step, seed):
    labels = np.asarray(labels)
    candidates = np.flatnonzero((labels == step % 2) & (np.arange(len(labels)) != current))
    if not len(candidates):
        raise ValueError('Cross-subject reward requires another training subject in each class')
    return int(np.random.default_rng(np.random.SeedSequence([seed, step])).choice(candidates))


def reference_losses(model, sample, templates):
    from model import GCC
    from .anchored_taskforce import matched_forward
    from .replacement_training import original_losses, rng_state

    image, brain, _, _, label = sample
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        def forward():
            output = model(templates[0], image, templates[1], templates[2], if_train=False)
            losses = original_losses(output, brain, label)
            losses['registration'] = GCC()(output[2][-1], templates[0])
            return losses
        return matched_forward(model, rng_state(), forward)
    finally:
        for module, mode in modes:
            module.training = mode


def step_with_reference(model, optimizer, sample, templates):
    before = reference_losses(model, sample, templates)
    optimizer.step()
    return before, reference_losses(model, sample, templates)
