"""Task-wise actors with a UniBrain reference update and matched-sample reward."""

import copy

import torch

from .taskforce import TASKS
from .taskforce_paper import PaperTaskForce


def upgrade_anchored_identity(previous, current):
    from .training_stability import PREVIOUS_TRAINERS, PREVIOUS_POLICY, PREVIOUS_CONTROL

    known = {(trainer, PREVIOUS_POLICY) for trainer in PREVIOUS_TRAINERS}
    known.add(('e35c39b0a4f6cd688f3e0b95bb91fad2d34db53c11c7d41d55a955dc5500ad82',
               '19cb651a1a78b1f5bc43f5923a22a4747fe9e9703d29cee91d3c9fe9bd4b4671'))
    migrated = copy.deepcopy(previous)
    hashes = migrated['implementation_hashes']
    baseline = previous.get('coordination') == 'none' and hashes.get('replacement_training.py') in PREVIOUS_TRAINERS
    if (not baseline and (hashes.get('replacement_training.py'), hashes.get('taskforce_paper.py')) not in known
            or current.get('taskforce_gradient_normalization') != 'anchored'):
        raise ValueError('Unsupported anchored coordinator migration')
    if baseline:
        migrated.update(coordination='taskforce', taskforce_profile='paper', taskforce_action_bound=1.)
    migrated['taskforce_gradient_normalization'] = 'anchored'
    for name in ('replacement_training.py', 'taskforce_paper.py', 'training_stability.py', 'anchored_taskforce.py'):
        hashes[name] = current['implementation_hashes'][name]
    if hashes.get('training_control.py') == PREVIOUS_CONTROL:
        hashes['training_control.py'] = current['implementation_hashes']['training_control.py']
    if migrated != current:
        raise ValueError('Coordinator migration cannot change model, data or unrelated configuration')


def matched_forward(model, before_rng, forward):
    from .replacement_training import rng_state, restore_rng

    current_rng = rng_state()
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        restore_rng(before_rng)
        with torch.no_grad():
            return forward()
    finally:
        restore_rng(current_rng)
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)


def shield_coefficients(base, proposal, gram, radius):
    base = torch.as_tensor(base, dtype=torch.float64)
    delta = proposal.double() - base
    reference = gram @ base
    change = gram @ delta
    floor = torch.where(reference > 0, reference * .5, reference)
    limits = torch.where(change < 0, (reference - floor) / (-change).clamp_min(1e-30), torch.ones_like(change))
    fraction = min(1., max(0., float(limits.min())))
    deviation = float((delta @ gram @ delta).clamp_min(0).sqrt())
    norm = float((base @ gram @ base).clamp_min(0).sqrt())
    if deviation:
        fraction = min(fraction, radius * norm / deviation)
    return base + fraction * delta, fraction


class AnchoredTaskForce(PaperTaskForce):
    def __init__(self, base_weights, seed=17, batch_size=64, action_bound=1., radius=.1):
        super().__init__(seed, batch_size, 'bounded_l2', action_bound)
        if not 0 < radius <= .1:
            raise ValueError('Invalid coordination trust radius')
        self.normalization = 'anchored'
        self.base = tuple(base_weights)
        if len(self.base) != len(TASKS) or any(value <= 0 for value in self.base):
            raise ValueError('Positive original task coefficients are required')
        self.radius = radius
        self.matched_transition = None

    def observe(self, losses, parameters):
        self.normalization = 'l2'
        try:
            return super().observe(losses, parameters)
        finally:
            self.normalization = 'anchored'

    def combine(self, losses, parameters):
        transition = self.matched_transition
        self.previous = None
        _, record = super().combine(losses, parameters)
        observation, logits, _, values = self.previous
        if transition is not None:
            self.replay.append((*transition, observation))
        self.matched_transition = None
        self.before_update = (observation, logits, values)
        self.previous = None
        base = torch.tensor(self.base, dtype=torch.float64)
        residual = (self.count * logits.softmax(0).double() - 1).clamp(-1, 1)
        proposal = base * (1 + self.radius * residual)
        coefficients, fraction = shield_coefficients(base, proposal, self.raw_gram, self.radius)
        weights = dict(zip(TASKS, coefficients.tolist()))
        record.update(profile='unibrain_anchored', weight_rule='original_weights_with_guarded_residual',
                      base_weights=dict(zip(TASKS, self.base)), weights=weights,
                      trust_radius=self.radius, accepted_action_fraction=fraction,
                      reference_alignment=(self.raw_gram @ base).tolist(),
                      applied_alignment=(self.raw_gram @ coefficients).tolist(),
                      reward_scope='same_training_subject_before_after_update')
        return sum(weights[task] * losses[task] for task in TASKS), record

    def after_model_step(self, post_losses=None):
        if post_losses is None:
            raise ValueError('Matched post-update training losses are required')
        observation, logits, before = self.before_update
        after = torch.tensor([float(post_losses[task].detach()) for task in TASKS])
        if not torch.isfinite(after).all() or (after < 0).any():
            raise ValueError('Stability guard: invalid matched reward loss')
        base = torch.tensor(self.base)
        improvement = float(((before.log1p() - after.log1p()) * base).sum() / base.sum())
        reward = max(-1., min(1., improvement))
        self.matched_transition = (observation, logits, reward)
        self.last_record.update(reward=reward, reward_terms={'same_subject_improvement': improvement,
                                'validation_used': False, 'stochastic_state_replayed': True},
                                post_update_losses=dict(zip(TASKS, after.tolist())))
        return super().after_model_step()

    def state_dict(self):
        state = super().state_dict()
        state.update(trust_radius=self.radius, matched_transition=copy.deepcopy(self.matched_transition))
        return state

    def load_state_dict(self, state):
        if not 0 < state.get('trust_radius', 0) <= .1:
            raise ValueError('Invalid saved coordination radius')
        super().load_state_dict(state)
        self.radius = state['trust_radius']
        self.matched_transition = copy.deepcopy(state.get('matched_transition'))
