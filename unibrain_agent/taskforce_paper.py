"""TaskForce core equations and Appendix B agent configuration."""

import copy
import math
from collections import deque

import torch
from torch import nn

from .taskforce import GradientTeam, TASKS


def full_gradient_gram(losses, parameters):
    parameters = tuple(parameters)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError('Full gradient observation requires trainable parameters')
    gradients = [torch.autograd.grad(losses[task], parameters, retain_graph=True, allow_unused=True)
                 if losses[task].requires_grad else (None,) * len(parameters) for task in TASKS]
    gram = torch.zeros(len(TASKS), len(TASKS), dtype=torch.float64, device=parameters[0].device)
    for pieces in zip(*gradients):
        for i, left in enumerate(pieces):
            if left is None:
                continue
            for j in range(i, len(TASKS)):
                right = pieces[j]
                if right is not None:
                    gram[i, j] += torch.dot(left.detach().reshape(-1).double(), right.detach().reshape(-1).double())
    gram = gram + gram.T - torch.diag(gram.diag())
    if not torch.isfinite(gram).all():
        raise ValueError('Nonfinite full gradient Gram matrix')
    return gram.cpu()


def paper_network(inputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


class BoundedActor(nn.Sequential):
    def __init__(self, inputs, bound):
        super().__init__(*paper_network(inputs))
        self.bound = bound

    def forward(self, inputs):
        return self.bound * torch.tanh(super().forward(inputs))


class PaperTaskForce(GradientTeam):
    def __init__(self, seed=17, batch_size=64, normalization='l2', action_bound=None):
        if normalization not in ('l2', 'none') or not 1 <= batch_size <= 100000:
            raise ValueError('Invalid TaskForce normalization or replay batch')
        if action_bound is not None and (not math.isfinite(action_bound) or action_bound <= 0):
            raise ValueError('Action bound must be finite and positive')
        super().__init__((1.,) * len(TASKS), seed, batch_size)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actors = [BoundedActor(self.count + 1, action_bound) if action_bound is not None
                           else paper_network(self.count + 1) for _ in TASKS]
            self.critics = [paper_network(self.count * (self.count + 2)) for _ in TASKS]
        for actor in self.actors:
            nn.init.zeros_(actor[-1].weight)
            nn.init.zeros_(actor[-1].bias)
        self.target_actors = copy.deepcopy(self.actors)
        self.target_critics = copy.deepcopy(self.critics)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=5e-4) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=5e-4) for critic in self.critics]
        self.replay = deque(maxlen=100000)
        self.normalization = normalization
        self.action_bound = action_bound
        self.noise_state = torch.zeros(self.count)
        self.action_steps = 0
        self.model_steps = 0
        self.pending_model_step = False
        self.raw_gram = None
        self.scales = None

    def observe(self, losses, parameters):
        self.raw_gram = full_gradient_gram(losses, parameters)
        norms = self.raw_gram.diag().clamp_min(0).sqrt()
        self.scales = torch.where(norms > 1e-12, norms, torch.ones_like(norms)) if self.normalization == 'l2' else torch.ones_like(norms)
        gram = self.raw_gram / (self.scales[:, None] * self.scales[None, :])
        values = torch.tensor([float(losses[task].detach()) for task in TASKS])
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('TaskForce requires finite nonnegative task losses')
        observation = torch.cat((gram.float(), torch.log1p(values)[:, None]), dim=1)
        if not torch.isfinite(observation).all():
            raise ValueError('Nonfinite TaskForce observation')
        return observation, gram, values

    def action(self, observation, explore=True):
        with torch.no_grad():
            logits = torch.cat([actor(observation[i]) for i, actor in enumerate(self.actors)])
            scale = .3 + (.05 - .3) * min(self.action_steps / 10000, 1.)
            if explore:
                self.noise_state += -.15 * self.noise_state + .2 * torch.randn(self.count, generator=self.generator)
                logits += scale * self.noise_state
                self.action_steps += 1
            if self.action_bound is not None:
                logits.clamp_(-self.action_bound, self.action_bound)
            self.action_logits = dict(zip(TASKS, logits.tolist()))
            self.noise_scale = scale
            return logits.detach()

    def combine(self, losses, parameters):
        if self.pending_model_step:
            raise RuntimeError('Complete the model optimizer step before requesting another TaskForce action')
        parameters = tuple(parameters)
        observation, gram, values = self.observe(losses, parameters)
        logits = self.action(observation)
        shares = logits.softmax(0)
        reward = None
        self.reward_terms = None
        if self.previous is not None:
            previous_observation, previous_logits, previous_gram, previous_values = self.previous
            previous_shares = previous_logits.softmax(0).double()
            improvement = (torch.log1p(previous_values) - torch.log1p(values)).sum()
            penalty = previous_shares @ previous_gram @ previous_shares
            reward = float(improvement - .001 * penalty)
            if not torch.isfinite(torch.tensor(reward)):
                raise ValueError('Nonfinite shared TaskForce reward')
            self.reward_terms = {'training_loss_improvement': float(improvement), 'gradient_norm_penalty': float(penalty),
                                 'loss_reward_weight': 1., 'gradient_penalty_weight': .001, 'validation_used': False}
            self.replay.append((previous_observation, previous_logits, reward, observation))
        self.previous = (observation, logits, gram, values)
        coefficients = shares.double() / self.scales
        weights = dict(zip(TASKS, coefficients.tolist()))
        total = sum(weights[task] * losses[task] for task in TASKS)
        norms = gram.diag().clamp_min(0).sqrt()
        cosine = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
        record = {'profile': 'paper_core', 'weight_rule': 'softmax_div_gradient_scale',
                  'gradient_scope': 'all_trainable_parameters', 'gradient_normalization': self.normalization,
                  'critic_action_space': 'raw_logits', 'update_order': 'model_then_policy',
                  'weights': weights, 'actions': dict(zip(TASKS, shares.tolist())), 'action_logits': self.action_logits,
                  'base_weights': dict(zip(TASKS, self.base)), 'task_losses': dict(zip(TASKS, values.tolist())),
                  'observations': dict(zip(TASKS, observation.tolist())), 'gram': gram.tolist(), 'raw_gram': self.raw_gram.tolist(),
                  'gradient_norms': dict(zip(TASKS, norms.tolist())), 'raw_gradient_norms': dict(zip(TASKS, self.raw_gram.diag().sqrt().tolist())),
                  'gradient_scales': dict(zip(TASKS, self.scales.tolist())), 'cosine_similarity': cosine.clamp(-1, 1).tolist(),
                  'gradient_parameter_tensors': len(parameters), 'gradient_parameter_elements': sum(p.numel() for p in parameters),
                  'conflict_pairs': [[TASKS[i], TASKS[j]] for i in range(self.count) for j in range(i+1, self.count) if gram[i, j] < 0],
                  'reward': reward, 'reward_terms': copy.deepcopy(self.reward_terms), 'replay_size': len(self.replay),
                  'noise_type': 'ornstein_uhlenbeck', 'noise_scale': self.noise_scale,
                  'action_bound': self.action_bound,
                  'policy_updates': self.updates, 'actor_optimizer_steps': self.count * self.updates,
                  'critic_optimizer_steps': self.count * self.updates, 'model_steps': self.model_steps,
                  'model_step_completed': False}
        self.pending_model_step = True
        self.last_record = record
        return total, record

    def after_model_step(self):
        if not self.pending_model_step:
            raise RuntimeError('No pending TaskForce model step')
        self.pending_model_step = False
        self.model_steps += 1
        if len(self.replay) >= self.batch_size:
            self.update()
        self.last_record.update(policy_updates=self.updates, actor_optimizer_steps=self.count * self.updates,
                                critic_optimizer_steps=self.count * self.updates, model_steps=self.model_steps,
                                model_step_completed=True, **self.learning_stats)
        return copy.deepcopy(self.last_record)

    def update(self):
        if self.pending_model_step:
            raise RuntimeError('Policy learning must follow the model optimizer step')
        indices = torch.randperm(len(self.replay), generator=self.generator)[:self.batch_size].tolist()
        batch = [self.replay[index] for index in indices]
        states = torch.stack([row[0] for row in batch])
        actions = torch.stack([row[1] for row in batch])
        rewards = torch.tensor([row[2] for row in batch]).unsqueeze(1)
        next_states = torch.stack([row[3] for row in batch])
        reward_std = rewards.std(unbiased=False)
        if len(batch) > 1 and reward_std > 1e-6:
            rewards = (rewards - rewards.mean()) / reward_std
        flat = states.flatten(1)
        with torch.no_grad():
            next_actions = torch.cat([actor(next_states[:, i]) for i, actor in enumerate(self.target_actors)], dim=1)
            target_input = torch.cat((next_states.flatten(1), next_actions), dim=1)
            joint_actions = torch.cat([actor(states[:, i]) for i, actor in enumerate(self.actors)], dim=1)
        self.learning_stats = {'critic_td_losses': {}, 'actor_objectives': {}}
        for i, task in enumerate(TASKS):
            with torch.no_grad():
                target = rewards + .95 * self.target_critics[i](target_input)
            error = (self.critics[i](torch.cat((flat, actions), dim=1)) - target).square().mean()
            self.critic_optimizers[i].zero_grad()
            error.backward()
            self.critic_optimizers[i].step()
            candidate = joint_actions.clone()
            candidate[:, i:i+1] = self.actors[i](states[:, i])
            self.critics[i].requires_grad_(False)
            objective = -self.critics[i](torch.cat((flat, candidate), dim=1)).mean()
            self.actor_optimizers[i].zero_grad()
            objective.backward()
            self.actor_optimizers[i].step()
            self.critics[i].requires_grad_(True)
            self.learning_stats['critic_td_losses'][task] = float(error.detach())
            self.learning_stats['actor_objectives'][task] = float(objective.detach())
            with torch.no_grad():
                for source, target_module in ((self.actors[i], self.target_actors[i]), (self.critics[i], self.target_critics[i])):
                    for current, target_parameter in zip(source.parameters(), target_module.parameters()):
                        target_parameter.lerp_(current, .01)
        if not all(torch.isfinite(p).all() for group in (self.actors, self.critics) for module in group for p in module.parameters()):
            raise ValueError('Nonfinite TaskForce policy parameters')
        self.updates += 1

    def state_dict(self):
        state = super().state_dict()
        state.update(version=2, profile='paper_core', normalization=self.normalization,
                     action_bound=self.action_bound,
                     noise_state=self.noise_state.clone(), action_steps=self.action_steps,
                     model_steps=self.model_steps, pending_model_step=self.pending_model_step)
        return state

    def load_state_dict(self, state):
        if (state.get('version') != 2 or state.get('profile') != 'paper_core'
                or state.get('normalization') != self.normalization or state.get('action_bound') != self.action_bound):
            raise ValueError('TaskForce core configuration differs from checkpoint')
        adapted = {**state, 'version': 1}
        super().load_state_dict(adapted)
        self.noise_state = state['noise_state'].clone()
        self.action_steps = state['action_steps']
        self.model_steps = state['model_steps']
        self.pending_model_step = state['pending_model_step']


def make_team(base_weights, seed=17, batch_size=None, profile='paper', normalization='l2', action_bound=None):
    if profile == 'paper':
        return PaperTaskForce(seed, 64 if batch_size is None else batch_size, normalization, action_bound)
    if profile == 'legacy':
        if action_bound is not None:
            raise ValueError('Bounded actions require the paper core profile')
        return GradientTeam(base_weights, seed, 4 if batch_size is None else batch_size)
    raise ValueError('Unknown TaskForce profile')


def policy_parameters(model, profile='paper', probe_budget=50000, probe_modules=None):
    if profile == 'legacy':
        from .taskforce import select_probes
        return select_probes(model, probe_budget, probe_modules)
    if profile != 'paper' or probe_modules is not None:
        raise ValueError('The paper core requires every trainable parameter')
    selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not selected:
        raise ValueError('The model has no trainable parameters')
    names, parameters = zip(*selected)
    return list(parameters), list(names)
