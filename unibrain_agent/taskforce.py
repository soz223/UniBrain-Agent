"""Task-wise actor-critic control for UniBrain loss gradients."""

import copy
from collections import deque

import torch
from torch import nn


TASKS = ("extraction", "registration", "segmentation", "classification")
PROBE_MODULES = ("preprocessor", "denoise_net", "bias_net", "norm_net", "ext_net", "reg_net", "seg_net", "par_net", "feature_net", "graph_gen", "gnn_pred")


def select_probes(model, per_module=50000, modules=None):
    if per_module < 1:
        raise ValueError("The probe budget must be positive")
    available = dict(model.named_children())
    modules = tuple(available) if modules is None else tuple(modules)
    if len(set(modules)) != len(modules) or any(name not in available for name in modules):
        raise ValueError("Probe modules must be distinct existing model children")
    chosen = []
    names = []
    for module in modules:
        used = 0
        for name, parameter in model.named_parameters():
            if name.startswith(module + ".") and parameter.requires_grad and parameter.numel() <= per_module - used:
                chosen.append(parameter)
                names.append(name)
                used += parameter.numel()
    if not chosen:
        raise ValueError("No trainable parameters available for gradient probes")
    return chosen, names


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, outputs))


class GradientTeam:
    def __init__(self, base_weights=(1.0, 0.1, 1.0, 1.0), seed=17, batch_size=4):
        self.base = tuple(base_weights)
        self.count = len(self.base)
        if self.count != len(TASKS) or batch_size < 1 or any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in self.base):
            raise ValueError("Four positive task weights and a positive replay batch are required")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actors = [network(self.count + 1, 1) for _ in self.base]
            self.critics = [network(self.count * (self.count + 2), 1) for _ in self.base]
        self.target_actors = [copy.deepcopy(actor) for actor in self.actors]
        self.target_critics = [copy.deepcopy(critic) for critic in self.critics]
        for actor in self.actors:
            nn.init.zeros_(actor[-1].weight)
            nn.init.zeros_(actor[-1].bias)
        for target, actor in zip(self.target_actors, self.actors):
            target.load_state_dict(actor.state_dict())
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]
        self.replay = deque(maxlen=512)
        self.generator = torch.Generator().manual_seed(seed)
        self.batch_size = batch_size
        self.previous = None
        self.updates = 0
        self.last_record = None

    def observe(self, losses, parameters):
        parameters = tuple(parameters)
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("Gradient probes must contain trainable parameters only")
        gradients = []
        for name, weight in zip(TASKS, self.base):
            pieces = torch.autograd.grad(losses[name], parameters, retain_graph=True, allow_unused=True) if losses[name].requires_grad else [None] * len(parameters)
            gradients.append(torch.cat([(torch.zeros_like(parameter) if piece is None else piece.detach() * weight).flatten()
                                        for parameter, piece in zip(parameters, pieces)]))
        vectors = torch.stack(gradients)
        gram = (vectors @ vectors.T).float().cpu()
        gram = (gram + gram.T) / 2
        if not torch.isfinite(gram).all():
            raise ValueError("Nonfinite task gradient Gram matrix")
        values = torch.tensor([float(losses[name].detach()) for name in TASKS])
        if not torch.isfinite(values).all() or (values <= -1).any():
            raise ValueError("Invalid task losses for gradient control")
        scale = gram.diag().mean().clamp_min(1e-12)
        observation = torch.cat((gram / scale, torch.log1p(values)[:, None]), dim=1).clamp(-100, 100)
        return observation, gram, values

    def action(self, observation, explore=True):
        with torch.no_grad():
            logits = torch.cat([actor(observation[i]) for i, actor in enumerate(self.actors)])
            if explore:
                logits += 0.05 * torch.randn(logits.shape, generator=self.generator)
            return torch.softmax(logits, dim=0)

    def learn(self, observation, gram, values):
        reward = None
        if self.previous is not None:
            old_observation, old_action, old_gram, old_values = self.previous
            improvement = (torch.log1p(old_values) - torch.log1p(values)).sum()
            mixed_norm = old_action @ old_gram @ old_action
            penalty = mixed_norm / old_gram.diag().sum().clamp_min(1e-12)
            reward = float((improvement - 0.01 * penalty).clamp(-10, 10))
            self.replay.append((old_observation, old_action, reward, observation))
            if len(self.replay) >= self.batch_size:
                self.update()
        action = self.action(observation)
        self.previous = (observation, action, gram, values)
        return action, reward

    def update(self):
        batch = list(self.replay)[-self.batch_size:]
        states = torch.stack([row[0] for row in batch])
        actions = torch.stack([row[1] for row in batch])
        rewards = torch.tensor([row[2] for row in batch]).unsqueeze(1)
        next_states = torch.stack([row[3] for row in batch])
        flat_states = states.flatten(1)
        with torch.no_grad():
            next_actions = torch.softmax(torch.cat([actor(next_states[:, i]) for i, actor in enumerate(self.target_actors)], dim=1), dim=1)
            target_input = torch.cat((next_states.flatten(1), next_actions), dim=1)
        for i in range(self.count):
            critic_input = torch.cat((flat_states, actions), dim=1)
            with torch.no_grad():
                target = rewards + 0.95 * self.target_critics[i](target_input)
            error = (self.critics[i](critic_input) - target).square().mean()
            self.critic_optimizers[i].zero_grad()
            error.backward()
            self.critic_optimizers[i].step()
            logits = torch.cat([self.actors[j](states[:, j]) if i == j else self.actors[j](states[:, j]).detach() for j in range(self.count)], dim=1)
            candidate = torch.softmax(logits, dim=1)
            objective = -self.critics[i](torch.cat((flat_states, candidate), dim=1)).mean()
            self.actor_optimizers[i].zero_grad()
            objective.backward()
            self.actor_optimizers[i].step()
            for source, target_module in ((self.actors[i], self.target_actors[i]), (self.critics[i], self.target_critics[i])):
                for current, target_parameter in zip(source.parameters(), target_module.parameters()):
                    target_parameter.data.lerp_(current.data, 0.01)
        self.updates += 1

    def combine(self, losses, parameters):
        observation, gram, values = self.observe(losses, parameters)
        action, reward = self.learn(observation, gram, values)
        weights = {name: self.count * float(action[i]) * self.base[i] for i, name in enumerate(TASKS)}
        total = sum(weights[name] * losses[name] for name in TASKS)
        record = {"weights": weights, "task_losses": dict(zip(TASKS, values.tolist())),
                  "gram": gram.tolist(), "reward": reward, "policy_updates": self.updates,
                  "actor_optimizer_steps": self.count * self.updates, "critic_optimizer_steps": self.count * self.updates,
                  "replay_size": len(self.replay)}
        self.last_record = copy.deepcopy(record)
        return total, record

    def state_dict(self):
        state = {"version": 1, "tasks": TASKS, "base_weights": self.base, "batch_size": self.batch_size,
                 "replay_capacity": self.replay.maxlen, "policy_updates": self.updates,
                 "generator": self.generator.get_state(), "replay": list(self.replay),
                 "previous": self.previous, "last_record": self.last_record}
        for name in ("actors", "critics", "target_actors", "target_critics", "actor_optimizers", "critic_optimizers"):
            state[name] = [item.state_dict() for item in getattr(self, name)]
        return copy.deepcopy(state)

    def load_state_dict(self, state):
        if (state.get("version") != 1 or tuple(state["tasks"]) != TASKS or tuple(state["base_weights"]) != self.base
                or state["batch_size"] != self.batch_size or state["replay_capacity"] != self.replay.maxlen):
            raise ValueError("Gradient coordinator configuration differs from its checkpoint")
        for name in ("actors", "critics", "target_actors", "target_critics", "actor_optimizers", "critic_optimizers"):
            objects, values = getattr(self, name), state[name]
            if len(objects) != len(values):
                raise ValueError("Incomplete gradient coordinator checkpoint")
            for item, value in zip(objects, values):
                item.load_state_dict(copy.deepcopy(value))
        self.generator.set_state(state["generator"])
        self.replay = deque(copy.deepcopy(state["replay"]), maxlen=state["replay_capacity"])
        self.previous = copy.deepcopy(state["previous"])
        self.updates = int(state["policy_updates"])
        self.last_record = copy.deepcopy(state["last_record"])

    def save(self, path):
        torch.save(self.state_dict(), path)
