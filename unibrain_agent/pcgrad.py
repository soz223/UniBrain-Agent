"""PCGrad in task coordinates, using every trainable parameter."""

import torch

from .taskforce import TASKS
from .taskforce_paper import full_gradient_gram


def project_weights(losses, base_weights, parameters, seed, step):
    scaled = {task: base_weights[task] * losses[task] for task in TASKS}
    gram = full_gradient_gram(scaled, parameters)
    rows = torch.eye(len(TASKS), dtype=gram.dtype)
    order_rng = torch.Generator().manual_seed((seed << 32) + step)
    for i, row in enumerate(rows):
        for j in torch.randperm(len(TASKS), generator=order_rng).tolist():
            if i == j or gram[j, j] <= 0:
                continue
            projection = row.dot(gram[:, j]) / gram[j, j]
            row[j] -= projection.clamp(max=0)
    return {task: base_weights[task] * weight for task, weight in zip(TASKS, rows.sum(0).tolist())}
