import math

import torch
from torch.nn import functional as F

from .pipeline import MODULES


def gradients(model, losses, output):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    task_grads, matrix, norms = {}, {}, {}
    for task, loss in losses.items():
        values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        task_grads[task] = values
        row = {}
        for module in MODULES:
            selected = [g for (name, _), g in zip(named, values) if name.startswith(module + ".")]
            present = [g for g in selected if g is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in present)
            squared = sum(float(g.detach().square().sum()) for g in present)
            state = ("inactive" if not selected else "disconnected" if not present else
                     "nonfinite" if not finite else "active" if squared > 0 else "zero")
            row[module] = {"state": state, "norm": math.sqrt(squared) if finite else None,
                           "parameter_tensors": len(selected), "connected_tensors": len(present),
                           "nonzero_tensors": sum(bool(torch.count_nonzero(g)) for g in present)}
        matrix[task] = row
        norms[task] = math.sqrt(sum(float(g.detach().square().sum()) for g in values if g is not None))
    cosines = []
    tasks = list(losses)
    for i, first in enumerate(tasks):
        for second in tasks[i + 1:]:
            shared = [(a, b) for a, b in zip(task_grads[first], task_grads[second]) if a is not None and b is not None]
            dot = sum(float((a * b).sum()) for a, b in shared)
            norm_a = sum(float(a.square().sum()) for a, _ in shared)
            norm_b = sum(float(b.square().sum()) for _, b in shared)
            denominator = math.sqrt(norm_a * norm_b)
            cosines.append({"tasks": [first, second], "shared_parameter_tensors": len(shared),
                            "cosine": max(-1., min(1., dot / denominator)) if denominator else None})
    edges = torch.autograd.grad(losses["classification"], output["raw_edges"],
                                retain_graph=True, allow_unused=True)[0]
    edge_norm = float(edges.norm()) if edges is not None else None
    return {"matrix": matrix, "task_norms": norms, "shared_gradient_cosines": cosines,
            "classification_to_graph_weights": {"connected": edges is not None, "norm": edge_norm},
            "notes": ["Disconnected means autograd returned None; zero means a connected path has a zero derivative.",
                      "A zero-initialized affine output layer can temporarily block earlier gradients.",
                      "Graph generation has no parameters; its output gradient is checked separately.",
                      "The inactive parcellation head is excluded from gradient coverage.",
                      "Cosines use only parameters shared by both losses; null is not a conflict."]}


def spatial_checks(output):
    finite = all(bool(torch.isfinite(output[key]).all()) for key in
                 ("mask", "warped", "affine", "atlas", "logits", "edges"))
    determinant = float(torch.linalg.det(output["affine"][0, :, :3]).detach())
    grid = output["inverse_grid"].detach()
    mass_error = float((output["atlas"].sum(1) - 1).abs().max().detach())
    outside = float((grid.abs() > 1).any(-1).float().mean())
    shape_ok = list(output["warped"].shape) == [1, 1, 96, 96, 96]
    result = {"finite": finite, "affine_determinant": determinant,
            "out_of_bounds_fraction": outside, "atlas_probability_sum_error": mass_error,
            "shape_preserved": shape_ok,
            "passed": finite and shape_ok and 0.1 < determinant < 10 and mass_error < 1e-4 and outside < 0.5,
            "physical_coordinates_verified": False}
    if "forward_grid" in output:
        forward = output["forward_grid"].detach()
        jacobian = torch.stack(torch.gradient(forward, spacing=(2 / 95,) * 3, dim=(1, 2, 3)), dim=-1)
        jacobian = jacobian[..., [2, 1, 0]]
        determinants = torch.linalg.det(jacobian)
        folded = float((determinants <= 0).float().mean())
        identity = F.affine_grid(torch.eye(3, 4, device=grid.device).unsqueeze(0), output["warped"].shape, align_corners=True)
        cycle = F.grid_sample(forward.permute(0, 4, 1, 2, 3), grid, padding_mode="border", align_corners=True).permute(0, 2, 3, 4, 1)
        interior = (grid.abs() < 0.9).all(-1) & (identity.abs() < 0.9).all(-1)
        error = float((cycle[interior] - identity[interior]).abs().max()) if bool(interior.any()) else 100.0
        result.update(folding_fraction=folded, inverse_cycle_max_error=error,
                      minimum_grid_jacobian=float(determinants.min()))
        result["passed"] &= folded < 0.01 and error < 0.1
    return result


def gate_candidate(baseline, candidate, max_cost_ratio=4.0, max_loss_ratio=1.10):
    reasons = []
    if not candidate["spatial"]["passed"]:
        reasons.append("Spatial or finite-value checks failed")
    path = candidate["gradients"]["matrix"]["classification"]["reg_net"]
    if path["state"] != "active":
        reasons.append("Classification does not produce a nonzero registration gradient")
    edge = candidate["gradients"]["classification_to_graph_weights"]
    if not edge["connected"] or not edge.get("norm") or not math.isfinite(edge["norm"]):
        reasons.append("Graph-weight gradient is disconnected")
    if not all(value > 0 for value in candidate["updated_parameters"].values()):
        reasons.append("At least one active trainable module did not update")
    loss_ratio = candidate["validation"]["reference_loss"] / max(baseline["validation"]["reference_loss"], 1e-8)
    cost_ratio = candidate["training_seconds"] / max(baseline["training_seconds"], 1e-8)
    if not math.isfinite(loss_ratio) or loss_ratio > max_loss_ratio:
        reasons.append("Validation reference loss exceeds the prototype tolerance")
    if not math.isfinite(cost_ratio) or cost_ratio > max_cost_ratio:
        reasons.append("Training time exceeds the prototype cost budget")
    return {"decision": "revert" if reasons else "keep", "reasons": reasons or ["All prototype acceptance checks passed"],
            "loss_ratio": loss_ratio, "cost_ratio": cost_ratio,
            "thresholds": {"max_loss_ratio": max_loss_ratio, "max_cost_ratio": max_cost_ratio},
            "scope": "Provisional engineering decision on five validation examples, not evidence of generalization"}
