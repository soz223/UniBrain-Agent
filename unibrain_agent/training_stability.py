"""Audited TaskForce stabilization and training-only checkpoint recovery."""

import argparse
import copy
import math
import os
import shutil
import uuid
from pathlib import Path


PREVIOUS_TRAINERS = {
    "209c1e88d5aadabf55a61a36e132d2c862e8dc84244979a62b6f474f00463cd4",
    "0a6be5da0464ade9b52cba1135a9189121d2322448f767e67b157240fae587f8",
}
PREVIOUS_POLICY = "9b6cd71300eb294cd15ceaf58fdbb3294610c26da28db71b7671ead264d1f781"
PREVIOUS_CONTROL = "84ee53065fc03eb0bc6cb34e8ca02913b3b02b07eb811da1ea1b5ccf313d6d71"


def retained_copy(source, destination):
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def install_recovery(checkpoint, folder, attempt):
    for name in ("last.pt", "best.pt"):
        path = folder / name
        if path.exists():
            retained_copy(path, folder / ("before_recovery_" + str(attempt) + "_" + name))
    retained_copy(checkpoint, folder / "last.pt")
    if (folder / "stable_best.pt").exists():
        retained_copy(folder / "stable_best.pt", folder / "best.pt")


def upgrade_identity(previous, current):
    migrated = copy.deepcopy(previous)
    hashes = migrated["implementation_hashes"]
    if (previous.get("coordination") != "taskforce"
            or previous.get("taskforce_gradient_normalization") != "l2"
            or current.get("taskforce_gradient_normalization") != "bounded_l2"
            or hashes.get("replacement_training.py") not in PREVIOUS_TRAINERS
            or hashes.get("taskforce_paper.py") != PREVIOUS_POLICY):
        raise ValueError("Unsupported stability checkpoint migration")
    migrated["taskforce_gradient_normalization"] = "bounded_l2"
    for name in ("replacement_training.py", "taskforce_paper.py", "training_stability.py"):
        hashes[name] = current["implementation_hashes"][name]
    if hashes.get("training_control.py") == PREVIOUS_CONTROL:
        hashes["training_control.py"] = current["implementation_hashes"]["training_control.py"]
    if migrated != current:
        raise ValueError("Stability migration cannot change data, model or unrelated configuration")


def check_training_loss(loss, state):
    value = float(loss.detach())
    if not math.isfinite(value):
        raise ValueError("Stability guard: nonfinite training classification loss")
    window = state.setdefault("loss_window", [])
    window.append(value)
    del window[:-32]
    if len(window) >= 16 and sum(window) / len(window) > 20:
        raise ValueError("Stability guard: excessive training classification loss")


def stable_probe(metrics):
    loss = metrics.get("classification_loss", math.inf)
    fraction = metrics.get("predicted_positive_fraction", -1)
    return math.isfinite(loss) and loss < 5 and .05 < fraction < .95


def qualify_probe(metrics, progress, completed_epochs, bootstrap_epochs=0):
    state = progress.setdefault('stability_bootstrap', {'qualified': False, 'epoch_limit': bootstrap_epochs})
    if stable_probe(metrics):
        state['qualified'] = True
        return True
    loss = metrics.get('classification_loss', math.inf)
    fraction = metrics.get('predicted_positive_fraction', -1)
    if (not math.isfinite(loss) or loss >= 5 or not 0 <= fraction <= 1
            or state['qualified'] or completed_epochs >= bootstrap_epochs):
        raise ValueError('Stability guard: training probe classification collapse')
    return False


def recover(source, destination, attempt):
    import torch

    if not 1 <= attempt <= 3 or source.resolve() == destination.resolve():
        raise ValueError("Invalid stability recovery attempt")
    state = torch.load(source, map_location="cpu", weights_only=True)
    mode = state["identity"].get("taskforce_gradient_normalization")
    if state.get("checkpoint_kind") != "resume" or mode not in ("bounded_l2", "anchored"):
        raise ValueError("Recovery requires a complete stabilized checkpoint")
    rates = [group["lr"] * (1. if mode == "anchored" else .5 ** attempt) for group in state["optimizer"]["param_groups"]]
    if min(rates) < 1e-7:
        raise ValueError("Recovery learning rate floor reached")
    for group, rate in zip(state["optimizer"]["param_groups"], rates):
        group["lr"] = rate
    state["progress"]["gradient_health"] = {}
    state["progress"]["classification_health"] = {}
    state["progress"]["stability_recovery"] = {"attempt": attempt, "learning_rates": rates}
    if mode == "anchored":
        from .anchored_taskforce import AnchoredTaskForce
        old = state["coordination"]
        radius = min(old["trust_radius"], .1 * .5 ** attempt)
        team = AnchoredTaskForce(old["base_weights"], state["identity"]["seed"], old["batch_size"], old["action_bound"], radius)
        team.reward_mode = old.get('reward_mode', 'same_subject')
        team.model_steps = state["progress"]["steps"]
        state["coordination"] = team.state_dict()
        state["progress"]["stability_recovery"].update(action="reset_policy_and_contract_trust_region", trust_radius=radius)
    else:
        state["coordination"].update(replay=[], previous=None, pending_model_step=False)
    temp = destination.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(destination)
    return rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("attempt", type=int)
    args = parser.parse_args()
    print(recover(args.source, args.destination, args.attempt))
