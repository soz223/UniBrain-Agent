"""Verify learned coordinator parameters against their seeded initialization."""

import hashlib
import json
import math
from pathlib import Path

import torch

from .taskforce import TASKS
from .taskforce_paper import make_team


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_digest(name="taskforce.py"):
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def finite_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(item) for item in value)
    return math.isfinite(value) if isinstance(value, float) else True


def parameter_delta(initial, trained):
    if initial.keys() != trained.keys():
        raise ValueError("Coordinator parameter keys differ from the seeded architecture")
    changed, total, l1, maximum = 0, 0, 0., 0.
    for name, before in initial.items():
        after = trained[name]
        if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
            raise ValueError("Coordinator parameter shape, dtype or finiteness is invalid")
        difference = after.double() - before.double()
        changed += int(not torch.equal(before, after))
        total += before.numel()
        l1 += float(difference.abs().sum())
        maximum = max(maximum, float(difference.abs().max()))
    return {"changed_tensors": changed, "tensor_count": len(initial), "parameters": total,
            "delta_l1": l1, "delta_max": maximum, "finite": True}


def inspect_policy(checkpoint_path, report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    config, identity = report.get("config", {}), checkpoint.get("identity", {})
    if report.get("status") not in ("complete", "budget_exhausted") or checkpoint.get("checkpoint_kind") != "resume":
        raise ValueError("A completed training report and complete resumable checkpoint are required")
    for key in ("seed", "coordination_batch_size", "coordination", "method", "setting", "dataset", "fold"):
        if key not in config or key not in identity or identity[key] != config[key]:
            raise ValueError("Checkpoint and report identities disagree: " + key)
    for key in ("method", "setting", "seed"):
        if report.get(key) != identity[key]:
            raise ValueError("Top level training identity differs from its checkpoint")
    if identity["coordination"] != "taskforce":
        raise ValueError("The checkpoint does not use TaskForce coordination")
    implementation_hash = source_digest()
    if identity.get("implementation_hashes", {}).get("taskforce.py") != implementation_hash:
        raise ValueError("Coordinator implementation differs from the checkpoint source")
    profile = identity.get("taskforce_profile", "legacy")
    normalization = identity.get("taskforce_gradient_normalization", "l2")
    if profile == "paper":
        for key in ("taskforce_profile", "taskforce_gradient_normalization"):
            if config.get(key) != identity.get(key):
                raise ValueError("Checkpoint and report identities disagree: " + key)
        if config.get("taskforce_action_bound") != identity.get("taskforce_action_bound"):
            raise ValueError("Checkpoint and report action bounds differ")
        if identity.get("implementation_hashes", {}).get("taskforce_paper.py") != source_digest("taskforce_paper.py"):
            raise ValueError("Paper coordinator implementation differs from the checkpoint source")
    policy = checkpoint.get("coordination")
    summary = report.get("coordination", {})
    if not isinstance(policy, dict) or summary.get("mode") != "taskforce":
        raise ValueError("Complete coordinator checkpoint and report evidence are required")
    if not finite_state(policy):
        raise ValueError("The coordinator contains nonfinite parameters or state")
    if tuple(policy.get("tasks", ())) != TASKS:
        raise ValueError("Coordinator task identities differ")
    declared_weights = summary.get("base_weights", {})
    if set(declared_weights) != set(TASKS):
        raise ValueError("All four base task weights must be declared")
    base = tuple(declared_weights[name] for name in TASKS)
    options = dict(seed=identity["seed"], batch_size=identity["coordination_batch_size"], profile=profile,
                   normalization=normalization, action_bound=identity.get("taskforce_action_bound"))
    initial = make_team(base, **options)
    trained = make_team(base, **options)
    trained.load_state_dict(policy)
    progress = checkpoint.get("progress", {})
    steps, updates = progress.get("steps"), policy.get("policy_updates")
    if (not isinstance(steps, int) or isinstance(steps, bool) or steps != report.get("steps")
            or not isinstance(updates, int) or isinstance(updates, bool) or updates < 1
            or updates > steps - progress.get("pretrain_steps", 0)):
        raise ValueError("Positive policy updates must match the completed training steps")
    if summary.get("last_decision") != policy.get("last_record") or not policy.get("last_record"):
        raise ValueError("The last policy decision differs between report and checkpoint")
    if profile == "paper" and (policy.get("pending_model_step") or policy.get("model_steps") != steps - progress.get("pretrain_steps", 0)
                                or not policy["last_record"].get("model_step_completed")):
        raise ValueError("Policy learning must follow completed model updates")
    counts = {"policy_updates": updates, "actor_optimizer_steps": len(TASKS) * updates,
              "critic_optimizer_steps": len(TASKS) * updates}
    if any(summary.get(key) != value or policy["last_record"].get(key) != value for key, value in counts.items()):
        raise ValueError("Policy optimizer counts disagree between checkpoint and report")
    for group in ("actor_optimizers", "critic_optimizers"):
        for optimizer in policy[group]:
            states = list(optimizer.get("state", {}).values())
            if not states or any(float(state.get("step", -1)) != updates for state in states):
                raise ValueError("Coordinator optimizer state does not support its update count")
    deltas = {}
    for group in ("actors", "critics", "target_actors", "target_critics"):
        deltas[group] = {task: parameter_delta(before.state_dict(), after.state_dict())
                         for task, before, after in zip(TASKS, getattr(initial, group), getattr(trained, group))}
    if any(value["changed_tensors"] == 0 for group in ("actors", "critics") for value in deltas[group].values()):
        raise ValueError("Positive counters do not establish learning: at least one actor or critic is unchanged")
    action_check = None
    if profile == "paper":
        observation = torch.tensor([policy["last_record"]["observations"][task] for task in TASKS])
        with torch.no_grad():
            before = torch.cat([actor(observation[i]) for i, actor in enumerate(initial.actors)]).softmax(0)
            after = torch.cat([actor(observation[i]) for i, actor in enumerate(trained.actors)]).softmax(0)
        action_check = {"initial_shares": dict(zip(TASKS, before.tolist())), "trained_shares": dict(zip(TASKS, after.tolist())),
                        "share_delta_max": float((after - before).abs().max()), "exploration": False,
                        "scope": "Same recorded observation; different deterministic policy weights after learning"}
        if not torch.isfinite(after).all() or action_check["share_delta_max"] == 0:
            raise ValueError("Policy parameters changed but no functional weighting change was observed")
    return {"status": "passed", "device": "cpu", "gpu_used": False, "torch_version": str(torch.__version__), "seed": identity["seed"],
            "tasks": list(TASKS), "profile": profile, "batch_size": initial.batch_size, "base_weights": declared_weights,
            "training_steps": steps, **counts, "deltas": deltas, "action_check": action_check,
            "checkpoint_sha256": digest(checkpoint_path), "report_sha256": digest(report_path),
            "coordinator_source_sha256": implementation_hash,
            "scope": "Each actor and critic changed from its seeded initialization; this does not establish policy effectiveness"}
