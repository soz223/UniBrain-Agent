import hashlib

from .paths import PROJECT

SPECIALISTS = {
    "extraction": "soft brain masks, extraction supervision and paths to later image stages",
    "registration": "alignment loss, sampling grids, deformation validity and downstream gradient paths",
    "segmentation": "tissue supervision, auxiliary heads and shared image representations",
    "classification": "diagnosis loss, ROI features, graph weights and upstream learning signals",
}
ROLE_SCHEMA = {"type": "object", "additionalProperties": False,
               "properties": {"summary": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}},
                              "recommendation": {"type": "string"}, "uncertainty": {"type": "string"}},
               "required": ["summary", "evidence", "recommendation", "uncertainty"]}
TEAM_SCHEMA = {"type": "object", "additionalProperties": False,
               "properties": {"action": {"type": "string", "enum": ["keep_weights", "balance_gradients", "request_measurement"]},
                              "rationale": {"type": "string"}}, "required": ["action", "rationale"]}


def prompt_for(role):
    if role in SPECIALISTS:
        return (PROJECT / "prompts/task_specialist.md").read_text(encoding="utf-8").format(task=role, focus=SPECIALISTS[role])
    if role not in {"research", "coordinator"}:
        raise ValueError("Unknown assistant role")
    return (PROJECT / f"prompts/{role}.md").read_text(encoding="utf-8")


def role_catalog():
    return [{"id": role, "name": role.title() + (" specialist" if role in SPECIALISTS else ""),
             "purpose": SPECIALISTS.get(role, "Resolve task recommendations into a bounded training decision"),
             "prompt": prompt_for(role), "sha256": hashlib.sha256(prompt_for(role).encode()).hexdigest()}
            for role in [*SPECIALISTS, "coordinator"]]


def measured_tasks(context):
    import math
    norms = (context.get("diagnostic") or {}).get("task_norms", {})
    return all(type(norms.get(task)) in (int, float) and math.isfinite(norms[task]) and norms[task] >= 0 for task in SPECIALISTS)


def run_team(agent, context, role="all"):
    if role not in {"all", "coordinator", *SPECIALISTS}:
        raise ValueError("Unknown training role")
    reviews = []
    for task in SPECIALISTS if role == "all" else [role] if role in SPECIALISTS else []:
        result = agent.decide(task, context, ROLE_SCHEMA,
                              {"summary": "Deterministic test review", "evidence": [],
                               "recommendation": "Inspect measured task gradients", "uncertainty": "No language model used"},
                              instructions=prompt_for(task))
        reviews.append({"role": task, **result})
    if role in SPECIALISTS:
        return {"roles": reviews, "action": "request_measurement", "rationale": "Single role review; no coordinator decision."}
    decision = agent.decide("coordinator", {**context, "specialists": reviews}, TEAM_SCHEMA,
                            {"action": "balance_gradients" if measured_tasks(context) else "request_measurement",
                             "rationale": "Deterministic test decision based on available gradient measurements."},
                            instructions=prompt_for("coordinator"))
    if decision["action"] == "balance_gradients" and not measured_tasks(context):
        decision = {"action": "request_measurement", "rationale": "Executor blocked balancing: finite per-task gradients are unavailable. " + decision["rationale"]}
    return {**decision, "roles": reviews}
