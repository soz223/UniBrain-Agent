import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from .builds import load_pipeline
from .data import batch, load_split, templates
from .paths import UPSTREAM
from .pipeline import MODULES, loss_terms
from .runner import REFERENCE_WEIGHTS, seed_everything, write_json
from .validation import gradients, spatial_checks
from .runtime import device_for, move_model


def validate(folder):
    folder = Path(folder)
    events = []

    def progress(phase, message):
        events.append({"phase": phase, "message": message, "at": datetime.now(timezone.utc).isoformat()})
        write_json(folder / "progress.json", {**events[-1], "events": events})

    progress("training", "Loading models and the GitHub training sample")
    record = json.loads((folder / "build.json").read_text(encoding="utf-8"))
    runtime = record.get("runtime", {})
    torch.set_num_threads(runtime.get("threads", 4))
    device = device_for(runtime.get("device"))
    data_dir = Path(runtime.get("data", UPSTREAM / "dataset"))
    seed_everything(17)
    model = move_model(load_pipeline(folder, ready=False), device)
    sample = batch(load_split(data_dir, "train")[0], device)
    sample["raw_img_np"].requires_grad_()
    template = templates(data_dir, device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    named = {key: p for key, p in model.named_parameters() if p.requires_grad}
    before = {key: p.detach().clone() for key, p in named.items()}
    passes = []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sample["raw_img_np"].grad = None
        progress("training", f"Optimizer step {step + 1}/2: forward pass")
        output = model(sample, template)
        terms = loss_terms(output, sample, template, record["plan"]["extraction_loss"])
        total = sum(REFERENCE_WEIGHTS[key] * value for key, value in terms.items())
        from .objective import observe_objective
        objective = observe_objective(model, terms, REFERENCE_WEIGHTS, total, step + 1)
        objective.update(profile='Executed compact build check', selection=record['selection'], phase='build_validation')
        progress("training", f"Optimizer step {step + 1}/2: backpropagation and gradient checks")
        total.backward()
        missing = [name for name, p in named.items() if p.grad is None]
        nonfinite = [name for name, p in named.items() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        input_grad = sample["raw_img_np"].grad
        input_ok = input_grad is not None and bool(torch.isfinite(input_grad).all()) and float(input_grad.norm()) > 0
        if not bool(torch.isfinite(total)) or missing or nonfinite or not input_ok:
            raise ValueError(f"Trainability failed at step {step + 1}: missing={missing}, nonfinite={nonfinite}, input_gradient={input_ok}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        progress("training", f"Optimizer step {step + 1}/2 completed")
        passes.append({"step": step + 1, "loss": float(total.detach()), "parameter_tensors": len(named),
                       "finite_gradient_tensors": len(named), "input_gradient_norm": float(input_grad.norm())})
        del output, terms, total
    updates = {}
    for name in MODULES:
        members = [(key, p) for key, p in named.items() if key.startswith(name + ".")]
        if members:
            updates[name] = {"parameter_tensors": len(members),
                             "updated_tensors": sum(not torch.equal(before[key], p.detach()) for key, p in members)}
    model.eval()
    progress("audit", "Measuring loss to module gradients and spatial consistency")
    output = model(sample, template)
    terms = loss_terms(output, sample, template, record["plan"]["extraction_loss"])
    audit = gradients(model, terms, output)
    spatial = spatial_checks(output)
    failures = []
    if not spatial["passed"]:
        failures.append("Spatial checks failed")
    failures.extend(name + " had no observable parameter update" for name, info in updates.items() if info["updated_tensors"] == 0)
    for name in ("ext_net", "reg_net", "feature_net", "gnn_pred"):
        if audit["matrix"]["classification"][name]["state"] != "active":
            failures.append("Classification gradient did not reach " + name)
    edge = audit["classification_to_graph_weights"]
    if not edge["connected"] or not edge["norm"]:
        failures.append("Classification gradient did not reach graph weights")
    progress("audit", "Gradient and spatial checks completed" if not failures else "; ".join(failures))
    return {"passed": not failures, "error": "; ".join(failures) or None, "passes": passes,
            "objective": objective,
            "updates": updates, "gradients": audit, "spatial": spatial, "model": model.description(),
            "scope": "Two optimizer steps on one original GitHub training sample. Random weights. Not an accuracy evaluation."}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    try:
        result = validate(folder)
    except Exception as error:
        traceback.print_exc()
        result = {"passed": False, "error": str(error)}
    write_json(folder / "validation.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "error")}))
    sys.exit(0 if result["passed"] else 1)
