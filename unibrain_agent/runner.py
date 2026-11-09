import gc
import json
import random
import shutil
import time
import traceback
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .agents import Agent, balance_weights
from .data import batch, digest, inspect_data, load_split, templates
from .paths import PROJECT, RUNS, UPSTREAM
from .pipeline import MODULES, TASKS, Pipeline, PipelineConfig, loss_terms
from .research import evidence_bundle
from .upstream import SYNTHSTRIP_COMMIT, UNIBRAIN_COMMIT, verify_sources
from .upstream import unibrain_core
from .validation import gate_candidate, gradients, spatial_checks

REFERENCE_WEIGHTS = dict(zip(TASKS, [1.0, 0.1, 1.0, 1.0]))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dice(prediction, target):
    prediction, target = prediction.bool(), target.bool()
    return float((2 * (prediction & target).sum() + 1) / (prediction.sum() + target.sum() + 1))


def evaluate(model, array, template, emit=None):
    model.eval()
    records = []
    with torch.no_grad():
        for row in array:
            sample = batch(row)
            output = model(sample, template)
            terms = loss_terms(output, sample, template)
            # A fixed reference objective keeps changes in training weights comparable.
            terms["segmentation"] = F.cross_entropy(output["tissue_logits"], sample["raw_img_gm_mask_np"][:, 0].long())
            terms["registration"] = unibrain_core().GCC()(output["warped"], template["image"])
            predicted = output["tissue_logits"].argmax(1, keepdim=True)
            records.append({"id": sample["id"], "label": int(sample["label"][0]),
                            "prediction": int(output["logits"].argmax(1)[0]),
                            "class_probability": float(output["logits"].softmax(1)[0, 1]),
                            "extraction_dice": dice(output["mask"] >= 0.5, sample["raw_img_brain_mask_np"]),
                            "tissue_dice": sum(dice(predicted == k, sample["raw_img_gm_mask_np"] == k) for k in (1, 2, 3)) / 3,
                            "losses": {key: float(value) for key, value in terms.items()}})
            if emit:
                emit("validation_progress", f"Evaluated {len(records)}/{len(array)} validation samples",
                     completed=len(records), total=len(array))
    means = {task: sum(r["losses"][task] for r in records) / len(records) for task in TASKS}
    return {"count": len(records), "records": records, "mean_losses": means,
            "reference_loss": sum(REFERENCE_WEIGHTS[k] * v for k, v in means.items()),
            "extraction_dice": sum(r["extraction_dice"] for r in records) / len(records),
            "tissue_dice": sum(r["tissue_dice"] for r in records) / len(records),
            "accuracy": sum(r["prediction"] == r["label"] for r in records) / len(records),
            "reference_definition": "BCE extraction + 0.1 GCC registration + ground-truth tissue CE + diagnosis CE; deterministic soft forward",
            "classification_interpretation": "Smoke-test metric only: training data contain one class"}


@contextmanager
def trace_modules(model, emit):
    from .catalog import catalog
    stages = {"denoise_net": "denoising", "bias_net": "bias", "ext_net": "extraction", "norm_net": "normalization",
              "reg_net": "registration", "seg_net": "segmentation", "par_net": "parcellation",
              "graph_gen": "graph", "gnn_pred": "classification"}
    choices = model.config.stages or {}
    library = {stage["id"]: stage for stage in catalog()["stages"]}
    handles = []
    starts = {}

    def attach(key, module, label):
        def before(module, inputs):
            starts[key] = time.perf_counter()
            emit("module", label + " is running", module=key, label=label, action="started")

        def after(module, inputs, output):
            emit("module", label + " finished", module=key, label=label, action="completed",
                 duration=round(time.perf_counter() - starts[key], 4))

        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

    for key, module in model.named_children():
        if key not in MODULES or isinstance(module, torch.nn.Identity):
            continue
        stage = library.get(stages.get(key))
        option = next((item for item in stage["options"] if item["id"] == choices.get(stage["id"])), None) if stage else None
        label = option["label"].split(" (")[0] if option and option["research"] else stage["name"] if stage else "ROI encoder"
        attach(key, module, label)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def train_branch(model, train, val, template, steps, seed, folder, name, emit, agent=None, extraction_loss="bce"):
    with trace_modules(model, emit):
        return _train_branch(model, train, val, template, steps, seed, folder, name, emit, agent, extraction_loss)


def _train_branch(model, train, val, template, steps, seed, folder, name, emit, agent, extraction_loss):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-6)
    initial = {key: value.detach().clone() for key, value in model.named_parameters() if value.requires_grad}
    weights, history, control = dict(REFERENCE_WEIGHTS), [], None
    training_seconds = 0.0
    order = np.random.default_rng(seed).permutation(len(train)).tolist()
    for step in range(steps):
        model.train()
        torch.manual_seed(seed + step)
        sample = batch(train[order[step % len(order)]])
        optimizer.zero_grad(set_to_none=True)
        emit("phase", f"Model is running: training step {step + 1}/{steps}", phase="training", operation="forward")
        started = time.perf_counter()
        output = model(sample, template, augment=True)
        terms = loss_terms(output, sample, template, extraction_loss)
        training_seconds += time.perf_counter() - started
        if agent is not None and step == 1:
            emit("phase", "Checking task gradients for the training coordinator", phase="coordination")
            diagnostic = gradients(model, terms, output)
            control = agent.coordinate(diagnostic, weights)
            previous = dict(weights)
            if control["action"] == "balance_gradients":
                weights = balance_weights(weights, diagnostic["task_norms"])
            control = {**control, "weights_before": previous, "weights_after": dict(weights),
                       "step": step + 1, "diagnostic": diagnostic}
        started = time.perf_counter()
        emit("phase", f"Backpropagation and parameter update {step + 1}/{steps}", phase="training", operation="backward")
        total = sum(weights[key] * value for key, value in terms.items())
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"Nonfinite loss in {name}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        history.append({"step": step + 1, "subject_id": sample["id"], "weighted_loss": float(total.detach()),
                        "losses": {key: float(value.detach()) for key, value in terms.items()}, "weights": dict(weights)})
        emit("training", f"{name}: step {step + 1}/{steps}", branch=name, **history[-1])
        del output, terms, total
    updated = {}
    for module in MODULES:
        values = [(key, value) for key, value in model.named_parameters() if key in initial and key.startswith(module + ".")]
        if values:
            updated[module] = sum(int(torch.count_nonzero(value.detach() != initial[key])) for key, value in values)
    del initial
    model.eval()
    emit("phase", "Measuring learning signals and spatial consistency", phase="audit")
    torch.manual_seed(seed + 1000)
    sample = batch(train[order[-1]])
    output = model(sample, template, augment=True)
    terms = loss_terms(output, sample, template, extraction_loss)
    diagnostic = gradients(model, terms, output)
    spatial = spatial_checks(output)
    preview = {key: output[key].detach().cpu().numpy() for key in ("mask", "extracted", "warped", "edges")}
    preview["input"] = sample["raw_img_np"].numpy()
    del output, terms
    emit("phase", f"Evaluating the model on {len(val)} validation samples", phase="validation")
    validation = evaluate(model, val, template, emit)
    emit("phase", "Saving trained weights and volume previews", phase="saving")
    checkpoint = folder / f"{name}.pt"
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "weights": weights, "seed": seed,
                "steps": steps, "extraction_loss": extraction_loss}, checkpoint)
    result = {"name": name, "model": model.description(), "steps": steps, "history": history,
              "training_seconds": training_seconds, "validation": validation, "gradients": diagnostic,
              "spatial": spatial, "updated_parameters": updated, "coordination": control,
              "checkpoint": checkpoint.name, "extraction_loss": extraction_loss,
              "training_settings": {"optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-6,
                                    "gradient_clip_norm": 5.0, "feature_noise_std": 0.05}}
    write_json(folder / f"{name}.json", result)
    np.savez_compressed(folder / f"{name}_preview.npz", **preview)
    emit("validation", f"{name}: validation completed", reference_loss=validation["reference_loss"], spatial=spatial)
    return result


def run(backend="codex", steps=5, seed=17, data_dir=None, output_dir=None, on_event=None, build_dir=None, comparison=True):
    if not 2 <= steps <= 100:
        raise ValueError("Use 2 to 100 steps per branch")
    torch.set_num_threads(4)
    seed_everything(seed)
    folder = Path(output_dir) if output_dir else RUNS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "created": datetime.now(timezone.utc).isoformat(), "backend": backend,
              "seed": seed, "steps_per_branch": steps, "run_dir": str(folder.resolve()),
              "mode": "comparison" if comparison else "single",
              "upstream_commit": UNIBRAIN_COMMIT, "synthstrip_commit": SYNTHSTRIP_COMMIT,
              "environment": {"torch": torch.__version__, "device": "cpu", "threads": 4},
              "events": []}

    def emit(kind, message, **extra):
        event = {"seconds": round(time.perf_counter() - started, 3), "kind": kind, "message": message, **extra}
        report["events"].append(event)
        with (folder / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(folder / "report.json", report)
        if kind != "module":
            print(f"[{event['seconds']:7.1f}s] {message}", flush=True)
        if on_event:
            on_event(event, folder)

    try:
        emit("phase", "Loading the selected pipeline and checking source files", phase="preparing")
        report["source_verification"] = verify_sources()
        report["implementation_hashes"] = {str(path.relative_to(PROJECT)): digest(path)
                                            for path in sorted((PROJECT / "unibrain_agent").glob("*.py"))}
        data_dir = Path(data_dir) if data_dir else UPSTREAM / "dataset"
        train, val = load_split(data_dir, "train"), load_split(data_dir, "val")
        template = templates(data_dir)
        report["data"] = inspect_data(data_dir, train, val)
        emit("data", "Validated original GitHub sample format and atlas label mapping")
        evidence = evidence_bundle()
        write_json(folder / "research_evidence.json", evidence)
        agent = Agent(backend, folder / "agent", emit)
        if build_dir:
            from .builds import verify_build
            build = verify_build(build_dir)
            plan = {**build["plan"], "extraction": "synthstrip" if build["selection"]["extraction"] == "M02" else "unibrain",
                    "soft_atlas": True, "continuous_edges": True, "stages": build["selection"]}
            report["build"] = {"id": build["id"], "selection": build["selection"], "code_sha256": build["code_sha256"],
                               "validation": build["validation"], "sources": build["sources"], "agent_calls": build["agent_calls"]}
            for name in ("pipeline.py", "build.json", "validation.json"):
                shutil.copy2(Path(build_dir) / name, folder / ("build_" + name))
        else:
            plan = agent.plan(evidence)
        report["plan"] = plan
        report["research"] = {key: value for key, value in evidence.items() if key != "connection_code"}
        seed_everything(seed)
        initial_state = {}
        if comparison:
            baseline_model = Pipeline()
            initial_state = {key: value.clone() for key, value in baseline_model.state_dict().items()}
            report["baseline"] = train_branch(baseline_model, train, val, template, steps, seed, folder, "baseline", emit)
            del baseline_model
            gc.collect()
        config = PipelineConfig(plan["extraction"], plan["soft_atlas"], plan["continuous_edges"])
        seed_everything(seed)
        if build_dir:
            from .builds import load_pipeline
            candidate_model = load_pipeline(build_dir)
            config = candidate_model.config
        else:
            candidate_model = Pipeline(config)
        candidate_state = candidate_model.state_dict()
        shared = {key: value for key, value in initial_state.items()
                  if key in candidate_state and candidate_state[key].shape == value.shape and
                  (not key.startswith("ext_net.") or config.extraction == "unibrain")}
        candidate_model.load_state_dict(shared, strict=False)
        if comparison:
            report["comparison"] = {"shared_initial_tensors": len(shared), "same_seed": True, "same_subject_order": True,
                                "same_step_budget": True, "same_reference_objective": True,
                                "independent_extraction_initialization": config.extraction != "unibrain",
                                "is_single_factor_ablation": False}
        del initial_state, shared, candidate_state
        emit("execution", "Applied source-backed module and connection plan", config=asdict(config))
        report["candidate"] = train_branch(candidate_model, train, val, template, steps, seed, folder, "candidate",
                                            emit, agent=agent, extraction_loss=plan["extraction_loss"])
        del candidate_model
        selected = "candidate"
        if comparison:
            report["gate"] = gate_candidate(report["baseline"], report["candidate"])
            review_summary = {"gate": report["gate"], "plan": plan, "data_warnings": report["data"]["warnings"]}
            for branch in ("baseline", "candidate"):
                result = report[branch]
                review_summary[branch] = {"training_seconds": result["training_seconds"], "gradients": result["gradients"],
                                          "validation": {key: value for key, value in result["validation"].items() if key != "records"}}
            emit("phase", "Reviewing the measured comparison", phase="review")
            report["review"] = agent.review(review_summary)
            selected = "candidate" if report["gate"]["decision"] == "keep" and report["review"]["recommendation"] == "keep" else "baseline"
        shutil.copy2(folder / f"{selected}.pt", folder / "selected.pt")
        report["selection"] = {"branch": selected, "checkpoint": "selected.pt",
                               "sha256": digest(folder / "selected.pt"),
                               "action": "saved" if not comparison else "keep" if selected == "candidate" else "revert",
                               "reason": "Saved the selected pipeline. No comparison or acceptance claim." if not comparison else
                                         "Candidate passed both the hard gate and review" if selected == "candidate" else "Restored the baseline checkpoint"}
        report["agent_calls"] = (build["agent_calls"] if build_dir else []) + agent.calls
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "complete"
        emit("phase", "Writing the experiment report", phase="saving")
        from .report import render
        render(report, folder)
        write_json(RUNS / "latest.json", {"run_dir": str(folder.resolve())})
        emit("complete", "Run completed. Weights, images and training records saved.")
        return report
    except Exception as error:
        if (folder / "baseline.pt").exists():
            shutil.copy2(folder / "baseline.pt", folder / "selected.pt")
            report["selection"] = {"branch": "baseline", "checkpoint": "selected.pt", "action": "revert",
                                   "sha256": digest(folder / "selected.pt"),
                                   "reason": "Candidate or review failed; baseline restored"}
        report["status"] = "failed"
        report["error"] = str(error)
        (folder / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        emit("failed", str(error))
        raise
