"""Reproducible full model comparisons with separate validation and test."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

METHODS = {"original": None, "synthstrip": "ext_net", "voxelmorph": "reg_net", "lapirn": "reg_net", "nnunet": "seg_net"}
SETTINGS = ("original", "original_completed", "joint", "independent_official", "independent_pretrained", "independent_finetuned")


def fingerprint(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def preflight(args):
    import numpy as np
    from .native import verify_original
    from .data import OriginalDataset
    from .full_replacement import check_code
    from .sources import verify

    if args.epochs < 1 or args.pretrain_epochs < 0 or args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Epoch and update budgets must be positive")
    if args.loss_scale <= 0 or args.regularization < 0:
        raise ValueError("Loss scales must be positive and regularization nonnegative")
    if (args.setting in ("original", "original_completed")) != (args.method == "original"):
        raise ValueError("Original setting and original method must be selected together")
    if args.setting == "independent_official" and args.initialization != "official":
        raise ValueError("The official independent comparator requires official weights")
    if args.initialization != "random" and args.weights is None:
        raise ValueError("Pretrained initialization requires a checkpoint and its provenance")
    if args.weights is not None and args.initialization == "random":
        raise ValueError("Use an explicit pretrained initialization when supplying weights")
    if args.setting == "independent_pretrained" and args.initialization != "training_fold":
        raise ValueError("The pretrained comparator requires training fold weights")
    if args.initialization == "training_fold" and (not args.dataset or args.fold is None or args.weights_dataset != args.dataset or args.weights_fold != args.fold):
        raise ValueError("Pretrained weights must declare the same dataset and training fold")
    if args.method == "original" and (args.weights or args.trainable != "all" or args.cut_extraction):
        raise ValueError("The original comparator cannot carry replacement ablations")
    if args.setting == "original" and (args.connect_paths or args.cut_boundary != "none" or args.undo_repair != "none" or args.repair_strategy != "soft"):
        raise ValueError("Use original_completed for connection changes to an unchanged module set")
    if not (args.connect_paths or args.setting == "original_completed") and (args.cut_boundary != "none" or args.undo_repair != "none" or args.repair_strategy != "soft"):
        raise ValueError("Connection ablations require --connect-paths")
    if args.setting.startswith("independent") and args.trainable != "all":
        raise ValueError("Independent settings train the host after freezing the candidate")
    if args.secondary_extraction:
        if args.setting != "joint" or args.trainable != "all":
            raise ValueError("Multi stage composition currently requires joint training of all modules")
        from .full_composition import composition_preflight
        composition_preflight(args)
    if args.setting == "independent_finetuned" and args.pretrain_epochs < 1:
        raise ValueError("The fine tuned comparator needs explicit standalone epochs")
    if args.method in ("voxelmorph", "lapirn") and args.setting == "independent_official" and args.host_checkpoint is None:
        raise ValueError("Dense official comparator requires a trained shared host checkpoint; random frozen affine stages are not a valid control")
    if args.host_checkpoint:
        if (not args.dataset or args.fold is None or args.host_dataset != args.dataset or args.host_fold != args.fold
                or not args.host_origin or not args.host_sha256):
            raise ValueError("Host checkpoint needs origin, hash and matching training dataset/fold")
        if fingerprint(args.host_checkpoint) != args.host_sha256:
            raise ValueError("Host checkpoint hash mismatch")
    if args.method == "synthstrip":
        if args.adapter is None:
            raise ValueError("SynthStrip requires a validated generated adapter")
        check_code(args.adapter.read_text(encoding="utf-8"))
    if args.method == "nnunet" and args.plans is None:
        raise ValueError("nnU-Net requires a real plans file")
    if args.eval_limit is not None and args.eval_limit < 1 or args.threads < 1:
        raise ValueError("Evaluation limit and thread count must be positive")
    if args.patience < 0:
        raise ValueError("Early stopping patience cannot be negative")
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError("Resume in the same output directory to retain validation selection and history")
    if args.resume and args.resume.name != "last.pt":
        raise ValueError("Resume from last.pt; best.pt is reserved for validation selected evaluation")
    for name in ("adapter", "weights", "plans", "resume", "host_checkpoint", "secondary_extraction", "previous_checkpoint", "previous_report", "extraction_weights"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            raise ValueError("Missing file: " + name)
    if args.weights and (not args.weights_origin or not args.weights_sha256):
        raise ValueError("Official weights require origin and expected SHA256")
    if args.weights and fingerprint(args.weights) != args.weights_sha256:
        raise ValueError("Official checkpoint hash mismatch")
    if args.extraction_weights and (not args.extraction_weights_origin or not args.extraction_weights_sha256 or fingerprint(args.extraction_weights) != args.extraction_weights_sha256):
        raise ValueError("Composition extraction weights require origin and matching SHA256")
    source = verify_original(args.source)
    repositories = {"synthstrip": "freesurfer_synthstrip", "voxelmorph": "reg_voxelmorph_master", "lapirn": "reg_lapirn", "nnunet": "dynamic-network-architectures"}
    required = [repositories[args.method]] if args.method in repositories else []
    if args.secondary_extraction:
        required.append("freesurfer_synthstrip")
    candidate_sources = verify(sorted(set(required))) if required else []
    datasets = {split: OriginalDataset(args.data / f"adhd_{split}_sample.npy") for split in ("train", "val", "test")}
    ids = {split: set(str(v) for v in dataset.array["id"]) for split, dataset in datasets.items()}
    if any(len(ids[s]) != len(datasets[s]) for s in ids):
        raise ValueError("Duplicate subject ID within a split")
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise ValueError("Subject IDs overlap across splits")
    templates = ["template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy"]
    if any(not (args.data / name).is_file() for name in templates):
        raise ValueError("Required templates are missing")
    for name in templates:
        value = np.load(args.data / name, mmap_mode="r", allow_pickle=False)
        if value.shape != (96, 96, 96) or not np.isfinite(value).all():
            raise ValueError("Template must contain finite 96 cubed values: " + name)
    root = args.output.resolve()
    if any(root == p.resolve() or p.resolve() in root.parents for p in (args.source, args.data)):
        raise ValueError("Output must be separate from source and data")
    files = [args.data / f"adhd_{split}_sample.npy" for split in datasets] + [args.data / name for name in templates]
    data_hashes = {path.name: fingerprint(path) for path in files}
    plan_provenance = None
    if args.method == "nnunet":
        from .nnunet_prepare import validate_plan_provenance
        plan_provenance = validate_plan_provenance(args.plans, data_hashes["adhd_train_sample.npy"], args.dataset, args.fold)
    return {"status": "ready", "scope": "Static resource checks; not CUDA execution validation", "source": source, "candidate_sources": candidate_sources,
            "splits": {key: len(value) for key, value in datasets.items()},
            "data_hashes": data_hashes, "plan_provenance": plan_provenance}


def configure_trainable(model, method, setting, trained_part, standalone=False):
    candidate = METHODS[method]
    for name, parameter in model.named_parameters():
        is_new = candidate is not None and name.startswith(candidate + ".")
        enabled = is_new if standalone or trained_part == "new" else not is_new if trained_part == "host" or setting.startswith("independent") else True
        parameter.requires_grad_(enabled)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("No trainable parameters in this configuration")


def detached_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, list):
        return [detached_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detached_tree(v) for v in value)
    return value


def train_mode(model):
    model.train()
    for module in model.children():
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()


def configure_boundaries(model, method, setting, cut_extraction):
    handles = []
    if setting.startswith("independent"):
        def detach_candidate(module, inputs, output):
            for key in ("forward_grid", "inverse_grid", "velocity"):
                if hasattr(module, key):
                    setattr(module, key, detached_tree(getattr(module, key)))
            return detached_tree(output)
        handles.append(getattr(model, METHODS[method]).register_forward_hook(detach_candidate))
    if cut_extraction:
        handles.append(model.ext_net.register_forward_hook(lambda module, inputs, output: (detached_tree(output[0]), output[1])))
    return handles


def metrics_from_cases(cases):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    labels, predictions, scores = ([row[key] for row in cases] for key in ("label", "prediction", "probability"))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {"accuracy": float(accuracy_score(labels, predictions)), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
              "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
              "auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
              "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
              "specificity": float(tn / (tn + fp)) if tn + fp else None}
    for key in ("brain_dice", "tissue_dice", "atlas_dice", "registration_dice", "reference_loss"):
        result[key] = float(np.mean([row[key] for row in cases]))
    return result


def original_losses(output, brain, label):
    from torch.nn import functional as F
    tissue = output[5] if isinstance(output[5], list) else [output[5]]
    target = output[6] if output[6].shape[1] == 4 else F.one_hot(output[6][:, 0].long(), num_classes=4).movedim(-1, 1)
    factors = [2. ** -index for index in range(len(tissue))]
    segmentation = sum(weight * F.cross_entropy(logits, F.interpolate(target.float(), size=logits.shape[2:], mode="nearest"))
                       for weight, logits in zip(factors, tissue)) / sum(factors)
    return {"extraction": F.binary_cross_entropy(output[1][-1], brain),
            "segmentation": segmentation,
            "classification": F.cross_entropy(output[-1], label)}


def sample_tensors(row):
    import torch
    image, brain, tissue, atlas = [torch.tensor(row[i], dtype=torch.float32, device="cuda").reshape(1, 1, 96, 96, 96) for i in range(2, 6)]
    return image, brain, tissue, atlas, torch.tensor([int(row[1])], device="cuda")


def evaluate(model, dataset, templates, limit=None, output_dir=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from model import GCC
    from utils import compute_label_dice_v2, f1_loss
    model.eval()
    cases = []
    with torch.no_grad():
        for index in range(min(len(dataset), limit or len(dataset))):
            row = dataset[index]
            image, brain, tissue, atlas, label = sample_tensors(row)
            output = model(templates[0], image, templates[1], templates[2], if_train=False)
            mask = torch.prod(torch.cat(output[1]), 0).unsqueeze(0)
            grid = getattr(model, "last_forward_grid", None)
            if grid is None:
                grid = F.affine_grid(output[3][-1], tissue.shape, align_corners=True)
            registered = F.grid_sample(tissue, grid, mode="nearest", align_corners=True)
            losses = original_losses(output, brain, label)
            tissue_logits = output[5][0] if isinstance(output[5], list) else output[5]
            reference = losses["extraction"] + losses["segmentation"] + losses["classification"] + .1 * GCC()(output[2][-1], templates[0])
            case = {"id": str(row[0]), "label": int(label.item()), "prediction": int(output[-1].argmax(1).item()),
                    "probability": float(output[-1].softmax(1)[0, 1]), "reference_loss": float(reference),
                    "brain_dice": float(f1_loss(mask, brain)), "tissue_dice": float(compute_label_dice_v2(tissue_logits.argmax(1), tissue, "gm")),
                    "atlas_dice": float(compute_label_dice_v2(output[8], atlas, "aal")),
                    "registration_dice": float(compute_label_dice_v2(templates[1], registered, "gm"))}
            if any(not torch.isfinite(torch.tensor(v)) for k, v in case.items() if k != "id"):
                raise ValueError("Nonfinite evaluation metrics")
            if output_dir is not None:
                destination = Path(output_dir) / f"case_{index:05d}"
                destination.mkdir(parents=True, exist_ok=True)
                for name, prediction, reference in (("brain", mask[0, 0] >= .5, brain[0, 0]),
                                                     ("tissue", tissue_logits.argmax(1)[0], tissue[0, 0]),
                                                     ("atlas", output[8][0, 0], atlas[0, 0])):
                    np.save(destination / f"{name}_prediction.npy", prediction.cpu().numpy())
                    np.save(destination / f"{name}_reference.npy", reference.cpu().numpy())
                case["prediction_directory"] = str(destination.resolve())
            cases.append(case)
    return {"sample_count": len(cases), "full_split_count": len(dataset), "metrics": metrics_from_cases(cases), "cases": cases}


def rng_state():
    import numpy as np
    import torch
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value["python"])
    numpy = value["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def train(args):
    import numpy as np
    import torch
    from .data import OriginalDataset
    from .full_model import build_full_model
    from .full_replacement_worker import available_gpu, selected_device

    audit = preflight(args)
    if args.preflight:
        return audit
    device = selected_device(args.gpu)
    gpu = available_gpu(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; use a new directory or an explicit resume checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity = {k: settings[k] for k in ("method", "setting", "initialization", "seed", "epochs", "pretrain_epochs", "trainable", "cut_extraction", "loss_scale", "regularization", "dataset", "fold", "weights_origin", "connect_paths", "cut_boundary", "repair_strategy", "undo_repair", "patience")}
    identity.update(data_hashes=audit["data_hashes"], adapter_hash=fingerprint(args.adapter) if args.adapter else None,
                    weights_hash=fingerprint(args.weights) if args.weights else None, plans_hash=fingerprint(args.plans) if args.plans else None,
                    host_hash=fingerprint(args.host_checkpoint) if args.host_checkpoint else None,
                    plans_provenance_hash=audit["plan_provenance"]["sidecar_sha256"] if audit["plan_provenance"] else None)
    identity["composition"] = {"mode": args.composition_mode, **{key: fingerprint(getattr(args, key)) if getattr(args, key) else None
                                                                   for key in ("secondary_extraction", "previous_checkpoint", "previous_report", "extraction_weights")}}
    identity["implementation_hashes"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                          for name in ("replacement_training.py", "full_model.py", "full_candidates.py", "full_connectivity.py", "full_composition.py")}
    identity["source"] = audit["source"]
    if args.secondary_extraction:
        from .full_composition import build_composed_model
        model, config, architecture = build_composed_model(args.source, args.method, adapter=args.secondary_extraction,
                                                          mode=args.composition_mode, registration_weights=args.weights,
                                                          extraction_weights=args.extraction_weights, previous_checkpoint=args.previous_checkpoint,
                                                          previous_report=args.previous_report, previous_sha256=args.previous_sha256,
                                                          dataset=args.dataset, fold=args.fold, host_checkpoint=args.host_checkpoint)
    elif args.method in ("original", "synthstrip"):
        model, config, architecture = build_full_model(args.source, args.adapter if args.method == "synthstrip" else None, host_checkpoint=args.host_checkpoint)
        if args.weights:
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            model.ext_net.core.load_state_dict(state.get("model_state_dict", state), strict=True)
    else:
        from .full_candidates import build_candidate_model
        model, config, architecture = build_candidate_model(args.source, args.method, adapter=args.adapter, weights=args.weights, plans=args.plans, host_checkpoint=args.host_checkpoint)
    if args.connect_paths or args.setting == "original_completed":
        from .full_connectivity import connect_model
        model = connect_model(model, cut=args.cut_boundary, repair=args.repair_strategy, undo=args.undo_repair)
    model.cuda()
    from model import GCC
    from utils import train_aug
    datasets = {split: OriginalDataset(args.data / f"adhd_{split}_sample.npy") for split in ("train", "val", "test")}
    templates = [torch.from_numpy(np.load(args.data / name)).float().cuda().reshape(1, 1, 96, 96, 96) for name in
                 ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")]
    phases = [("standalone", args.pretrain_epochs), ("pipeline", args.epochs)] if args.setting == "independent_finetuned" else [("pipeline", args.epochs)]
    progress = {"phase": 0, "epoch": 0, "position": 0, "steps": 0, "pretrain_steps": 0, "best_loss": float("inf"), "bad_epochs": 0}
    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resumed:
        if resumed["identity"] != identity:
            raise ValueError("Resume configuration, sources or data differ from the checkpoint")
        model.load_state_dict(resumed["model"], strict=True)
        progress = resumed["progress"]
        restore_rng(resumed["rng"])
    report = {"status": "running", "method": args.method, "setting": args.setting, "seed": args.seed, "config": settings,
              "architecture": architecture, "audit": audit, "gpu": gpu, "history": [], "steps": progress["steps"], "pretrain_steps": progress["pretrain_steps"]}
    report["trainer_scope"] = "Matched full-model training loop; original main.py reproduction remains a separate baseline"
    report["auxiliary_loss"] = "Normalized inverse powers of two over tissue output scales" if args.method == "nnunet" else None
    previous_seconds = 0.
    if args.resume and (args.output / "report.json").exists():
        previous = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        report["history"] = previous.get("history", [])
        if "best_validation" in previous:
            report["best_validation"] = previous["best_validation"]
        previous_seconds = previous.get("train_seconds", 0.)
        with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "resume", "checkpoint_step": progress["steps"],
                                     "note": "Earlier attempted updates are retained; this run resumes checkpoint state"}) + "\n")
    started = time.monotonic()
    handles = []
    try:
        for phase_index in range(progress["phase"], len(phases)):
            phase, epochs = phases[phase_index]
            standalone = phase == "standalone"
            configure_trainable(model, args.method, args.setting, args.trainable, standalone)
            for handle in handles:
                handle.remove()
            handles = [] if standalone else configure_boundaries(model, args.method, args.setting, args.cut_extraction)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            if resumed and phase_index == progress["phase"]:
                optimizer.load_state_dict(resumed["optimizer"])
                restore_rng(resumed["rng"])
                resumed = None

            def checkpoint(path):
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(), "progress": dict(progress), "identity": identity}
                temporary = path.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(path)

            for epoch in range(progress["epoch"], epochs):
                train_mode(model)
                order = torch.randperm(len(datasets["train"]), generator=torch.Generator().manual_seed(args.seed + 10000 * phase_index + epoch)).tolist()
                for position in range(progress["position"], len(order)):
                    if args.max_steps is not None and progress["steps"] >= args.max_steps:
                        checkpoint(args.output / "last.pt")
                        report.update(status="budget_exhausted", **{k: progress[k] for k in ("steps", "pretrain_steps")})
                        return report
                    image, brain, _, _, label = sample_tensors(datasets["train"][order[position]])
                    if config["if_train_aug"]:
                        image, brain = train_aug(image, brain, 96, 2, 2, .98, 1.02)
                    optimizer.zero_grad(set_to_none=True)
                    if standalone and args.method == "synthstrip":
                        output = model.ext_net(image, True)
                        losses = {"extraction": torch.nn.functional.binary_cross_entropy(output[1][-1], brain)}
                        total = losses["extraction"]
                    else:
                        output = model(templates[0], image, templates[1], templates[2], if_train=True)
                        losses = original_losses(output, brain, label)
                        losses["registration"] = GCC()(output[2][-1], templates[0])
                        new_loss = {"synthstrip": "extraction", "voxelmorph": "registration", "lapirn": "registration", "nnunet": "segmentation"}.get(args.method)
                        weights = {"extraction": config["lambda_ext"], "registration": config["lambda_reg"], "segmentation": config["lambda_seg"], "classification": config["lambda_cls"]}
                        if new_loss:
                            weights[new_loss] *= args.loss_scale
                        total = losses[new_loss] if standalone else sum(weights[key] * value for key, value in losses.items())
                        if hasattr(model, "regularization_loss") and args.regularization:
                            total = total + args.regularization * model.regularization_loss()
                    if not torch.isfinite(total) or not total.requires_grad:
                        raise ValueError("Joint objective is nonfinite or disconnected")
                    total.backward()
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise ValueError("Nonfinite parameter gradient")
                    probes = {}
                    for name, parameter in model.named_parameters():
                        module = name.split(".")[0]
                        if module not in probes and parameter.grad is not None and torch.count_nonzero(parameter.grad):
                            probes[module] = (parameter, parameter.detach().cpu().clone(), name)
                    optimizer.step()
                    observed = {module: {"tensor": name, "changed": not torch.equal(parameter.detach().cpu(), before)}
                                for module, (parameter, before, name) in probes.items()}
                    if not observed or not any(item["changed"] for item in observed.values()):
                        raise ValueError("No sampled trainable parameter changed after the optimizer step")
                    progress.update(phase=phase_index, epoch=epoch, position=position + 1, steps=progress["steps"] + 1,
                                    pretrain_steps=progress["pretrain_steps"] + int(standalone))
                    event = {"phase": phase, "epoch": epoch + 1, "step": progress["steps"], "losses": {k: float(v.detach()) for k, v in losses.items()}, "total_loss": float(total.detach()),
                             "sampled_parameter_updates": observed}
                    with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event) + "\n")
                    del output, losses, total, probes
                progress.update(epoch=epoch + 1, position=0)
                if not standalone:
                    validation = evaluate(model, datasets["val"], templates, args.eval_limit)
                    loss = validation["metrics"]["reference_loss"]
                    report["history"].append({"epoch": epoch + 1, "validation": validation["metrics"]})
                    if loss < progress["best_loss"]:
                        progress["best_loss"] = loss
                        progress["bad_epochs"] = 0
                        checkpoint(args.output / "best.pt")
                        report["best_validation"] = validation["metrics"]
                    else:
                        progress["bad_epochs"] += 1
                stopped = not standalone and args.patience and progress["bad_epochs"] >= args.patience
                if stopped:
                    progress.update(phase=len(phases), epoch=0, position=0)
                    report["stop_reason"] = "validation_patience"
                checkpoint(args.output / "last.pt")
                save_json(args.output / "report.json", {**report, "steps": progress["steps"], "pretrain_steps": progress["pretrain_steps"]})
                if stopped:
                    break
            progress.update(phase=phase_index + 1, epoch=0, position=0)
            if standalone:
                candidate = getattr(model, METHODS[args.method])
                core = getattr(candidate, "core", candidate)
                torch.save({"model_state_dict": core.state_dict()}, args.output / "candidate_core.pt")
                save_json(args.output / "candidate_core.json", {"sha256": fingerprint(args.output / "candidate_core.pt"),
                                                                "dataset": args.dataset, "fold": args.fold, "scope": "training_fold",
                                                                "steps": progress["pretrain_steps"], "identity": identity})
        report["status"] = "complete"
        if args.evaluate_test:
            best = args.output / "best.pt"
            if not best.exists():
                raise ValueError("No validation selected checkpoint is available for test")
            model.load_state_dict(torch.load(best, map_location="cpu", weights_only=True)["model"], strict=True)
            report["test"] = evaluate(model, datasets["test"], templates, args.eval_limit, args.output / "predictions" if args.save_predictions else None)
            report["test"]["checkpoint_sha256"] = fingerprint(best)
            save_json(args.output / "test.json", report["test"])
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(steps=progress["steps"], pretrain_steps=progress["pretrain_steps"], train_seconds=previous_seconds + time.monotonic() - started,
                      checkpoints=[p.name for p in args.output.glob("*.pt")])
        save_json(args.output / "report.json", report)
        for handle in handles:
            handle.remove()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("adapter", "weights", "plans", "resume", "host-checkpoint"):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--weights-origin")
    result.add_argument("--weights-sha256")
    result.add_argument("--host-origin")
    result.add_argument("--host-sha256")
    result.add_argument("--host-dataset")
    result.add_argument("--host-fold", type=int)
    result.add_argument("--secondary-extraction", type=Path)
    result.add_argument("--composition-mode", choices=("simultaneous", "sequential"), default="simultaneous")
    result.add_argument("--previous-checkpoint", type=Path)
    result.add_argument("--previous-report", type=Path)
    result.add_argument("--previous-sha256")
    result.add_argument("--extraction-weights", type=Path)
    result.add_argument("--extraction-weights-origin")
    result.add_argument("--extraction-weights-sha256")
    result.add_argument("--method", choices=METHODS, default="synthstrip")
    result.add_argument("--setting", choices=SETTINGS, default="joint")
    result.add_argument("--initialization", choices=("random", "official", "training_fold"), default="random")
    result.add_argument("--dataset")
    result.add_argument("--fold", type=int)
    result.add_argument("--weights-dataset")
    result.add_argument("--weights-fold", type=int)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--pretrain-epochs", type=int, default=20)
    result.add_argument("--max-steps", type=int)
    result.add_argument("--eval-limit", type=int)
    result.add_argument("--patience", type=int, default=0)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--trainable", choices=("all", "new", "host"), default="all")
    result.add_argument("--cut-extraction", action="store_true")
    result.add_argument("--connect-paths", action="store_true")
    result.add_argument("--cut-boundary", choices=("none", "extraction", "registration", "parcellation", "all"), default="none")
    result.add_argument("--repair-strategy", choices=("soft", "straight_through", "gumbel_softmax"), default="soft")
    result.add_argument("--undo-repair", choices=("none", "nearest", "graph", "mask"), default="none")
    result.add_argument("--loss-scale", type=float, default=1.)
    result.add_argument("--regularization", type=float, default=0.)
    result.add_argument("--evaluate-test", action="store_true")
    result.add_argument("--save-predictions", action="store_true")
    result.add_argument("--preflight", action="store_true")
    return result
