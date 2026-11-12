"""Reproducible full model comparisons with separate validation and test."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path

METHODS = {"original": None, "synthstrip": "ext_net", "voxelmorph": "reg_net", "lapirn": "reg_net", "nnunet": "seg_net"}
from .full_extended import TARGETS, REPOSITORIES
METHODS.update(TARGETS)

SETTINGS = ("original", "original_completed", "original_frozen_stages", "joint", "independent_official", "independent_pretrained", "independent_finetuned")
LEGACY_REFERENCE_TRAINER = "4977739c34f4b8cbb7326052e23a9cc77199a40297db8e28bfe95bee2bd60904"


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


def host_checkpoint_spec(args):
    from .full_model import VerifiedHostCheckpoint
    return (VerifiedHostCheckpoint(args.host_checkpoint, args.dataset, args.fold, args.host_sha256, args.host_report)
            if args.host_checkpoint else None)


def resume_identity_matches(previous, current):
    if previous == current:
        return False
    migrated = copy.deepcopy(previous)
    hashes = migrated.get("implementation_hashes", {})
    if ("selection_metric" not in previous and current.get("selection_metric") == "reference_loss"
            and hashes.get("replacement_training.py") == LEGACY_REFERENCE_TRAINER):
        migrated["selection_metric"] = "reference_loss"
        hashes["replacement_training.py"] = current.get("implementation_hashes", {}).get("replacement_training.py")
        if migrated == current:
            return True
    raise ValueError("Resume configuration, sources or data differ from the checkpoint; legacy selection requires explicit reference_loss and a verified compatible trainer")


def selection_value(metrics, name):
    value = metrics.get(name)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Validation selection metric must be present and finite: " + name)
    return value


def finalize_pipeline(model, architecture, args):
    if args.connect_paths or args.setting == "original_completed":
        from .full_connectivity import connect_model
        model = connect_model(model, cut=args.cut_boundary, repair=args.repair_strategy, undo=args.undo_repair)
    if args.adapter is not None and args.method not in ("original", "synthstrip"):
        from .candidate_integration import bind_adapter, bind_extended_adapters
        if args.method in TARGETS:
            bind_extended_adapters(model, args.method, args.adapter)
        else:
            bind_adapter(model, args.method, args.adapter)
        architecture["generated_adapter_sha256"] = fingerprint(args.adapter)
        architecture["generated_adapter_active_in_training"] = True
    return model


def trained_host_evidence(args):
    from .full_model import load_host_checkpoint
    _, evidence = load_host_checkpoint(host_checkpoint_spec(args))
    return evidence


def preflight(args):
    import numpy as np
    from .native import verify_original
    from .data import OriginalDataset
    from .full_replacement import check_code
    from .sources import verify
    from .site_training import validate_site_options

    validate_site_options(args)

    if args.method in TARGETS and args.setting != "joint":
        raise ValueError("Extended candidates currently support joint training only")
    if args.method in TARGETS and args.initialization != "random":
        raise ValueError("Extended candidate pretrained wrapper checkpoints require separate round trip validation; use explicit random initialization")
    if args.epochs < 1 or args.pretrain_epochs < 0 or args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Epoch and update budgets must be positive")
    if args.loss_scale <= 0 or args.regularization < 0:
        raise ValueError("Loss scales must be positive and regularization nonnegative")
    if (args.setting in ("original", "original_completed", "original_frozen_stages")) != (args.method == "original"):
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
    if args.setting in ("original", "original_frozen_stages") and (args.connect_paths or args.cut_boundary != "none" or args.undo_repair != "none" or args.repair_strategy != "soft"):
        raise ValueError("Use original_completed for connection changes to an unchanged module set")
    if not (args.connect_paths or args.setting == "original_completed") and (args.cut_boundary != "none" or args.undo_repair != "none" or args.repair_strategy != "soft"):
        raise ValueError("Connection ablations require --connect-paths")
    if args.setting.startswith("independent") and args.trainable != "all":
        raise ValueError("Independent settings train the host after freezing the candidate")
    if args.setting == "original_frozen_stages" and (not args.host_checkpoint or args.coordination != "none"):
        raise ValueError("Frozen original stages require a trained host checkpoint and coordination none")
    if args.probe_budget < 1 or args.coordination_batch_size < 1:
        raise ValueError("Coordinator replay and probe budgets must be positive")
    if args.probe_modules is not None and args.coordination == "none":
        raise ValueError("Explicit gradient probes require taskforce coordination")
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
    if args.method == "synthstrip":
        if args.adapter is None:
            raise ValueError("SynthStrip requires a validated generated adapter")
        check_code(args.adapter.read_text(encoding="utf-8"))
    elif args.adapter is not None:
        from .candidate_integration import check_adapter
        check_adapter(args.adapter.read_text(encoding='utf-8'), args.method)
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
    if args.resume and args.checkpoint_mode != "resume":
        raise ValueError("Weights only checkpoints cannot resume training; select checkpoint mode resume")
    for name in ("adapter", "weights", "plans", "resume", "host_checkpoint", "host_report", "secondary_extraction", "previous_checkpoint", "previous_report", "extraction_weights"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            raise ValueError("Missing file: " + name)
    if args.weights and (not args.weights_origin or not args.weights_sha256):
        raise ValueError("Official weights require origin and expected SHA256")
    if args.weights and fingerprint(args.weights) != args.weights_sha256:
        raise ValueError("Official checkpoint hash mismatch")
    if args.extraction_weights and (not args.extraction_weights_origin or not args.extraction_weights_sha256 or fingerprint(args.extraction_weights) != args.extraction_weights_sha256):
        raise ValueError("Composition extraction weights require origin and matching SHA256")
    if args.host_report and not args.host_checkpoint:
        raise ValueError("A host training report requires its checkpoint")
    host_evidence = trained_host_evidence(args) if args.host_checkpoint else None
    source = verify_original(args.source)
    repositories = {"synthstrip": "freesurfer_synthstrip", "voxelmorph": "reg_voxelmorph_master", "lapirn": "reg_lapirn", "nnunet": "dynamic-network-architectures"}
    repositories.update(REPOSITORIES)
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
    from .cohort_provenance import validate_cohort, validate_site
    cohort = (validate_site(args.data, args.dataset, args.split_id, data_hashes) if args.split_id else
              validate_cohort(args.data, args.dataset, args.fold, data_hashes))
    plan_provenance = None
    if args.method == "nnunet":
        from .nnunet_prepare import validate_plan_provenance
        plan_provenance = validate_plan_provenance(args.plans, data_hashes["adhd_train_sample.npy"], args.dataset, args.fold)
    return {"status": "ready", "scope": "Static resource checks; not CUDA execution validation", "source": source, "candidate_sources": candidate_sources,
            "splits": {key: len(value) for key, value in datasets.items()},
            "data_hashes": data_hashes, "plan_provenance": plan_provenance, "host_training_evidence": host_evidence, "cohort": cohort}


def configure_trainable(model, method, setting, trained_part, standalone=False):
    if not hasattr(model, '_native_frozen_names'):
        model._native_frozen_names = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    candidate = METHODS[method]
    names = getattr(model, "candidate_modules", ()) or ((candidate,) if candidate else ())
    for name, parameter in model.named_parameters():
        is_new = any(name.startswith(prefix + ".") for prefix in names)
        enabled = is_new if standalone or trained_part == "new" else not is_new if trained_part == "host" or setting.startswith("independent") else True
        if setting == "original_frozen_stages":
            enabled = name.startswith("gnn_pred.")
        parameter.requires_grad_(enabled and name not in model._native_frozen_names)
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
    def detach_candidate(module, inputs, output):
        for key in ("forward_grid", "inverse_grid", "velocity"):
            if hasattr(module, key):
                setattr(module, key, detached_tree(getattr(module, key)))
        return detached_tree(output)
    if setting.startswith("independent"):
        handles.append(getattr(model, METHODS[method]).register_forward_hook(detach_candidate))
    if setting == "original_frozen_stages":
        handles.extend(module.register_forward_hook(detach_candidate) for name, module in model.named_children() if name != "gnn_pred")
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
    for key in ("extraction_loss", "registration_loss", "segmentation_loss", "classification_loss"):
        if all(key in row for row in cases):
            result[key] = float(np.mean([row[key] for row in cases]))
        elif any(key in row for row in cases):
            raise ValueError("Incomplete validation component measurements: " + key)
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


def atlas_class_indices(target, template):
    import torch
    if not torch.isfinite(target).all() or not torch.equal(target, target.round()):
        raise ValueError("AAL targets must contain finite integer labels")
    vocabulary = torch.unique(template, sorted=True)
    if len(vocabulary) != 117 or not torch.equal(vocabulary, vocabulary.round()):
        raise ValueError("The adaptive AAL head requires exactly 117 template labels")
    indices = torch.searchsorted(vocabulary, target.contiguous())
    valid = (indices < len(vocabulary)) & (vocabulary[indices.clamp_max(len(vocabulary) - 1)] == target)
    if not valid.all():
        raise ValueError("Subject AAL target contains labels outside the template vocabulary")
    return indices.squeeze(1)


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
            registration_loss = GCC()(output[2][-1], templates[0])
            reference = losses["extraction"] + losses["segmentation"] + losses["classification"] + .1 * registration_loss
            case = {"id": str(row[0]), "label": int(label.item()), "prediction": int(output[-1].argmax(1).item()),
                    "probability": float(output[-1].softmax(1)[0, 1]), "reference_loss": float(reference),
                    "brain_dice": float(f1_loss(mask, brain)), "tissue_dice": float(compute_label_dice_v2(tissue_logits.argmax(1), tissue, "gm")),
                    "atlas_dice": float(compute_label_dice_v2(output[8], atlas, "aal")),
                    "registration_dice": float(compute_label_dice_v2(templates[1], registered, "gm"))}
            case.update({name + "_loss": float(loss) for name, loss in losses.items()})
            case["registration_loss"] = float(registration_loss)
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


def task_weights(config, method, scale):
    weights = {"extraction": config["lambda_ext"], "registration": config["lambda_reg"],
               "segmentation": config["lambda_seg"], "classification": config["lambda_cls"]}
    task = {"synthstrip": "extraction", "voxelmorph": "registration", "lapirn": "registration", "nnunet": "segmentation"}.get(method)
    if task:
        weights[task] *= scale
    return weights


def checkpoint_state(model, optimizer, progress, identity, team=None, evaluation_only=False, weights_only=False):
    state = {"model": model.state_dict(), "progress": dict(progress), "identity": identity,
             "checkpoint_kind": "evaluation_weights_only" if weights_only else "evaluation" if evaluation_only else "resume"}
    if not (evaluation_only or weights_only):
        state.update(optimizer=optimizer.state_dict(), rng=rng_state(), coordination=team.state_dict() if team else None)
    return state


def check_checkpoint_space(folder, state):
    import torch
    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(item) for item in value)
        return 0
    required = int(1.1 * tensor_bytes(state)) + 64 * 1024 ** 2
    available = shutil.disk_usage(folder).free
    if available < required + 1024 ** 3:
        raise OSError(f"Checkpoint requires approximately {required / 1024 ** 3:.2f} GiB plus a 1 GiB reserve; {available / 1024 ** 3:.2f} GiB free. Existing results were retained.")


def write_checkpoint(path, state, previous=None):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    weights = state["checkpoint_kind"] == "evaluation_weights_only"
    same = (weights and previous and previous["path"] != path and previous["path"].is_file()
            and previous["progress"] == state["progress"] and previous["identity"] == state["identity"])
    if same:
        try:
            os.link(previous["path"], temporary)
        except OSError:
            pass
        else:
            temporary.replace(path)
            return "hardlink"
    check_checkpoint_space(path.parent, state)
    torch.save(state, temporary)
    temporary.replace(path)
    return "serialized"


def gradient_statistics(gradients):
    import torch
    gradients = list(gradients)
    present = [value.detach() for value in gradients if value is not None]
    finite = all(bool(torch.isfinite(value).all()) for value in present)
    count = sum(int(torch.count_nonzero(value)) for value in present)
    return {"tensor_count": len(gradients), "observed_tensors": len(present), "nonzero_elements": count,
            "finite": finite, "nonzero": count > 0, "l1": sum(float(value.abs().sum()) for value in present) if finite else None}


def classification_gradient_paths(model, loss, method, required, image=None):
    import torch
    names = getattr(model, "candidate_modules", ()) or ((METHODS[method],) if METHODS[method] else ())
    groups = {name: [parameter for parameter in getattr(model, name).parameters() if parameter.requires_grad] for name in names}
    parameters = [parameter for group in groups.values() for parameter in group]
    inputs = parameters + ([image] if image is not None and image.requires_grad else [])
    gradients = torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True) if inputs and loss.requires_grad else [None] * len(inputs)
    result, offset = {}, 0
    for name in names:
        count = len(groups[name])
        auxiliary = method == "nnunet" and name == "seg_net"
        result[name] = {**gradient_statistics(gradients[offset:offset + count]), "required": required and not auxiliary,
                        "expected_absent": auxiliary,
                        "reason": "The tissue prediction head is supervised by tissue loss; diagnosis does not consume its logits" if auxiliary else None}
        offset += count
    input_gradient = gradients[-1] if image is not None and image.requires_grad else None
    return {"candidates": result, "input": {**gradient_statistics([input_gradient]), "required": required and method != "nnunet"}}


def classification_gradients(model, loss, method, required):
    return classification_gradient_paths(model, loss, method, required)["candidates"]


def total_candidate_gradients(model, method):
    names = getattr(model, "candidate_modules", ()) or ((METHODS[method],) if METHODS[method] else ())
    return {name: gradient_statistics(parameter.grad for parameter in getattr(model, name).parameters() if parameter.requires_grad)
            for name in names}


def train(args):
    import numpy as np
    import torch
    from .data import OriginalDataset
    from .full_model import build_full_model
    from .full_replacement_worker import available_gpu, selected_device
    from .taskforce import GradientTeam, TASKS, select_probes

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
    identity = {k: settings[k] for k in ("method", "setting", "initialization", "seed", "epochs", "pretrain_epochs", "trainable", "cut_extraction", "loss_scale", "regularization", "dataset", "fold", "split_id", "weights_origin", "connect_paths", "cut_boundary", "repair_strategy", "undo_repair", "patience", "coordination", "coordination_batch_size", "probe_budget", "probe_modules", "verify_gradients", "checkpoint_mode")}
    identity["selection_metric"] = args.selection_metric
    identity.update(data_hashes=audit["data_hashes"], adapter_hash=fingerprint(args.adapter) if args.adapter else None,
                    weights_hash=fingerprint(args.weights) if args.weights else None, plans_hash=fingerprint(args.plans) if args.plans else None,
                    host_hash=fingerprint(args.host_checkpoint) if args.host_checkpoint else None,
                    host_report_hash=fingerprint(args.host_report) if args.host_report else None,
                    plans_provenance_hash=audit["plan_provenance"]["sidecar_sha256"] if audit["plan_provenance"] else None)
    identity["composition"] = {"mode": args.composition_mode, **{key: fingerprint(getattr(args, key)) if getattr(args, key) else None
                                                                   for key in ("secondary_extraction", "previous_checkpoint", "previous_report", "extraction_weights")}}
    identity["implementation_hashes"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                          for name in ("replacement_training.py", "full_model.py", "full_candidates.py", "full_connectivity.py", "full_composition.py", "full_extended.py", "full_preprocessing.py", "full_anatomy_candidates.py", "full_graph_candidates.py", "taskforce.py", "cohort_provenance.py", "candidate_integration.py", "full_replacement.py", "sources.py", "site_training.py")}
    identity["source"] = audit["source"]
    host_checkpoint = host_checkpoint_spec(args)
    if args.secondary_extraction:
        from .full_composition import build_composed_model
        model, config, architecture = build_composed_model(args.source, args.method, adapter=args.secondary_extraction,
                                                          mode=args.composition_mode, registration_weights=args.weights,
                                                          extraction_weights=args.extraction_weights, previous_checkpoint=args.previous_checkpoint,
                                                          previous_report=args.previous_report, previous_sha256=args.previous_sha256,
                                                          dataset=args.dataset, fold=args.fold, host_checkpoint=host_checkpoint)
    elif args.method in ("original", "synthstrip"):
        model, config, architecture = build_full_model(args.source, args.adapter if args.method == "synthstrip" else None, host_checkpoint=host_checkpoint)
        if args.method == "synthstrip":
            architecture.update(generated_adapter_active_in_training=True, generated_adapter_sha256=fingerprint(args.adapter))
        if args.weights:
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            model.ext_net.core.load_state_dict(state.get("model_state_dict", state), strict=True)
    elif args.method in TARGETS:
        from .full_extended import build_extended_model
        model, config, architecture = build_extended_model(args.source, args.method, host_checkpoint, args.weights)
    else:
        from .full_candidates import build_candidate_model
        model, config, architecture = build_candidate_model(args.source, args.method, adapter=args.adapter, weights=args.weights, plans=args.plans, host_checkpoint=host_checkpoint)
    model = finalize_pipeline(model, architecture, args)
    model.cuda()
    from model import GCC
    from utils import train_aug
    datasets = {split: OriginalDataset(args.data / f"adhd_{split}_sample.npy") for split in ("train", "val", "test")}
    templates = [torch.from_numpy(np.load(args.data / name)).float().cuda().reshape(1, 1, 96, 96, 96) for name in
                 ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")]
    phases = [("standalone", args.pretrain_epochs), ("pipeline", args.epochs)] if args.setting == "independent_finetuned" else [("pipeline", args.epochs)]
    base_weights = task_weights(config, args.method, args.loss_scale)
    team = GradientTeam(tuple(base_weights[name] for name in TASKS), seed=args.seed,
                        batch_size=args.coordination_batch_size) if args.coordination == "taskforce" else None
    progress = {"phase": 0, "epoch": 0, "position": 0, "steps": 0, "pretrain_steps": 0, "best_loss": float("inf"), "bad_epochs": 0,
                "selection_metric": args.selection_metric, "best_epoch": None, "best_value": None}
    migrated_selection = False
    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resumed:
        if resumed.get("checkpoint_kind") != "resume":
            raise ValueError("This is an evaluation checkpoint; resume requires the complete last.pt state")
        migrated_selection = resume_identity_matches(resumed["identity"], identity)
        model.load_state_dict(resumed["model"], strict=True)
        if team is not None:
            if resumed.get("coordination") is None:
                raise ValueError("The checkpoint lacks the requested gradient coordinator state")
            team.load_state_dict(resumed["coordination"])
        progress = resumed["progress"]
        if migrated_selection:
            progress.update(selection_metric=args.selection_metric, best_epoch=None, best_value=None)
        restore_rng(resumed["rng"])
    report = {"status": "running", "method": args.method, "setting": args.setting, "seed": args.seed, "config": settings,
              "architecture": architecture, "audit": audit, "gpu": gpu, "history": [], "steps": progress["steps"], "pretrain_steps": progress["pretrain_steps"]}
    report["trainer_scope"] = "Matched full-model training loop; original main.py reproduction remains a separate baseline"
    report["selection"] = {"metric": args.selection_metric, "direction": "minimize", "scope": "validation_only",
                           "best_epoch": progress.get("best_epoch"), "best_value": progress.get("best_value"),
                           "legacy_reference_resume": migrated_selection}
    report["checkpoint_policy"] = {"mode": args.checkpoint_mode, "resume_allowed": args.checkpoint_mode == "resume", "hardlinked_saves": 0}
    report["auxiliary_loss"] = "Normalized inverse powers of two over tissue output scales" if args.method == "nnunet" else None
    report["coordination"] = {"mode": args.coordination, "policy_updates": team.updates if team else 0,
                              "actor_optimizer_steps": team.count * team.updates if team else 0,
                              "critic_optimizer_steps": team.count * team.updates if team else 0,
                              "algorithm": "TaskForce inspired task actors and centralized critics; not an exact paper reproduction" if team else None,
                              "activation": "Pipeline training only; validation and test do not update or query the policy",
                              "base_weights": base_weights, "probe_names": [], "probe_elements": 0,
                              "fixed_extra_weights": {"parcellation": args.loss_scale if getattr(model, "learned_parcellation", False) else 0.,
                                                      "regularization": args.regularization}}
    report["gradient_verification"] = {"enabled": args.verify_gradients, "protocol_version": 2, "verified_steps": 0, "last_step": None,
                                       "diagnosis_reaches_all_candidates": None}
    if args.setting == "original_frozen_stages":
        report["stagewise_control"] = "Fixed trained UniBrain stages and trainable original classifier; not an ANTs or FSL toolchain"
    previous_seconds = 0.
    if migrated_selection and math.isfinite(progress["best_loss"]) and not (args.output / "report.json").is_file():
        raise ValueError("Legacy reference selection requires its retained validation report")
    if args.resume and (args.output / "report.json").exists():
        previous = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        report["history"] = previous.get("history", [])
        if "best_validation" in previous:
            report["best_validation"] = previous["best_validation"]
        previous_seconds = previous.get("train_seconds", 0.)
        if migrated_selection:
            if not report["history"] and math.isfinite(progress["best_loss"]):
                raise ValueError("Legacy checkpoint lacks its validation selection history")
            if report["history"]:
                selected = min(report["history"], key=lambda row: selection_value(row["validation"], args.selection_metric))
                value = selection_value(selected["validation"], args.selection_metric)
                if value != progress["best_loss"]:
                    raise ValueError("Legacy checkpoint selection disagrees with retained validation history")
                progress.update(best_epoch=selected["epoch"], best_value=value)
                report["selection"].update(best_epoch=selected["epoch"], best_value=value)
        with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "resume", "checkpoint_step": progress["steps"],
                                     "note": "Earlier attempted updates are retained; this run resumes checkpoint state"}) + "\n")
    started = time.monotonic()
    handles = []
    previous_weights = None
    try:
        for phase_index in range(progress["phase"], len(phases)):
            phase, epochs = phases[phase_index]
            standalone = phase == "standalone"
            configure_trainable(model, args.method, args.setting, args.trainable, standalone)
            gradient_probes = []
            if team is not None and not standalone:
                gradient_probes, names = select_probes(model, args.probe_budget, args.probe_modules)
                report["coordination"].update(probe_names=names, probe_elements=sum(parameter.numel() for parameter in gradient_probes))
            for handle in handles:
                handle.remove()
            handles = [] if standalone else configure_boundaries(model, args.method, args.setting, args.cut_extraction)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            if resumed and phase_index == progress["phase"]:
                optimizer.load_state_dict(resumed["optimizer"])
                restore_rng(resumed["rng"])
                resumed = None

            def checkpoint(path):
                nonlocal previous_weights
                state = checkpoint_state(model, optimizer, progress, identity, team, evaluation_only=path.name == "best.pt",
                                         weights_only=args.checkpoint_mode == "weights")
                result = write_checkpoint(path, state, previous_weights)
                report["checkpoint_policy"]["hardlinked_saves"] += int(result == "hardlink")
                if args.checkpoint_mode == "weights":
                    previous_weights = {"path": path, "progress": dict(progress), "identity": identity}

            for epoch in range(progress["epoch"], epochs):
                train_mode(model)
                order = torch.randperm(len(datasets["train"]), generator=torch.Generator().manual_seed(args.seed + 10000 * phase_index + epoch)).tolist()
                for position in range(progress["position"], len(order)):
                    if args.max_steps is not None and progress["steps"] >= args.max_steps:
                        checkpoint(args.output / "last.pt")
                        report.update(status="budget_exhausted", **{k: progress[k] for k in ("steps", "pretrain_steps")})
                        return report
                    image, brain, _, atlas_target, label = sample_tensors(datasets["train"][order[position]])
                    if config["if_train_aug"] and not getattr(model, "learned_parcellation", False):
                        image, brain = train_aug(image, brain, 96, 2, 2, .98, 1.02)
                    if args.verify_gradients:
                        image = image.detach().requires_grad_(True)
                    optimizer.zero_grad(set_to_none=True)
                    policy_record = None
                    if standalone and args.method == "synthstrip":
                        output = model.ext_net(image, True)
                        losses = {"extraction": torch.nn.functional.binary_cross_entropy(output[1][-1], brain)}
                        total = losses["extraction"]
                    else:
                        output = model(templates[0], image, templates[1], templates[2], if_train=True)
                        losses = original_losses(output, brain, label)
                        losses["registration"] = GCC()(output[2][-1], templates[0])
                        new_loss = {"synthstrip": "extraction", "voxelmorph": "registration", "lapirn": "registration", "nnunet": "segmentation"}.get(args.method)
                        weights = dict(base_weights)
                        if getattr(model, "learned_parcellation", False):
                            atlas_indices = atlas_class_indices(atlas_target, templates[2])
                            losses["parcellation"] = torch.nn.functional.cross_entropy(output[7], atlas_indices)
                            weights["parcellation"] = args.loss_scale
                        if standalone:
                            total = losses[new_loss]
                        elif args.setting == "original_frozen_stages":
                            total = weights["classification"] * losses["classification"]
                        elif team is not None:
                            total, policy_record = team.combine(losses, gradient_probes)
                            if "parcellation" in losses:
                                total = total + weights["parcellation"] * losses["parcellation"]
                        else:
                            total = sum(weights[key] * value for key, value in losses.items())
                        if hasattr(model, "regularization_loss") and args.regularization:
                            total = total + args.regularization * model.regularization_loss()
                    if not torch.isfinite(total) or not total.requires_grad:
                        raise ValueError("Joint objective is nonfinite or disconnected")
                    gradient_record = None
                    if args.verify_gradients:
                        required = (not standalone and args.setting == "joint" and args.trainable in ("all", "new")
                                    and (args.connect_paths or args.method in TARGETS) and not args.cut_extraction
                                    and args.cut_boundary == "none" and args.undo_repair == "none")
                        paths = classification_gradient_paths(model, losses["classification"], args.method, required, image) if "classification" in losses else None
                        gradient_record = {"protocol_version": 2, "classification_to_candidates": paths["candidates"] if paths else {},
                                           "classification_to_input": paths["input"] if paths else None, "input_required": required}
                    total.backward()
                    if gradient_record is not None:
                        gradient_record["total_to_input"] = gradient_statistics([image.grad])
                        gradient_record["total_to_candidates"] = total_candidate_gradients(model, args.method)
                        report["gradient_verification"]["last_step"] = gradient_record
                        failed = [name for name, value in gradient_record["classification_to_candidates"].items()
                                  if value["required"] and not (value["finite"] and value["nonzero"])]
                        input_path = gradient_record["classification_to_input"]
                        if input_path and input_path["required"] and not (input_path["finite"] and input_path["nonzero"]):
                            failed.append("classification_to_input")
                        if required and not (gradient_record["total_to_input"]["finite"] and gradient_record["total_to_input"]["nonzero"]):
                            failed.append("input")
                        if required:
                            failed.extend(name + ":total" for name, value in gradient_record["total_to_candidates"].items()
                                          if not (value["finite"] and value["nonzero"]))
                        if failed:
                            raise ValueError("Required gradient paths failed: " + ", ".join(failed))
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
                    if args.method in TARGETS and args.trainable != 'host':
                        missing = [name for name in model.candidate_modules if not observed.get(name, {}).get('changed')]
                        if missing:
                            raise ValueError('No sampled parameter update in selected candidate modules: ' + ', '.join(missing))
                    progress.update(phase=phase_index, epoch=epoch, position=position + 1, steps=progress["steps"] + 1,
                                    pretrain_steps=progress["pretrain_steps"] + int(standalone))
                    event = {"phase": phase, "epoch": epoch + 1, "step": progress["steps"], "losses": {k: float(v.detach()) for k, v in losses.items()}, "total_loss": float(total.detach()),
                             "sampled_parameter_updates": observed}
                    if policy_record is not None:
                        event["gradient_policy"] = policy_record
                    if gradient_record is not None:
                        event["gradient_verification"] = gradient_record
                        report["gradient_verification"]["verified_steps"] += 1
                        paths = gradient_record["classification_to_candidates"].values()
                        report["gradient_verification"]["diagnosis_reaches_all_candidates"] = all(value["finite"] and value["nonzero"] for value in paths) if gradient_record["classification_to_candidates"] else None
                    with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event) + "\n")
                    del output, losses, total, probes
                progress.update(epoch=epoch + 1, position=0)
                if not standalone:
                    validation = evaluate(model, datasets["val"], templates, args.eval_limit)
                    loss = selection_value(validation["metrics"], args.selection_metric)
                    report["history"].append({"epoch": epoch + 1, "validation": validation["metrics"]})
                    if loss < progress["best_loss"]:
                        progress.update(best_loss=loss, best_value=loss, best_epoch=epoch + 1, selection_metric=args.selection_metric)
                        report["selection"].update(best_epoch=epoch + 1, best_value=loss)
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
        if team is not None:
            report["coordination"].update(policy_updates=team.updates, actor_optimizer_steps=team.count * team.updates,
                                          critic_optimizer_steps=team.count * team.updates, last_decision=team.last_record)
        report.update(steps=progress["steps"], pretrain_steps=progress["pretrain_steps"], train_seconds=previous_seconds + time.monotonic() - started,
                      checkpoints=[p.name for p in args.output.glob("*.pt")])
        save_json(args.output / "report.json", report)
        for handle in handles:
            handle.remove()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("adapter", "weights", "plans", "resume", "host-checkpoint", "host-report"):
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
    result.add_argument("--split-id")
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
    result.add_argument("--coordination", choices=("none", "taskforce"), default="none")
    result.add_argument("--coordination-batch-size", type=int, default=4)
    result.add_argument("--probe-budget", type=int, default=50000)
    result.add_argument("--probe-modules", nargs="+")
    result.add_argument("--verify-gradients", action="store_true")
    result.add_argument("--checkpoint-mode", choices=("resume", "weights"), default="resume")
    result.add_argument("--selection-metric", choices=("classification_loss", "reference_loss"), default="classification_loss")
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
