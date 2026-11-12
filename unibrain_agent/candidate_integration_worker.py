"""Independent one or two update checks for generated full candidate adapters."""

import argparse
import hashlib
import json
import os
import random
import traceback
from pathlib import Path

from .full_replacement import AdapterFailure, write_json


def source_hashes(source, repository):
    from .paths import CATALOG, SOURCES
    manifest = json.loads((CATALOG / "sources.json").read_text(encoding="utf-8"))
    result = {}
    for spec in manifest["repositories"]:
        if spec["directory"] not in {repository, "reg_unibrain"}:
            continue
        root = source if spec["directory"] == "reg_unibrain" else SOURCES / repository
        for item in spec["files"]:
            if spec["directory"] == "reg_unibrain" and item["path"] not in {"main.py", "train.py", "model.py", "utils.py"}:
                continue
            observed = hashlib.sha256((root / item["path"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if observed != item["sha256"]:
                raise ValueError("Pinned source hash mismatch: " + item["path"])
            result[spec["directory"] + "/" + item["path"]] = observed
    return result


def forward(model, templates, image, training):
    import torch
    try:
        return model(templates[0], image, templates[1], templates[2], if_train=training)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, TypeError, IndexError, AttributeError) as error:
        raise AdapterFailure("Full host rejected generated tensors: " + str(error)) from error


def mark_adapter_failure(record, error):
    if not any(value is False for value in record["checks"].values()):
        key = {"joint_gradient_check": "gradient", "joint_optimizer_update": "updates"}.get(record.get("phase"), "contract")
        record["checks"][key] = False
    record.update(status="adapter_failed", error=str(error))


def validate(args, record):
    from .full_replacement_worker import available_gpu, selected_device
    device = selected_device(args.gpu)
    record["gpu_check"] = available_gpu(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import numpy as np
    import torch
    from .candidate_integration import bind_adapter, bind_extended_adapters, core_probe, gradient_evidence, registration_geometry
    from .data import OriginalDataset, digest
    from .full_candidates import build_candidate_model
    from .replacement_training import original_losses

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from .full_extended import REPOSITORIES, build_extended_model
    repositories = {"voxelmorph": "reg_voxelmorph_master", "lapirn": "reg_lapirn", "nnunet": "dynamic-network-architectures", **REPOSITORIES}
    repository = repositories[args.method]
    record["source_hashes"] = source_hashes(args.source, repository)
    if args.method in ("voxelmorph", "lapirn", "nnunet"):
        model, config, architecture = build_candidate_model(args.source, args.method, plans=args.plans)
        wrapper = bind_adapter(model, args.method, args.adapter)
        wrappers = {"seg_net" if args.method == "nnunet" else "reg_net": wrapper}
    else:
        model, config, architecture = build_extended_model(args.source, args.method)
        wrappers = bind_extended_adapters(model, args.method, args.adapter)
    model.cuda()
    from model import GCC
    from utils import train_aug

    implementation_files = ["candidate_integration.py", "candidate_integration_worker.py", "full_connectivity.py"]
    if args.method in ("voxelmorph", "lapirn", "nnunet"):
        implementation_files.append("full_candidates.py")
    else:
        implementation_files.append("full_extended.py")
        implementation_files.append("full_preprocessing.py" if args.method in ("redwgan", "deepn4", "deepnormalize", "harmonizing_flows") else
                                    "full_anatomy_candidates.py" if args.method in ("ernet", "fastsurfer", "nobrainer") else "full_graph_candidates.py")
    record.update(architecture=architecture, original_config=config, adapter_sha256=digest(args.adapter),
                  initialization="Random candidate and original host weights; no pretrained accuracy claim",
                  implementation_hashes={name: digest(Path(__file__).with_name(name)) for name in implementation_files})
    dataset = OriginalDataset(args.data / "adhd_train_sample.npy")
    template_names = ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")
    templates = [torch.from_numpy(np.load(args.data / name)).float().cuda().reshape(1, 1, 96, 96, 96) for name in template_names]
    record["data"] = {"split": "train", "available": len(dataset), "sha256": digest(dataset.path),
                      "template_hashes": {name: digest(args.data / name) for name in template_names}}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    active = tuple(dict.fromkeys(("ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred", *wrappers)))
    new_loss = "segmentation" if args.method == "nnunet" else "registration" if args.method in ("voxelmorph", "lapirn", "ernet") else "classification"
    for step in range(args.steps):
        row = dataset[step % len(dataset)]
        image = torch.tensor(row[2], dtype=torch.float32, device="cuda").reshape(1, 1, 96, 96, 96)
        brain = torch.tensor(row[3], dtype=torch.float32, device="cuda").reshape_as(image)
        label = torch.tensor([int(row[1])], device="cuda")
        if config["if_train_aug"]:
            image, brain = train_aug(image, brain, 96, 2, 2, .98, 1.02)
        image = image.detach().requires_grad_(True)
        model.eval()
        record["phase"] = "independent_core_and_spatial_checks"
        with torch.no_grad():
            output = forward(model, templates, image, False)
            record["runs"] = True
            for key in ("contract", "fidelity", "spatial"):
                record["checks"][key] = True
                record["diagnostic"][key] = {}
            for name, wrapper in wrappers.items():
                inputs = tuple(v.detach() if torch.is_tensor(v) else v for v in wrapper.last_inputs)
                evidence, reference = core_probe(wrapper, inputs)
                for key in ("contract", "fidelity", "spatial"):
                    record["checks"][key] &= evidence[key]
                record["diagnostic"]["fidelity"][name] = {key: evidence[key] for key in ("maximum_core_error", "scope")}
                record["diagnostic"]["contract"][name] = {"reference_shapes": evidence["reference_shapes"]}
                if args.method in ("voxelmorph", "lapirn"):
                    spatial = registration_geometry(model, output, reference)
                    record["checks"]["spatial"] &= spatial["passed"]
                    record["diagnostic"]["spatial"][name] = spatial
                else:
                    record["diagnostic"]["spatial"][name] = {"scope": "Same reference stage output tensor coordinates or graph node order; physical image geometry is not certified"}
                del reference
        if not all(record["checks"][name] for name in ("contract", "spatial", "fidelity")):
            raise AdapterFailure("Generated adapter disagrees with the pinned tensor or spatial contract")
        del output
        model.train()
        optimizer.zero_grad(set_to_none=True)
        record["phase"] = "joint_gradient_check"
        output = forward(model, templates, image, True)
        try:
            losses = original_losses(output, brain, label)
            losses["registration"] = GCC()(output[2][-1], templates[0])
        except torch.cuda.OutOfMemoryError:
            raise
        except (ValueError, RuntimeError) as error:
            record["checks"]["contract"] = False
            raise AdapterFailure("Generated training outputs cannot enter the original losses: " + str(error)) from error
        gradient = {"candidate_objective": new_loss, "modules": {}}
        passed = True
        for name, wrapper in wrappers.items():
            local = gradient_evidence(losses[new_loss], wrapper.parameters(), wrapper.last_inputs)
            gradient["modules"][name] = {"candidate": local}
            passed &= local["passed"]
            if new_loss == "registration":
                downstream = gradient_evidence(losses["classification"], wrapper.parameters(), wrapper.last_inputs)
                gradient["modules"][name]["classification"] = downstream
                passed &= downstream["passed"]
        if args.method == "nnunet":
            gradient["scope"] = "Tissue loss to candidate and MRI; the original auxiliary tissue head does not feed classification"
        record["checks"]["gradient"] = bool(passed)
        record["diagnostic"]["gradient"] = gradient
        if not passed:
            raise AdapterFailure("A required candidate parameter or input gradient is disconnected")
        weights = {"extraction": config["lambda_ext"], "registration": config["lambda_reg"],
                   "segmentation": config["lambda_seg"], "classification": config["lambda_cls"]}
        total = sum(weights[name] * loss for name, loss in losses.items())
        if hasattr(model, "regularization_loss"):
            total = total + .01 * model.regularization_loss()
        if not torch.isfinite(total):
            record["checks"]["contract"] = False
            raise AdapterFailure("Nonfinite joint objective")
        total.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            record["checks"]["gradient"] = False
            raise AdapterFailure("Nonfinite joint gradients")
        snapshots = {name: [(p, p.detach().cpu().clone()) for p in getattr(model, name).parameters() if p.grad is not None]
                     for name in active}
        candidate_snapshots = {name: [(p, p.detach().cpu().clone()) for p in wrapper.parameters() if p.grad is not None]
                               for name, wrapper in wrappers.items()}
        record["phase"] = "joint_optimizer_update"
        optimizer.step()
        record["optimizer_updates"] += 1
        changed = {name: sum(not torch.equal(p.detach().cpu(), before) for p, before in pairs) for name, pairs in snapshots.items()}
        candidate_changed = {name: sum(not torch.equal(p.detach().cpu(), before) for p, before in pairs)
                             for name, pairs in candidate_snapshots.items()}
        record["checks"]["updates"] = all(candidate_changed.values()) and all(changed.values())
        record["diagnostic"].setdefault("updates", []).append({"step": step + 1, "candidate_changed_tensors": candidate_changed,
                                                             "host_changed_tensors": changed, "losses": {k: float(v.detach()) for k, v in losses.items()}})
        write_json(args.output, record)
        if not record["checks"]["updates"]:
            raise AdapterFailure("The candidate or an active original host module did not update")
        del snapshots, candidate_snapshots, output, losses, total
        for wrapper in wrappers.values():
            wrapper.last_inputs = None
    model.eval()
    with torch.no_grad():
        output = forward(model, templates, image.detach(), False)
        record["inference"] = {"logits_shape": list(output[-1].shape), "finite": bool(torch.isfinite(output[-1]).all())}
        if not record["inference"]["finite"]:
            record["checks"]["contract"] = False
            raise AdapterFailure("Nonfinite inference output")
    checkpoint = args.output.parent / "full_candidate_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    record["checkpoint"] = {"file": checkpoint.name, "sha256": digest(checkpoint)}
    record["source_unchanged"] = source_hashes(args.source, repository) == record["source_hashes"]
    if not record["source_unchanged"]:
        raise RuntimeError("Source provenance changed during evaluation")
    record["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
    record["status"] = "passed"


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "data", "adapter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    from .candidate_integration import METHODS
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing evaluation")
    if any(args.output.resolve() == p.resolve() or p.resolve() in args.output.resolve().parents for p in (args.source, args.data)):
        parser.error("Evaluation output must be separate from source and data")
    record = {"status": "checking", "method": args.method, "seed": args.seed, "runs": False, "optimizer_updates": 0,
              "checks": dict.fromkeys(("contract", "gradient", "spatial", "fidelity", "updates")), "diagnostic": {},
              "scope": "Generated adapter in the original full host, native 96 cubed, one or two joint updates"}
    try:
        from .candidate_integration import check_adapter
        try:
            check_adapter(args.adapter.read_text(encoding="utf-8"), args.method)
        except (SyntaxError, ValueError) as error:
            record["checks"]["contract"] = False
            raise AdapterFailure(str(error)) from error
        args.output.parent.mkdir(parents=True, exist_ok=True)
        validate(args, record)
    except AdapterFailure as error:
        mark_adapter_failure(record, error)
    except Exception as error:
        record.update(status="infrastructure_failed", error=str(error))
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
