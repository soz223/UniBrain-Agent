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


def validate(args, record):
    from .full_replacement_worker import available_gpu, selected_device
    device = selected_device(args.gpu)
    record["gpu_check"] = available_gpu(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import numpy as np
    import torch
    from .candidate_integration import bind_adapter, core_probe, gradient_evidence, registration_geometry
    from .data import OriginalDataset, digest
    from .full_candidates import build_candidate_model
    from .replacement_training import original_losses

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    repository = {"voxelmorph": "reg_voxelmorph_master", "lapirn": "reg_lapirn", "nnunet": "dynamic-network-architectures"}[args.method]
    record["source_hashes"] = source_hashes(args.source, repository)
    model, config, architecture = build_candidate_model(args.source, args.method, plans=args.plans)
    wrapper = bind_adapter(model, args.method, args.adapter)
    model.cuda()
    from model import GCC
    from utils import train_aug

    record.update(architecture=architecture, original_config=config, adapter_sha256=digest(args.adapter),
                  initialization="Random candidate and original host weights; no pretrained accuracy claim",
                  implementation_hashes={name: digest(Path(__file__).with_name(name)) for name in
                                         ("candidate_integration.py", "candidate_integration_worker.py", "full_candidates.py")})
    dataset = OriginalDataset(args.data / "adhd_train_sample.npy")
    template_names = ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")
    templates = [torch.from_numpy(np.load(args.data / name)).float().cuda().reshape(1, 1, 96, 96, 96) for name in template_names]
    record["data"] = {"split": "train", "available": len(dataset), "sha256": digest(dataset.path),
                      "template_hashes": {name: digest(args.data / name) for name in template_names}}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    active = ("ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred")
    new_loss = "segmentation" if args.method == "nnunet" else "registration"
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
            evidence, reference = core_probe(wrapper, tuple(v.detach() for v in wrapper.last_inputs))
            record["checks"].update({key: evidence[key] for key in ("contract", "fidelity", "spatial")})
            record["diagnostic"]["fidelity"] = {key: evidence[key] for key in ("maximum_core_error", "scope")}
            record["diagnostic"]["contract"] = {"reference_shapes": evidence["reference_shapes"]}
            if args.method != "nnunet":
                spatial = registration_geometry(model, output, reference)
                record["checks"]["spatial"] &= spatial["passed"]
                record["diagnostic"]["spatial"] = spatial
            else:
                record["diagnostic"]["spatial"] = {"scope": "Class logits and voxel labels on every declared native deep supervision grid"}
        if not all(record["checks"][name] for name in ("contract", "spatial", "fidelity")):
            raise AdapterFailure("Generated adapter disagrees with the pinned tensor or spatial contract")
        del output, reference
        model.train()
        optimizer.zero_grad(set_to_none=True)
        record["phase"] = "joint_gradient_check"
        output = forward(model, templates, image, True)
        losses = original_losses(output, brain, label)
        losses["registration"] = GCC()(output[2][-1], templates[0])
        local = gradient_evidence(losses[new_loss], wrapper.parameters(), wrapper.last_inputs)
        gradient = {"candidate_objective": new_loss, "candidate": local}
        if args.method != "nnunet":
            downstream = gradient_evidence(losses["classification"], wrapper.parameters(), wrapper.last_inputs)
            gradient["classification_to_registration"] = downstream
            passed = local["passed"] and downstream["passed"]
        else:
            gradient["scope"] = "Tissue loss to candidate and MRI; the original auxiliary tissue head does not feed classification"
            passed = local["passed"]
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
        candidate_snapshots = [(p, p.detach().cpu().clone()) for p in wrapper.parameters() if p.grad is not None]
        record["phase"] = "joint_optimizer_update"
        optimizer.step()
        record["optimizer_updates"] += 1
        changed = {name: sum(not torch.equal(p.detach().cpu(), before) for p, before in pairs) for name, pairs in snapshots.items()}
        candidate_changed = sum(not torch.equal(p.detach().cpu(), before) for p, before in candidate_snapshots)
        record["checks"]["updates"] = bool(candidate_changed and all(changed.values()))
        record["diagnostic"].setdefault("updates", []).append({"step": step + 1, "candidate_changed_tensors": candidate_changed,
                                                             "host_changed_tensors": changed, "losses": {k: float(v.detach()) for k, v in losses.items()}})
        write_json(args.output, record)
        if not record["checks"]["updates"]:
            raise AdapterFailure("The candidate or an active original host module did not update")
        del snapshots, candidate_snapshots, output, losses, total
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
    parser.add_argument("--method", choices=("voxelmorph", "lapirn", "nnunet"), required=True)
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
        if not record["runs"]:
            record["checks"]["contract"] = False
        record.update(status="adapter_failed", error=str(error))
    except Exception as error:
        record.update(status="infrastructure_failed", error=str(error))
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
