"""Check generated adapters in an unchanged full UniBrain host."""

import argparse
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

from .full_replacement import AdapterFailure, write_json


def available_gpu(index):
    query = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=uuid,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    uuid, free = query.stdout.strip().split(",")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True)
    occupied = []
    for line in processes.stdout.splitlines():
        fields = [v.strip() for v in line.split(",", 2)]
        if len(fields) != 3 or fields[0] != uuid.strip() or fields[1] == str(os.getpid()):
            continue
        # WDDM also lists desktop rendering applications as compute clients.
        if os.name != "nt" or any(word in fields[2].lower() for word in ("python", "train", "torch", "jupyter")):
            occupied.append({"pid": fields[1], "name": fields[2]})
    if occupied or int(free.strip()) < 4500:
        raise RuntimeError("Selected GPU is occupied or has less than 4500 MiB free")
    return {"index": index, "uuid": uuid.strip(), "free_mib": int(free),
            "scope": "No compute clients on Linux; no detected training clients on Windows WDDM"}


def selected_device(index):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if index < 0:
        raise ValueError("GPU index must be nonnegative")
    if visible is None:
        return str(index)
    devices = [value.strip() for value in visible.split(",") if value.strip() and value.strip() != "-1"]
    if index >= len(devices):
        raise ValueError("GPU index is outside CUDA_VISIBLE_DEVICES")
    return devices[index]


def validate(args, record):
    device = selected_device(args.gpu)
    record["gpu_check"] = available_gpu(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .data import OriginalDataset, digest
    from .native import verify_original
    from .sources import verify
    from .full_model import build_full_model

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    record["source_before"] = verify_original(args.source)
    record["candidate_source"] = verify(["freesurfer_synthstrip"])
    model, config, architecture = build_full_model(args.source, args.adapter)
    from model import GCC
    from utils import am_1d_2_Nd_torch, train_aug

    model.cuda().train()
    record.update(**architecture, original_config=config,
                  initialization="Random weights, full default SynthStrip and full original UniBrain host",
                  adapter_sha256=digest(args.adapter), seed=args.seed,
                  boundary="Registration loss to extraction parameters; original atlas and graph boundaries retained")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    dataset = OriginalDataset(args.data / "adhd_train_sample.npy")
    templates = [torch.from_numpy(np.load(args.data / name)).cuda().float().reshape(1, 1, 96, 96, 96)
                 for name in ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")]
    record["data"] = {"split": "train", "available": len(dataset), "sha256": digest(Path(dataset.path)),
                      "templates": {name: digest(args.data / name) for name in
                                    ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")}}
    active = ("ext_net", "reg_net", "seg_net", "feature_net", "gnn_pred")
    record["updates"] = []
    for step in range(args.steps):
        row = dataset[step % len(dataset)]
        image = torch.tensor(row[2], dtype=torch.float32, device="cuda").reshape(1, 1, 96, 96, 96)
        brain = torch.tensor(row[3], dtype=torch.float32, device="cuda").reshape_as(image)
        label = torch.tensor([int(row[1])], device="cuda")
        if config["if_train_aug"]:
            image, brain = train_aug(image, brain, 96, 2, 2, 0.98, 1.02)
        image = image.detach().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        try:
            output = model(templates[0], image, templates[1], templates[2], if_train=True)
        except (ValueError, TypeError, IndexError) as error:
            raise AdapterFailure(str(error)) from error
        mask, extracted = output[1][-1], output[0][-1]
        if (mask.shape != image.shape or extracted.shape != image.shape or not torch.isfinite(mask).all()
                or not torch.isfinite(extracted).all() or mask.min() < 0 or mask.max() > 1):
            raise AdapterFailure("Extraction tensor interface, finite values or probability range failed")
        if not torch.allclose(extracted, image * mask):
            raise AdapterFailure("Extracted image must equal input multiplied by mask")
        losses = {"extraction": F.binary_cross_entropy(mask, brain),
                  "registration": GCC()(output[2][-1], templates[0]),
                  "segmentation": F.cross_entropy(output[5], am_1d_2_Nd_torch(output[6])),
                  "classification": F.cross_entropy(output[-1], label)}
        ext_parameters = tuple(model.ext_net.parameters())
        gradients = torch.autograd.grad(losses["registration"], ext_parameters, allow_unused=True, retain_graph=True)
        present = [g for g in gradients if g is not None]
        norm = sum(g.detach().double().square().sum().item() for g in present) ** 0.5
        input_gradient = torch.autograd.grad(extracted.sum(), image, retain_graph=True, allow_unused=True)[0]
        evidence = {"registration_to_extraction_norm": norm, "connected_tensors": len(present),
                    "total_tensors": len(ext_parameters), "mask_requires_grad": mask.requires_grad,
                    "input_gradient_norm": input_gradient.norm().item() if input_gradient is not None else None}
        record["gradient_check"] = evidence
        if not present or not np.isfinite(norm) or norm == 0 or not mask.requires_grad:
            raise AdapterFailure("Registration loss cannot train the extraction core: " + json.dumps(evidence))
        if input_gradient is None or not torch.isfinite(input_gradient).all() or input_gradient.norm() == 0:
            raise AdapterFailure("Missing or invalid extraction input gradient")
        del gradients, present, input_gradient
        weights = dict(zip(losses, (config["lambda_ext"], config["lambda_reg"], config["lambda_seg"], config["lambda_cls"])))
        total = sum(weights[name] * value for name, value in losses.items())
        if not torch.isfinite(total):
            raise AdapterFailure("Nonfinite joint loss")
        total.backward()
        finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        if not finite:
            raise AdapterFailure("Nonfinite joint gradients")
        snapshots = {name: [(p, p.detach().cpu().clone()) for p in getattr(model, name).parameters() if p.grad is not None]
                     for name in active}
        optimizer.step()
        changed = {name: sum(not torch.equal(p.detach().cpu(), before) for p, before in pairs)
                   for name, pairs in snapshots.items()}
        record["updates"].append({"step": step + 1, "losses": {k: v.item() for k, v in losses.items()},
                                  "total_loss": total.item(), "changed_parameter_tensors": changed,
                                  "gradient": evidence, "finite_gradients": finite})
        write_json(args.output / "validation.json", record)
        if not all(changed.values()):
            raise AdapterFailure("An active host module did not update: " + json.dumps(changed))
        del snapshots, output, losses, total, mask, extracted
    model.eval()
    with torch.no_grad():
        output = model(templates[0], image.detach(), templates[1], templates[2], if_train=False)
        if not torch.isfinite(output[-1]).all():
            raise AdapterFailure("Nonfinite inference logits")
        record["inference"] = {"logits_shape": list(output[-1].shape), "finite": True}
    checkpoint = args.output / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    record["checkpoint"] = {"file": checkpoint.name, "sha256": digest(checkpoint)}
    record["source_after"] = verify_original(args.source)
    record["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
    record["status"] = "passed"


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "data", "adapter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    record = {"status": "checking", "scope": "One or two full-model joint updates, not an accuracy experiment"}
    try:
        validate(args, record)
    except AdapterFailure as error:
        record.update(status="adapter_failed", error=str(error))
    except Exception as error:
        record.update(status="infrastructure_failed", error=str(error))
        traceback.print_exc()
    write_json(args.output / "validation.json", record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
