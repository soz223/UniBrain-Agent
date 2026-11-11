"""Check a genuine planned candidate at full resolution on the CPU."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from unibrain_agent.data import OriginalDataset, digest
from unibrain_agent.full_candidates import planned_execution, planned_unet
from unibrain_agent.nnunet_prepare import validate_plan_provenance
from unibrain_agent.sources import verify


def check(args):
    if args.output.exists():
        raise ValueError("Use a new report path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "device": "cpu", "gpu_used": False, "seed": args.seed,
              "scope": "Isolated full resolution planned candidate; not a complete UniBrain training acceptance"}
    started = time.monotonic()
    try:
        training = args.data / "adhd_train_sample.npy"
        report["plan_provenance"] = validate_plan_provenance(args.plans, digest(training), args.dataset, args.fold)
        report["source"] = verify(["dynamic-network-architectures"])
        report["plan_sha256"] = digest(args.plans)
        report["execution"] = planned_execution(json.loads(args.plans.read_text(encoding="utf-8")))
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        model = planned_unet(args.plans).cpu().train()
        dataset = OriginalDataset(training)
        row = dataset[0]
        image = torch.tensor(row[2], dtype=torch.float32).reshape(1, 1, 96, 96, 96).requires_grad_(True)
        target = torch.tensor(row[4], dtype=torch.float32).reshape(1, 1, 96, 96, 96)
        if not torch.equal(target, target.round()) or target.min() < 0 or target.max() > 3:
            raise ValueError("Expected four tissue reference classes")
        output = model(image)
        if not isinstance(output, list) or output[0].shape != (1, 4, 96, 96, 96) or not all(torch.isfinite(value).all() for value in output):
            raise ValueError("Invalid multiscale full resolution outputs")
        factors = [2. ** -index for index in range(len(output))]
        loss = sum(weight * F.cross_entropy(value, F.interpolate(target, size=value.shape[2:], mode="nearest")[:, 0].long())
                   for weight, value in zip(factors, output)) / sum(factors)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        valid_gradients = bool(gradients and all(torch.isfinite(value).all() for value in gradients)
                               and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0)
        if not valid_gradients:
            raise ValueError("Invalid input or parameter gradients")
        before = [parameter.detach().clone() for parameter in model.parameters()]
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        changed = sum(not torch.equal(previous, current) for previous, current in zip(before, model.parameters()))
        if not changed or digest(args.plans) != report["plan_sha256"]:
            raise ValueError("Parameters did not change or the native plan was modified")
        report.update(status="passed", torch_version=torch.__version__, parameters=sum(p.numel() for p in model.parameters()),
                      input_shape=list(image.shape), output_shapes=[list(value.shape) for value in output],
                      finite_outputs=True, finite_input_gradient=True, input_gradient_l1=float(image.grad.abs().sum()),
                      finite_parameter_gradients=True, parameters_with_gradients=len(gradients),
                      parameter_gradient_l1=sum(float(value.abs().sum()) for value in gradients),
                      parameter_tensors_changed=changed, optimizer_updates=1, loss=float(loss.detach()),
                      target_kind="Training sample automatic tissue reference", native_plan_unchanged=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plans", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    result = check(arguments)
    print(json.dumps({"status": result["status"], "report": str(arguments.output)}))
