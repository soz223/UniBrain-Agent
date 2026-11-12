"""Run a full 96 cubed candidate contract and one CPU optimizer update."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from unibrain_agent.full_anatomy_candidates import build_anatomy_modules, ernet_definitions
from unibrain_agent.sources import load


def native_reference(method, modules, image, reference, extracted=None):
    if method == "ernet":
        namespace = ernet_definitions()
        load("reg_ernet/model.py", {"get_batch_identity_theta_4_4", "get_batch_identity_theta_3_4",
             "theta_3_4_to_4_4", "theta_dot", "multi_stage_reg"}, namespace)
        native = namespace["multi_stage_reg"](96, 5)
        native.load_state_dict(modules["reg_net"].state_dict(), strict=True)
        return native(reference, extracted)
    core = copy.deepcopy(modules["par_net"].core)
    if method == "nobrainer":
        return core(image)
    padded = F.pad(image[0, 0].permute(1, 2, 0), (3, 3), mode="replicate")
    slabs = padded.unfold(-1, 7, 1).permute(2, 3, 0, 1).contiguous()
    outputs = [core(batch) for batch in slabs.split(modules["par_net"].slice_batch)]
    return torch.cat(outputs).permute(1, 0, 2, 3).unsqueeze(0)


def run(args):
    if args.output.exists():
        raise ValueError("Use a new report path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"method": args.method, "status": "running", "device": "cpu", "gpu_used": False,
              "scope": "Full resolution candidate cores with synthetic tensors; not full host or published tool accuracy"}
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(17)
        parts, architecture = build_anatomy_modules(args.method)
        modules = nn.ModuleDict(parts).train()
        report["architecture"] = architecture
        image = torch.randn(1, 1, 96, 96, 96, requires_grad=True)
        reference = torch.randn_like(image)
        if args.method == "ernet":
            extracted, masks = modules["ext_net"](image, True)
            warped, affines, inverses = modules["reg_net"](reference, extracted[-1])
            with torch.no_grad():
                expected, expected_grids = native_reference(args.method, modules, image.detach(), reference, extracted[-1].detach())
            agreement = all(torch.allclose(a.detach(), b, atol=1e-6, rtol=1e-5) for a, b in zip(warped, expected))
            report["native_forward_max_error"] = max(float((a.detach() - b).abs().max()) for a, b in zip(warped, expected))
            agreement = agreement and torch.allclose(modules["reg_net"].forward_grid.detach(), expected_grids[-1], atol=1e-6, rtol=1e-5)
            identity = torch.eye(4)[None]
            last = torch.cat((affines[-1], identity[:, 3:4]), dim=1)
            inverse = torch.cat((inverses[-1], identity[:, 3:4]), dim=1)
            report["inverse_error"] = float((last @ inverse - identity).abs().max().detach())
            outputs = [extracted[-1], masks[-1], warped[-1], affines[-1], inverses[-1]]
            loss = F.binary_cross_entropy(masks[-1], (image.detach() > 0).float()) + (warped[-1] - reference).square().mean()
        else:
            logits = modules["par_net"](image)
            with torch.no_grad():
                expected = native_reference(args.method, modules, image.detach(), reference)
            agreement = torch.allclose(logits.detach(), expected, atol=1e-6, rtol=1e-5)
            report["native_forward_max_error"] = float((logits.detach() - expected).abs().max())
            target = (torch.arange(96 ** 3) % 117).reshape(1, 96, 96, 96)
            outputs = [logits]
            if logits.shape != (1, 117, 96, 96, 96):
                raise ValueError("The complete AAL volume was not produced")
            loss = F.cross_entropy(logits, target)
        del expected
        if not agreement or not all(torch.isfinite(value).all() for value in outputs):
            raise ValueError("Native core forward mismatch or nonfinite outputs")
        loss.backward()
        if image.grad is None or not torch.isfinite(image.grad).all() or image.grad.abs().sum() <= 0:
            raise ValueError("Invalid input gradient")
        gradients = [parameter.grad for parameter in modules.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise ValueError("Invalid parameter gradients")
        before = {name: parameter.detach().clone() for name, parameter in modules.named_parameters()}
        torch.optim.SGD(modules.parameters(), lr=1e-4).step()
        updated = {name: sum(not torch.equal(before[name + "." + key], parameter) for key, parameter in module.named_parameters())
                   for name, module in modules.items()}
        if not all(updated.values()):
            raise ValueError("A candidate module did not update")
        report.update(status="passed", input_shape=list(image.shape), output_shapes=[list(value.shape) for value in outputs],
                      native_core_forward_agreement=True, official_tool_fidelity_claimed=False,
                      finite_outputs=True, input_gradient_l1=float(image.grad.abs().sum()),
                      finite_parameter_gradients=True, parameter_gradient_l1=sum(float(value.abs().sum()) for value in gradients),
                      updated_parameter_tensors=updated, optimizer_updates=1, loss=float(loss.detach()))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fastsurfer", "nobrainer", "ernet"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "report": str(args.output)}))
