"""Independent full host and pinned extraction core evaluation."""

import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unibrain_agent.integration_benchmark import save
from unibrain_agent.full_replacement import AdapterFailure, check_code


def core_agreement(adapter, checkpoint, data, gpu, seed, device_checked=False):
    from unibrain_agent.full_replacement_worker import available_gpu, selected_device

    if not device_checked:
        device = selected_device(gpu)
        available_gpu(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    import torch
    from torch.nn import functional as F
    from unibrain_agent.data import OriginalDataset
    from unibrain_agent.full_model import GeneratedExtraction

    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    model = GeneratedExtraction(adapter)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = "ext_net.core."
        core_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model.core.load_state_dict(core_state, strict=True)
        del state, core_state
    model.cuda().eval()
    row = OriginalDataset(data / "adhd_train_sample.npy")[0]
    image = torch.tensor(row[2], dtype=torch.float32, device="cuda").reshape(1, 1, 96, 96, 96)
    with torch.no_grad():
        distance = model.core(F.pad(image, (16, 16, 16, 16, 16, 16)))[..., 16:112, 16:112, 16:112]
        expected = torch.sigmoid(-distance)
        actual_image, actual_mask = model(image, True)
        _, inference_mask = model(image, False)
        error = (actual_mask[0] - expected).abs().max().item()
        reconstructed = torch.allclose(actual_image[0], image * actual_mask[0], atol=1e-6, rtol=1e-5)
        inference = torch.allclose(inference_mask[0], expected, atol=1e-6, rtol=1e-5) or torch.equal(
            inference_mask[0], (expected >= 0.5).to(image.dtype))
        spatial = torch.equal(actual_mask[0] >= 0.5, expected >= 0.5) and reconstructed
    return {"passed": error <= 1e-5 and reconstructed and inference, "maximum_mask_error": error,
            "spatial_passed": spatial,
            "extracted_image_consistent": reconstructed, "inference_consistent": inference,
            "weights": "Trained full-model checkpoint" if checkpoint else "Independent seeded core initialization",
            "scope": "Pinned SynthStrip core under the declared padded 96 cubed tensor contract; not official preprocessing or pretrained clinical accuracy"}


def evaluate(args):
    if args.method != "synthstrip":
        from unibrain_agent.candidate_integration import check_adapter
        from unibrain_agent.candidate_integration_worker import mark_adapter_failure, validate
        report = {"status": "checking", "method": args.method, "seed": args.seed, "runs": False, "optimizer_updates": 0,
                  "checks": dict.fromkeys(("contract", "gradient", "spatial", "fidelity", "updates")), "diagnostic": {},
                  "scope": "Generated adapter in the original full host, native 96 cubed, one or two joint updates"}
        try:
            try:
                check_adapter(args.adapter.read_text(encoding="utf-8"), args.method)
            except (SyntaxError, ValueError) as error:
                report["checks"]["contract"] = False
                raise AdapterFailure(str(error)) from error
            validate(args, report)
        except AdapterFailure as error:
            mark_adapter_failure(report, error)
        except Exception as error:
            report.update(status="infrastructure_failed", error=str(error))
        return report
    try:
        check_code(args.adapter.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as error:
        return {"status": "adapter_failed", "runs": False, "optimizer_updates": 0,
                "checks": {"contract": False}, "diagnostic": {"contract": str(error)}}
    work = args.output.resolve().parent / "full_checker"
    work.mkdir(parents=True, exist_ok=False)
    from unibrain_agent.full_replacement_worker import validate
    observed = {"status": "checking"}
    worker_args = SimpleNamespace(**{**vars(args), "output": work})
    try:
        validate(worker_args, observed)
    except AdapterFailure as error:
        observed.update(status="adapter_failed", error=str(error))
    except Exception as error:
        observed.update(status="infrastructure_failed", error=str(error))
    save(work / "validation.json", observed)
    updates = observed.get("updates", [])
    gradient = observed.get("gradient_check", {})
    gradient_norm = gradient.get("registration_to_extraction_norm", 0)
    input_norm = gradient.get("input_gradient_norm") or 0
    record = {"status": observed["status"], "runs": bool(gradient or updates), "optimizer_updates": len(updates),
              "seed": args.seed, "checks": {"contract": observed.get("status") == "passed",
                                              "gradient": math.isfinite(gradient_norm) and gradient_norm > 0 and math.isfinite(input_norm) and input_norm > 0,
                                              "spatial": None, "fidelity": None,
                                              "updates": bool(updates) and all(all(u["changed_parameter_tensors"].values()) for u in updates)},
              "diagnostic": {"gradient": gradient, "updates": updates}, "executor": "in_process_full_worker",
              "data": observed.get("data", {}),
              "evaluator_scope": "Full host trainability and spatial/core output agreement for SynthStrip extraction"}
    if observed["status"] not in {"passed", "adapter_failed"}:
        record.update(status="infrastructure_failed", error=observed.get("error", "Worker did not complete"))
        return record
    checkpoint = work / "checkpoint.pt"
    try:
        from unibrain_agent.candidate_integration_worker import source_hashes
        record["source_hashes"] = source_hashes(args.source, "freesurfer_synthstrip")
        fidelity = core_agreement(args.adapter, checkpoint if checkpoint.exists() else None, args.data.resolve(), args.gpu, args.seed, device_checked=True)
    except AdapterFailure as error:
        record.update(status="adapter_failed", error=str(error))
        record["checks"]["contract"] = False
        return record
    except Exception as error:
        record.update(status="infrastructure_failed", error=str(error))
        return record
    record["checks"]["contract"] = True
    record["checks"]["fidelity"] = fidelity["passed"]
    record["checks"]["spatial"] = fidelity["spatial_passed"]
    record["diagnostic"]["fidelity"] = {key: value for key, value in fidelity.items() if key != "spatial_passed"}
    record["diagnostic"]["spatial"] = {"scope": "Aligned tensor grid and reference core output; physical image geometry not certified"}
    record["status"] = "passed" if all(record["checks"].values()) else "adapter_failed"
    return record


def main():
    parser = argparse.ArgumentParser()
    for field in ("source", "data", "adapter", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=17)
    from unibrain_agent.candidate_integration import METHODS
    parser.add_argument("--method", choices=("synthstrip", *METHODS), default="synthstrip")
    parser.add_argument("--plans", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing evaluation")
    if any(args.output.resolve() == p.resolve() or p.resolve() in args.output.resolve().parents for p in (args.source, args.data)):
        parser.error("Evaluation output must be separate from source and data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = evaluate(args)
    except Exception as error:
        report = {"status": "infrastructure_failed", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
