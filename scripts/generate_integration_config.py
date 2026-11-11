"""Create concrete full candidate benchmark configurations on this machine."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unibrain_agent.full_replacement import CONTRACT
from unibrain_agent.paths import SOURCES


CANDIDATES = {
    "synthstrip": ("extraction", "freesurfer_synthstrip", ["mri_synthstrip/mri_synthstrip"]),
    "voxelmorph": ("registration", "reg_voxelmorph_master", ["voxelmorph/torch/networks.py", "voxelmorph/torch/layers.py"]),
    "lapirn": ("registration", "reg_lapirn", ["Code/miccai2020_model_stage.py", "Code/Functions.py"]),
    "nnunet": ("segmentation", "dynamic-network-architectures", ["dynamic_network_architectures/architectures/unet.py"]),
}
UNIMPLEMENTED = ("redwgan", "deepn4", "ernet", "deepnormalize", "harmonizing_flows",
                 "fastsurfer", "nobrainer", "fbnetgen", "brainnetcnn", "bnt")


def relative(path, root):
    try:
        return os.path.relpath(Path(path).resolve(), Path(root).resolve())
    except ValueError:
        return str(Path(path).resolve())


def make_case(method, source, data, output, gpu=0, steps=2, plans=None):
    stage, repository, files = CANDIDATES[method]
    required = [Path(data) / name for name in ("adhd_train_sample.npy", "template_img_orig_96.npy",
                                               "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")]
    evidence = [Path(source) / "model.py", *[SOURCES / repository / name for name in files]]
    command = ["{python}", str(ROOT / "scripts/evaluate_integration.py"), "--method", method,
               "--source", str(Path(source).resolve()), "--data", str(Path(data).resolve()),
               "--adapter", "{adapter}", "--output", "{output}", "--seed", "{seed}", "--gpu", str(gpu), "--steps", str(steps)]
    if method == "synthstrip":
        interface = CONTRACT
        knowledge = "Preserve signed-distance polarity, center padding and cropping; keep the soft training mask differentiable."
    elif method == "nnunet":
        interface = ("Define only adapt(core, image). core is the unchanged dataset-planned full 3D nnU-Net tissue model with four classes. "
                     "Input is [1,1,96,96,96]. Return the core's native list of deep supervision logits, highest resolution first; "
                     "retain all scales and logits, with no argmax or softmax. Use only core, torch, F and tensor operations. "
                     "The adapter is installed into seg_net in the original full host. Preserve input and parameter gradients. "
                     "Do not import, mutate core, read files, execute tools or define helper functions.")
        knowledge = "The host uses normalized multi-scale soft-target tissue CE. The auxiliary tissue head does not feed the original classifier."
    else:
        details = ("Return the native tuple (forward warped image, backward warped image, pre-integration velocity). "
                   "Velocity is in dhw voxel units at the native integration resolution."
                   if method == "voxelmorph" else
                   "Return the complete native level-three tuple unchanged, including normalized xyz displacement at index zero "
                   "and velocity at index three.")
        interface = ("Define only adapt(core, moving, fixed). Both inputs are [1,1,96,96,96]. " + details +
                     " The generated function is installed inside the full registration module after five original affine stages. "
                     "The host converts and composes the dense field; preserve core output order, units, grids and gradients. "
                     "Use only core, torch, F and tensor operations. Do not import, mutate core, read files, execute tools or define helpers.")
        knowledge = ("Dense grids sample moving images in fixed coordinates. The host composes forward and inverse affine plus dense grids, "
                     "uses align_corners=True, and samples soft AAL channels for differentiable ROI features. "
                     "Registration and classification gradients must both reach the dense candidate core.")
    if method == "nnunet" and plans:
        required.append(Path(plans))
        evidence.append(Path(plans))
        command.extend(("--plans", str(Path(plans).resolve())))
    evaluator = {"cwd": relative(ROOT, output), "command": command, "timeout": 1800}
    if method == "synthstrip":
        evaluator["progress_file"] = "full_checker/validation.json"
    return {"id": method, "method": method, "stage": stage, "expected_fit": True,
            "applicability_scope": "Declared adapted core contract; not the complete official preprocessing or pretrained application",
            "implementation": "full_generated_adapter", "plans_supplied": plans is not None,
            "source_files": [relative(path, output) for path in evidence],
            "required_files": [relative(path, output) for path in required],
            "interface": interface, "knowledge": knowledge, "max_optimizer_updates": steps,
            "evaluator": evaluator}


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--methods", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--provider", choices=("codex", "claude", "openai", "anthropic", "compatible"), default="codex")
    parser.add_argument("--model", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = {"scope": "Concrete generated adapter evaluators; no GPU or AI calls executed",
             "configs": {}, "not_implemented": {method: "Full generated integration evaluator not implemented; compact adapters are excluded"
                                                 for method in UNIMPLEMENTED}}
    for method in args.methods:
        case = make_case(method, args.source, args.data, args.output, args.gpu, args.steps, args.plans)
        value = {"version": 1, "cases": [case], "providers": [{"provider": args.provider, "model": args.model}],
                 "variants": ["general", "checks", "workflow"], "seeds": [17], "repetitions": 3, "max_calls": 4}
        filename = method + ".json"
        (args.output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        index["configs"][method] = filename
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
