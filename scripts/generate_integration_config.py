"""Create concrete full candidate benchmark configurations on this machine."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unibrain_agent.full_replacement import CONTRACT
from unibrain_agent.paths import CATALOG, SOURCES


CANDIDATES = {
    "synthstrip": ("extraction", "freesurfer_synthstrip", ["mri_synthstrip/mri_synthstrip"]),
    "voxelmorph": ("registration", "reg_voxelmorph_master", ["voxelmorph/torch/networks.py", "voxelmorph/torch/layers.py"]),
    "lapirn": ("registration", "reg_lapirn", ["Code/miccai2020_model_stage.py", "Code/Functions.py"]),
    "nnunet": ("segmentation", "dynamic-network-architectures", ["dynamic_network_architectures/architectures/unet.py"]),
}
EXTENDED = {
    "redwgan": ("denoising", "denoise_redwgan"), "deepn4": ("bias_correction", "DeepN4"),
    "ernet": ("extraction_registration", "reg_ernet"), "deepnormalize": ("normalization", "deepNormalize"),
    "harmonizing_flows": ("harmonization", "HarmonizingFlows"), "fastsurfer": ("parcellation", "FastSurfer"),
    "nobrainer": ("parcellation", "nobrainer"), "fbnetgen": ("network_generation", "FBNETGEN"),
    "brainnetcnn": ("classification", "FBNETGEN"), "bnt": ("classification", "BrainNetworkTransformer"),
}
CANDIDATES.update({method: (stage, repository, []) for method, (stage, repository) in EXTENDED.items()})
UNIMPLEMENTED = ()


def relative(path, root):
    try:
        return os.path.relpath(Path(path).resolve(), Path(root).resolve())
    except ValueError:
        return str(Path(path).resolve())


def make_case(method, source, data, output, gpu=0, steps=2, plans=None):
    stage, repository, files = CANDIDATES[method]
    if not files:
        manifest = json.loads((CATALOG / "sources.json").read_text(encoding="utf-8"))
        entry = next(item for item in manifest["repositories"] if item["directory"] == repository)
        files = [item["path"] for item in entry["files"] if Path(item["path"]).suffix in (".py", ".json", ".yaml", ".yml")]
    required = [Path(data) / name for name in ("adhd_train_sample.npy", "template_img_orig_96.npy",
                                               "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")]
    evidence = [Path(source) / "model.py", *[SOURCES / repository / name for name in files]]
    if method in EXTENDED:
        wrapper_file = ("full_anatomy_candidates.py" if method in ("ernet", "fastsurfer", "nobrainer") else
                        "full_graph_candidates.py" if method in ("fbnetgen", "brainnetcnn", "bnt") else "full_preprocessing.py")
        evidence.extend((ROOT / "unibrain_agent" / wrapper_file, ROOT / "unibrain_agent/full_connectivity.py"))
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
    elif method in ("voxelmorph", "lapirn"):
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
    elif method == "ernet":
        interface = ("Define exactly two functions in one file: adapt_extraction(core, image, if_train) and "
                     "adapt_registration(core, fixed, moving). The first core is the complete five-stage ERNet extraction module "
                     "and returns (extracted_images, masks), both native lists. The second core is the complete five-stage ERNet "
                     "affine registration module and returns (warped_images, forward_affines, inverse_affines), all native lists. "
                     "All images are [1,1,96,96,96]. Call the respective supplied core and preserve its full native output structure, "
                     "image coordinates and trainability. Do not import, mutate core, read files, execute tools or define helpers.")
        knowledge = ("Both generated functions execute in the same full host. Registration and classification gradients must reach "
                     "both source modules. Core grid properties remain delegated to the unchanged reference registration module.")
    elif method in ("brainnetcnn", "bnt"):
        interface = ("Define only adapt(core, graph, features). Graph is [1,116,116], features is [1,116,128]. "
                     "The reference classifier preserves native source widths and returns [1,2] class scores. "
                     "Pass both host arguments to core and retain its scores and graph gradient. The reference classifier does "
                     "not consume the additional ROI feature argument. Do not import, mutate core, read files, execute tools or define helpers.")
        knowledge = ("The connected host supplies continuous positive structural graph weights. BrainNetCNN uses the declared "
                     "FBNETGEN third-party PyTorch implementation. BNT retains the source 100-node pool, 1024-wide FFNs and frozen centers.")
    elif method == "fbnetgen":
        interface = ("Define only adapt(core, features). Input is the full UniBrain ROI tensor [1,116,128]. "
                     "The reference FBNetGen pairwise MLP returns [1,116,116] directed nonnegative weights. "
                     "Preserve its values, node order and gradients. The host handles continuous graph connectivity. "
                     "Do not import, mutate core, read files, execute tools or define helpers.")
        knowledge = "This reuses the source pairwise MLP on structural ROI embeddings, not the fMRI temporal extractor or its full training recipe."
    else:
        output_contract = ("Return raw AAL logits [1,117,96,96,96], including background. Retain the complete source width "
                           "and the declared new AAL head; do not argmax or reorder atlas labels."
                           if method in ("fastsurfer", "nobrainer") else
                           "Return the reference corrected image [1,1,96,96,96], retaining native spatial coordinates and the declared intensity convention.")
        interface = ("Define only adapt(core, image). Input is [1,1,96,96,96]. core is the full reference stage wrapper whose "
                     "locked architecture and declared MRI adaptation are provided in source evidence. " + output_contract +
                     " Preserve parameter and image gradients. Do not import, mutate core, read files, execute tools or define helpers.")
        knowledge = ("Parcellation uses a new trainable 117-class AAL head, not an official-label conversion; the connected host mixes "
                     "predicted and transferred probabilities equally so classification can train it."
                     if method in ("fastsurfer", "nobrainer") else
                     "The source architecture is retained at the native input geometry. This checks the declared trainable stage, "
                     "not the complete official command-line preprocessing, pretrained outputs or any fMRI recipe.")
    if method == "nnunet" and plans:
        required.append(Path(plans))
        command.extend(("--plans", str(Path(plans).resolve())))
    evaluator = {"cwd": relative(ROOT, output), "command": command, "timeout": 1800}
    if method == "synthstrip":
        evaluator["progress_file"] = "full_checker/validation.json"
    case = {"id": method, "method": method, "stage": stage, "expected_fit": True,
            "applicability_scope": "Declared adapted core contract; not the complete official preprocessing or pretrained application",
            "implementation": "full_generated_adapter", "plans_supplied": plans is not None,
            "source_files": [relative(path, output) for path in evidence],
            "required_files": [relative(path, output) for path in required],
            "interface": interface, "knowledge": knowledge, "max_optimizer_updates": steps,
            "evaluator": evaluator}
    if method == "nnunet" and plans and Path(plans).is_file():
        try:
            configuration = json.loads(Path(plans).read_text(encoding="utf-8"))["configurations"]["3d_fullres"]
            architecture = configuration["architecture"]
            fields = ("n_stages", "features_per_stage", "conv_op", "kernel_sizes", "strides", "n_conv_per_stage",
                      "n_conv_per_stage_decoder", "conv_bias", "norm_op", "norm_op_kwargs", "dropout_op",
                      "dropout_op_kwargs", "nonlin", "nonlin_kwargs")
            case["structured_evidence"] = {"nnunet_architecture": {
                "network_class_name": architecture["network_class_name"],
                "arch_kwargs": {key: value for key, value in architecture["arch_kwargs"].items() if key in fields},
                "patch_size": configuration["patch_size"]}}
        except (OSError, KeyError, TypeError, ValueError):
            case["source_context_error"] = "Could not extract a supported architecture from the nnU-Net plans file"
    return case


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
