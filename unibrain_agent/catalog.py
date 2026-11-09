import json

from .paths import CATALOG

STAGES = [
    ("denoising", "Denoising", "optional", ["none", "M14"]),
    ("bias", "Bias correction", "optional", ["none", "M01"]),
    ("extraction", "Brain extraction", "main", ["unibrain", "M02", "M03"]),
    ("normalization", "Intensity normalization", "optional", ["none", "M04", "M05"]),
    ("registration", "Registration", "main", ["unibrain", "M06", "M07", "M03"]),
    ("segmentation", "Tissue segmentation", "auxiliary", ["unibrain", "M08", "M04"]),
    ("parcellation", "Brain parcellation", "main", ["unibrain", "soft_atlas", "M09", "M10"]),
    ("graph", "Network generation", "main", ["unibrain", "continuous", "M11"]),
    ("classification", "Classification", "main", ["unibrain", "M12", "M13"]),
    ("surface", "Cortical surface", "optional branch", ["none", "M15"]),
]
DEFAULTS = {stage: choices[0] for stage, _, _, choices in STAGES}
ADAPTER_DEFAULTS = {**DEFAULTS, "parcellation": "soft_atlas", "graph": "continuous"}


def original_selection(selection):
    return selection == DEFAULTS


SOURCES = {
    "M01": ["DeepN4"], "M02": ["freesurfer_synthstrip"], "M03": ["reg_ernet"],
    "M04": ["deepNormalize"], "M05": ["HarmonizingFlows"], "M06": ["reg_voxelmorph_master"],
    "M07": ["reg_lapirn"], "M08": ["dynamic-network-architectures"], "M09": ["FastSurfer"],
    "M10": ["nobrainer"], "M11": ["FBNETGEN"], "M12": ["FBNETGEN"],
    "M13": ["BrainNetworkTransformer"], "M14": ["denoise_redwgan"], "M15": ["surface_corticalflow"],
}
ADAPTATIONS = {
    "M01": "Original DeepN4 core at 16 cubed. Bounded log field, tensor interpolation and division restore a 96 cubed output.",
    "M02": "Original SynthStrip core with compact widths. Signed distance output becomes a soft mask. Random weights and mask supervision.",
    "M03": "Original ERNet extraction and registration cores form one coupled choice. Compact widths and an invertible affine adapter.",
    "M04": "Original single channel UNet with compact widths at 32 cubed. Tissue option uses a newly configured four class head and ground truth CE.",
    "M05": "Original 2D Harmonizer at eight 224 by 224 slices. Tensor volume resizing for the CPU profile. No flow training or cross site claims.",
    "M06": "Original VoxelMorph SVF core at 32 cubed after UniBrain affine alignment. Seven integration steps, positive and negative velocity, normalized grid composition.",
    "M07": "Original LapIRN first scale after UniBrain affine alignment. Integrated positive and negative velocity at eight cubed; normalized displacement upsampled to 96 cubed.",
    "M08": "Pinned PlainConvUNet with compact three stage widths. Both native supervision heads enter the tissue loss.",
    "M09": "Original single view FastSurferCNN at 32 cubed with 16 filters. Newly initialized 117 class AAL head; no conversion of pretrained 95 class labels. AAL supervision is mandatory.",
    "M10": "Nobrainer HighResNet at 16 cubed. Newly initialized 117 class AAL head, soft probability interpolation and AAL supervision.",
    "M11": "Original pair MLP with a positive output bias at initialization. Symmetric adjacency adapter and device aware constant buffers. Retrained on sMRI ROI features.",
    "M12": "FBNetGen third party BrainNetCNN for 116 nodes. Dropout follows training mode. Random weights; no Caffe model conversion.",
    "M13": "Original two layer Brain Network Transformer for 116 nodes pooled to eight. Current PyTorch signature adapter and fixed cluster centers.",
    "M14": "Original RED WGAN generator at 16 cubed with tensor resizing to 96 cubed. Downstream task loss trains the generator; no GAN discriminator.",
    "M15": "Requires a template mesh, a physical image affine and a surface target or loss. These inputs are absent from the GitHub NPY examples.",
}


def catalog():
    research = json.loads((CATALOG / "research.json").read_text(encoding="utf-8"))
    lookup = {item["id"]: item for item in research["records"]}
    stages = []
    for key, title, route, choices in STAGES:
        options = []
        for choice in choices:
            record = lookup.get(choice)
            label = {"none": "Skip this optional stage", "unibrain": "UniBrain original core",
                     "soft_atlas": "UniBrain soft atlas transfer", "continuous": "UniBrain continuous cosine graph"}.get(choice)
            if choice == "unibrain" and key == "parcellation":
                label = "UniBrain atlas transfer (nearest labels)"
            elif choice == "unibrain" and key == "graph":
                label = "UniBrain cosine graph (original edge augmentation)"
            description = {"none": "This optional stage is skipped.",
                           "unibrain": "Original upstream component in the full UniBrain profile. Custom compositions use its declared compact adapter.",
                           "soft_atlas": "Differentiable atlas probability sampling for custom compositions.",
                           "continuous": "Continuous cosine adjacency without the original discrete edge perturbation."}
            options.append({"id": choice, "label": label or record["publication"], "research": record,
                            "adaptation": ADAPTATIONS.get(choice, description.get(choice)),
                            "status": "requires_inputs" if choice == "M15" else "build_validation_required",
                            "selectable": True})
        stages.append({"id": key, "name": title, "route": route, "options": options})
    return {"stages": stages, "defaults": DEFAULTS, "adapter_defaults": ADAPTER_DEFAULTS, "workbook": research["source"],
            "workbook_sha256": research["sha256"], "method_count": len(lookup), "pending": research["pending"]}


def validate_selection(selection):
    if not isinstance(selection, dict) or set(selection) != set(DEFAULTS):
        raise ValueError("Select exactly one tool for every stage")
    for key, _, _, choices in STAGES:
        if selection[key] not in choices:
            raise ValueError(f"Unsupported tool for {key}")
    if (selection["extraction"] == "M03") != (selection["registration"] == "M03"):
        raise ValueError("ERNet must occupy extraction and registration together")
    if selection["extraction"] == "M03" and selection["normalization"] != "none":
        raise ValueError("ERNet joins extraction directly to registration. Move normalization before the pair in a future adapter.")
    if selection["surface"] == "M15":
        raise ValueError(ADAPTATIONS["M15"])
    return dict(selection)


def required_sources(selection):
    return sorted({"reg_unibrain"} | {repo for choice in selection.values() for repo in SOURCES.get(choice, [])})
