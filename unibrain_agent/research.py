import json

from .paths import UPSTREAM, CATALOG
from .upstream import SYNTHSTRIP_COMMIT, UNIBRAIN_COMMIT, synthstrip_core, unibrain_core


def evidence_bundle():
    root = "https://github.com/Anonymous7852/UniBrain/blob/" + UNIBRAIN_COMMIT
    cards = [
        {"id": "unibrain", "stage": "extraction", "available": True,
         "publication": "UniBrain, KDD 2024 (project reference)",
         "code": root + "/model.py", "interface": "Volume -> soft mask and extracted volume",
         "conditions": ["Native 96 cubed input", "Random initialization in this demo"],
         "adaptation": "Original multi_stage_ext with compact channel widths"},
        {"id": "synthstrip", "stage": "extraction", "available": True,
         "publication": "SynthStrip, NeuroImage 2022", "paper": "https://arxiv.org/abs/2203.09974",
         "code": f"https://github.com/freesurfer/freesurfer/blob/{SYNTHSTRIP_COMMIT}/mri_synthstrip/mri_synthstrip",
         "interface": "Volume -> scalar field -> sigmoid(-field) mask -> extracted volume",
         "conditions": ["Soft-mask adaptation replaces hard threshold and connected-component processing",
                        "Compact 5-level, 4-to-16-channel network; new random weights",
                        "Mask BCE or BCE plus Dice training; this does not reproduce the published SDT training",
                        "No FreeSurfer conforming or pretrained accuracy is claimed"],
         "adaptation": "Original StripModel and ConvBlock, tensor-only mask adapter"},
        {"id": "soft_atlas", "stage": "parcellation", "available": True,
         "publication": "Local UniBrain adapter", "code": root + "/model.py#L559",
         "interface": "117-channel one-hot atlas + inverse affine grid -> soft ROI probabilities",
         "conditions": ["Fixed atlas label order", "Trilinear sampling in the same normalized grid",
                        "Border padding preserves probability mass; coverage is checked separately"],
         "adaptation": "One-hot before interpolation instead of nearest label sampling followed by hard one-hot"},
        {"id": "continuous_edges", "stage": "network generation", "available": True,
         "publication": "Local UniBrain adapter", "code": root + "/model.py#L582",
         "interface": "ROI features -> cosine adjacency -> original GCN classifier",
         "conditions": ["Positive-edge topology remains piecewise fixed in the original GCN",
                        "Feature noise retained; discrete binary edge reconstruction disabled"],
         "adaptation": "Retain learned feature-dependent edge values during training"}]
    archive = CATALOG / "research.json"
    archived = json.loads(archive.read_text(encoding="utf-8"))["records"] if archive.exists() else []
    other = [{"name": item["publication"].split(" (")[0], "publication": item["publication"], "stages": item["stages"],
              "paper": item["paper"], "code": item["code"], "available": False,
              "reason": "Choose a stage tool in the pipeline editor for a current composition check"}
             for item in archived]
    source = (UPSTREAM / "model.py").read_text(encoding="utf-8-sig").splitlines()
    return {"mode": "Review of source-pinned modules and the existing literature catalog",
            "open_web_search": False, "cards": cards, "other_candidates": other,
            "source_evidence": {"unibrain": unibrain_core().evidence, "synthstrip": synthstrip_core().evidence},
            "connection_code": "\n".join(source[548:590]),
            "boundary": "Published papers motivate candidates; local execution decides integration feasibility"}
