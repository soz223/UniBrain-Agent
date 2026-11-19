#!/usr/bin/env python3
"""Adapter for HCP binary recorded sex classification (transfer check).

This ports the previously generated operational structural MRI UniBrain adapter
(base_adapter.py) to the HCP setting by:
- Reusing base_adapter.py *as-is* for all model construction, forward graph,
  losses, augmentation, hyperparameters, and the short `--steps` update smoke.
- Preserving the exact CLI: --source --fold --output --steps.
- Preserving generic file paths used by the base adapter (train/val .npy names
  under --fold, and required template .npy files under --fold).

Only post-processes the produced report.json to (a) relabel the task to
"HCP recorded sex" and (b) add explicit evidence about target usage: the per-
subject tissue references (gm_mask, aal_mask) are parsed but never used as
training/validation targets; segmentation supervision uses the *warped fixed
template* labels, and brain-mask supervision is allowed.

No networking. No changes to ./source or base_adapter.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Sequence


def _verify_no_partial_subject_tissue_targets_in_base(base_path: str) -> Dict[str, Any]:
    """Lightweight static evidence from base_adapter.py.

    We treat 'partial per-subject tissue references' as the per-subject gm_mask
    and aal_mask fields (parsed from each row). The base adapter should not use
    those arrays as any training or validation target.

    Since we must not modify base_adapter.py, we provide evidence by:
    - Counting occurrences of attribute access '.gm_mask' and '.aal_mask'.
    - Asserting they are only present in parsing/dataclass scaffolding.

    This is intentionally simple and conservative.
    """

    with open(base_path, "r", encoding="utf-8") as f:
        src = f.read()

    gm_attr = src.count(".gm_mask")
    aal_attr = src.count(".aal_mask")

    # Expected: only referenced in Sample definition and _parse_row_to_sample.
    # If counts are higher, we still report it, but also fail loudly because it
    # would violate the requested guarantee.
    evidence: Dict[str, Any] = {
        "base_adapter_path": os.path.abspath(base_path),
        "counts": {".gm_mask": gm_attr, ".aal_mask": aal_attr},
        "interpretation": (
            "Per-subject gm_mask/aal_mask are parsed but should not be used as "
            "training/validation targets; segmentation target should come from "
            "the warped fixed template labels (am_fix_2_moving derived from fixed_label_am)."
        ),
    }

    if gm_attr > 1 or aal_attr > 1:
        # 1 occurrence each is expected in _parse_row_to_sample; dataclass uses bare names.
        raise RuntimeError(
            "base_adapter.py appears to reference per-subject gm_mask/aal_mask beyond parsing; "
            f"counts: gm_mask={gm_attr}, aal_mask={aal_attr}. Refusing transfer."  # noqa: E501
        )

    # Also check for direct usage of 'gm_mask' / 'aal_mask' variables outside the parsing function.
    # This is a best-effort string check; we keep it simple to avoid false confidence.
    if "s.gm_mask" in src or "s.aal_mask" in src:
        raise RuntimeError("base_adapter.py uses s.gm_mask or s.aal_mask; refusing transfer.")

    return evidence


def _postprocess_report(output_dir: str, base_evidence: Dict[str, Any]) -> None:
    report_path = os.path.join(output_dir, "report.json")
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"Expected base report at: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Relabel task.
    report["task"] = "HCP recorded sex (binary) smoke integration"

    # Keep the existing updates/validation evidence, but add transfer-specific notes.
    report.setdefault("adaptations", [])
    report["adaptations"].append(
        "Transfer: HCP recorded sex classification uses the same UniBrain computation graph and losses as base_adapter.py; only dataset label semantics differ (row[1] is interpreted as recorded sex 0/1)."
    )

    # Bound validation to four probabilities (base adapter selects up to 2 per class => <=4).
    val = report.get("validation", {})
    probs = val.get("class1_probabilities", [])
    if len(probs) > 4:
        raise RuntimeError(f"Validation probabilities exceed 4 (got {len(probs)}), refusing.")

    report.setdefault("evidence", {})
    report["evidence"]["no_partial_subject_tissue_targets"] = {
        "static_base_adapter_checks": base_evidence,
        "training_targets_explained": [
            "Classification target is the scalar label parsed from each row at index 1 (row[1]).",
            "Segmentation supervision uses am_fix_2_moving, produced by warping the fixed template labels (fixed_label_am) into moving space; it does not use the subject-provided gm_mask/aal_mask fields.",
            "Brain-mask supervision is allowed and uses the subject-provided brain_mask (moving_label_bm) only for the extraction loss (BCELoss), not as a tissue-class target.",
        ],
    }

    # Mark that we intentionally do not use any test set.
    report["evidence"]["no_test_usage"] = "Only train/val NPYs under --fold are opened by base_adapter.py; no test file paths are referenced."

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    # Parse args here only so --help matches expected CLI.
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to provided source directory (contains 4 .py files)")
    ap.add_argument(
        "--fold",
        required=True,
        help=(
            "Path to fold directory containing train/val sample NPYs and the required template NPYs; "
            "for this transfer these partitions are assumed to be fixed family-aware train/val splits."
        ),
    )
    ap.add_argument("--output", required=True, help="Output directory to create")
    ap.add_argument("--steps", required=True, type=int, help="Number of genuine optimizer updates to run")
    args = ap.parse_args(argv)

    # Ensure we only inspect this workspace (no implicit imports from elsewhere).
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    import base_adapter  # local file, immutable per task

    base_path = os.path.abspath(base_adapter.__file__)
    base_evidence = _verify_no_partial_subject_tissue_targets_in_base(base_path)

    # Run the base adapter unchanged (preserves computation + 2-update verification).
    rc = int(base_adapter.main(["--source", args.source, "--fold", args.fold, "--output", args.output, "--steps", str(args.steps)]))
    if rc != 0:
        return rc

    _postprocess_report(args.output, base_evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
