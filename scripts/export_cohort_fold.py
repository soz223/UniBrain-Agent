"""Export an original UniBrain interface from a master array and fold indices."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def export_fold(cohort, fold, destination, full=False):
    cohort, destination = Path(cohort), Path(destination)
    master = np.load(cohort / "unibrain_master.npy", mmap_mode="r")
    partitions = {split: np.load(cohort / "rounds" / f"round{fold}" / f"{split}_indices.npy")
                  for split in ("train", "val", "test")}
    sets = [set(indices.tolist()) for indices in partitions.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)) or set.union(*sets) != set(range(len(master))):
        raise ValueError("Fold indices must be disjoint and cover the master array")
    destination.mkdir(parents=True, exist_ok=False)
    record = {"source": str(cohort.resolve()), "fold": fold,
              "scope": "full_fold" if full else "smoke_subset", "splits": {}}
    for split, indices in partitions.items():
        selected = indices
        if not full:
            selected = []
            for label in (0, 1):
                candidates = indices[master["label"][indices] == label]
                if not len(candidates):
                    raise ValueError(f"No class {label} in {split}")
                supplemented = candidates[master["id"][candidates] == "sub-0050616"]
                selected.append(int(supplemented[0] if len(supplemented) else candidates[0]))
            selected = np.asarray(selected)
        target = np.lib.format.open_memmap(destination / f"adhd_{split}_sample.npy", mode="w+",
                                           dtype=master.dtype, shape=(len(selected),))
        for offset, index in enumerate(selected):
            target[offset] = master[index]
        target.flush()
        del target
        record["splits"][split] = {"count": len(selected), "original_count": len(indices),
                                   "master_indices": selected.tolist(), "ids": master["id"][selected].tolist()}
    templates = {"template_img_orig_96.npy": "mni_icbm152_t1_tal_nlin_asym_09c_brain_orig_96.nii",
                 "template_img_gm_mask_orig_96.npy": "mni_icbm152_csf_gm_wm_tal_nlin_asym_09c_orig_96.nii",
                 "template_img_aal_mask_orig_96.npy": "aal_resampled_in_mni152_space_orig_96.nii"}
    for target, source in templates.items():
        array = np.asarray(nib.load(cohort / "templates" / "original_96" / source).dataobj)
        if array.shape != (96, 96, 96) or not np.isfinite(array).all():
            raise ValueError("Invalid original template")
        np.save(destination / target, array)
    (destination / "cohort.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_fold(args.cohort, args.fold, args.destination, args.full), indent=2))
