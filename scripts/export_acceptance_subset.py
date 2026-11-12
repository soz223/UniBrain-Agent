"""Export an auditable acceptance subset of an unchanged original fold."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


SPLITS = ("train", "val", "test")
VOLUMES = ("raw_img_np", "raw_img_brain_mask_np", "raw_img_gm_mask_np", "raw_img_aal_mask_np")
TEMPLATES = {"template_img_orig_96.npy": "mni_icbm152_t1_tal_nlin_asym_09c_brain_orig_96.nii",
             "template_img_gm_mask_orig_96.npy": "mni_icbm152_csf_gm_wm_tal_nlin_asym_09c_orig_96.nii",
             "template_img_aal_mask_orig_96.npy": "aal_resampled_in_mni152_space_orig_96.nii"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identifier(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def export(cohort, output, fold=1, per_class=1, seed=17, dataset=None, full=False):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if dataset not in {"abide", "adhd"}:
        raise ValueError("Declare the source dataset explicitly")
    if type(fold) is not int or fold not in range(1, 6) or type(per_class) is not int or per_class < 1:
        raise ValueError("A preserved fold from one to five and a positive sample budget are required")
    if output.exists() or output == cohort or cohort in output.parents:
        raise ValueError("Use a new output directory outside the original cohort")
    master_path = cohort / "unibrain_master.npy"
    initial_stat = master_path.stat()
    master = np.load(master_path, mmap_mode="r", allow_pickle=False)
    if master.ndim != 1 or master.dtype.names != ("id", "label", *VOLUMES):
        raise ValueError("Expected the original structured UniBrain master")
    if any(master.dtype.fields[field][0].shape != (96, 96, 96) for field in VOLUMES):
        raise ValueError("All image and reference arrays must have shape 96 cubed")
    ids = [identifier(value) for value in master["id"]]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids) or not np.isin(master["label"], (0, 1)).all():
        raise ValueError("Master subjects must have unique IDs and binary diagnosis labels")
    source_paths = {split: cohort / "rounds" / f"round{fold}" / f"{split}_indices.npy" for split in SPLITS}
    indices = {split: np.load(path, allow_pickle=False) for split, path in source_paths.items()}
    if any(value.ndim != 1 or not np.issubdtype(value.dtype, np.integer) for value in indices.values()):
        raise ValueError("Original fold indices must be one dimensional integer arrays")
    flat = np.concatenate(list(indices.values()))
    if sorted(flat.tolist()) != list(range(len(master))):
        raise ValueError("Original fold indices must partition the master exactly")
    selected = {}
    for split, pool in indices.items():
        if full:
            if not len(pool):
                raise ValueError("Full fold splits must be nonempty")
            selected[split] = pool.tolist()
            continue
        rng = np.random.default_rng(seed)
        chosen = []
        for label in (0, 1):
            candidates = pool[master["label"][pool] == label].copy()
            if len(candidates) < per_class:
                raise ValueError("Insufficient subjects of a class in " + split)
            rng.shuffle(candidates)
            chosen.extend(candidates[:per_class].tolist())
        selected[split] = sorted(chosen)
    template_arrays, template_records = {}, {}
    first_affine = None
    for target, source in TEMPLATES.items():
        path = cohort / "templates/original_96" / source
        image = nib.load(path)
        array = np.asarray(image.dataobj)
        if (array.shape != (96, 96, 96) or not np.isfinite(array).all()
                or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12):
            raise ValueError("Invalid original template")
        if first_affine is not None and not np.allclose(first_affine, image.affine, rtol=0, atol=1e-5):
            raise ValueError("Original template grids do not agree")
        first_affine = image.affine
        template_arrays[target] = array
        template_records[target] = {"source_sha256": sha(path), "affine": image.affine.tolist()}
    if full:
        parent = output.parent
        while not parent.exists():
            parent = parent.parent
        required = master.dtype.itemsize * len(master) + sum(array.nbytes for array in template_arrays.values()) + 10 * 1024 ** 3
        if shutil.disk_usage(parent).free < required:
            raise ValueError("Insufficient disk space for the complete fold export and 10 GiB reserve")
    report = {"version": 2, "status": "exporting", "scope": "full_fold" if full else "acceptance_subset_not_full_cohort",
              "dataset": dataset, "fold": fold, "seed": seed, "per_class": None if full else per_class,
              "master_count": len(master), "master_size_bytes": initial_stat.st_size,
              "master_sha256": sha(master_path), "source_data_modified": False, "splits": {}, "templates": {}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_indices").mkdir()
    try:
        for split, chosen in selected.items():
            copied = output / "source_indices" / source_paths[split].name
            shutil.copyfile(source_paths[split], copied)
            index_hash = sha(source_paths[split])
            if sha(copied) != index_hash or not np.array_equal(np.load(copied, allow_pickle=False), indices[split]):
                raise ValueError("Original fold indices changed during export")
            target = output / f"adhd_{split}_sample.npy"
            result = np.lib.format.open_memmap(target, mode="w+", dtype=master.dtype, shape=(len(chosen),))
            try:
                for position, index in enumerate(chosen):
                    if any(not np.isfinite(master[index][field]).all() for field in VOLUMES):
                        raise ValueError("A selected master row contains nonfinite imaging data")
                    result[position] = master[index]
            finally:
                result.flush()
                del result
            report["splits"][split] = {"count": len(chosen), "original_count": len(indices[split]),
                                       "master_indices": chosen, "ids": [ids[index] for index in chosen],
                                       "class_counts": {str(label): int(np.count_nonzero(master["label"][chosen] == label)) for label in (0, 1)},
                                       "sha256": sha(target), "original_indices_sha256": index_hash,
                                       "original_indices_file": copied.relative_to(output).as_posix()}
        for target, array in template_arrays.items():
            np.save(output / target, array)
            report["templates"][target] = {**template_records[target], "sha256": sha(output / target)}
        final_stat = master_path.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("Master changed during export; retain this failed output for diagnosis")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (output / "cohort.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", choices=("abide", "adhd"), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = export(args.cohort, args.output, args.fold, args.per_class, args.seed, args.dataset)
    print(json.dumps({"status": result["status"], "scope": result["scope"], "master_count": result["master_count"],
                      "splits": {key: value["count"] for key, value in result["splits"].items()}}))
