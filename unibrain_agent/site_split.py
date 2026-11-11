"""Prepare declared site holdouts without changing the original five folds."""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .data import digest


TEMPLATES = ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")


def identifier(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def partitions(ids, labels, metadata, held_out, seed=17, validation_fraction=0.2):
    ids = [identifier(value) for value in ids]
    labels = np.asarray(labels)
    if len(set(ids)) != len(ids) or len(ids) != len(labels):
        raise ValueError("Master subject IDs must be unique and aligned with labels")
    if not 0 < validation_fraction < 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("A binary diagnosis and a validation fraction between zero and one are required")
    metadata = {key: value.strip() for key, value in metadata.items()}
    held_out = held_out.strip()
    missing = set(ids) - metadata.keys()
    if missing or any(not metadata[value] for value in ids):
        raise ValueError(f"Site metadata is missing for {len(missing)} master subjects")
    sites = np.asarray([metadata[value] for value in ids])
    test = np.flatnonzero(sites == held_out)
    available = np.flatnonzero(sites != held_out)
    if not len(test) or not len(available):
        raise ValueError("The held out site and other sites must both contain subjects")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in (0, 1):
        selected = available[labels[available] == label].copy()
        if len(selected) < 2:
            raise ValueError("Other sites need at least two subjects of each class for train and validation")
        rng.shuffle(selected)
        count = min(len(selected) - 1, max(1, round(len(selected) * validation_fraction)))
        validation.extend(selected[:count].tolist())
        train.extend(selected[count:].tolist())
    result = {"train": np.asarray(sorted(train)), "val": np.asarray(sorted(validation)), "test": test}
    if sorted(np.concatenate(list(result.values())).tolist()) != list(range(len(ids))):
        raise RuntimeError("Site partitions are not disjoint and exhaustive")
    return result


def export_site(master_path, metadata_path, templates, output, held_out, template_origin, template_scope,
                seed=17, validation_fraction=0.2):
    master_path, metadata_path, templates, output = map(Path, (master_path, metadata_path, templates, output))
    if not template_origin.strip() or template_scope not in ("external_public", "source_training_only"):
        raise ValueError("Templates need an explicit origin and external_public or source_training_only scope")
    held_out = held_out.strip()
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve existing data")
    metadata = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "site"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata CSV needs id and site columns; site names are never inferred")
        for row in reader:
            if not row["id"].strip() or row["id"] in metadata or not row["site"].strip():
                raise ValueError("Site metadata contains an empty or duplicate ID or an empty site")
            metadata[row["id"]] = row["site"].strip()
    master = np.load(master_path, mmap_mode="r", allow_pickle=False)
    volumes = ("raw_img_np", "raw_img_brain_mask_np", "raw_img_gm_mask_np", "raw_img_aal_mask_np")
    if not {"id", "label", *volumes}.issubset(master.dtype.names or []):
        raise ValueError("Expected the original UniBrain structured master array")
    if any(master.dtype.fields[field][0].shape != (96, 96, 96) for field in volumes):
        raise ValueError("The full UniBrain interface requires 96 cubed volumes")
    selected = partitions(master["id"], master["label"], metadata, held_out, seed, validation_fraction)
    for name in TEMPLATES:
        array = np.load(templates / name, allow_pickle=False)
        if array.shape != (96, 96, 96) or not np.isfinite(array).all():
            raise ValueError("Invalid template: " + name)
    record = {"scope": "site_holdout", "held_out_site": held_out, "seed": seed,
              "validation_fraction": validation_fraction, "reference_labels": "Automatic tissue and AAL reference labels",
              "sources": {"master_sha256": digest(master_path), "metadata_sha256": digest(metadata_path)},
              "template_provenance": {"origin": template_origin, "scope": template_scope,
                                      "boundary": "User declaration; hashes verify files, not absence of template training leakage"},
              "protocol": "All held out site subjects form test; other sites are stratified by diagnosis into train and validation",
              "splits": {}}
    output.mkdir(parents=True, exist_ok=False)
    for split, indices in selected.items():
        target = output / f"adhd_{split}_sample.npy"
        array = np.lib.format.open_memmap(target, mode="w+", dtype=master.dtype, shape=(len(indices),))
        for position, index in enumerate(indices):
            for field in volumes:
                if not np.isfinite(master[index][field]).all():
                    raise ValueError(f"Nonfinite input at master row {index}, field {field}; incomplete output retained")
            array[position] = master[index]
        array.flush()
        del array
        ids = [identifier(value) for value in master["id"][indices]]
        np.save(output / f"{split}_indices.npy", indices)
        record["splits"][split] = {"count": len(indices), "master_indices": indices.tolist(), "ids": ids,
                                   "site_counts": dict(Counter(metadata[value] for value in ids)),
                                   "class_counts": dict(Counter(str(int(x)) for x in master["label"][indices])),
                                   "sha256": digest(target)}
    record["templates"] = {}
    for name in TEMPLATES:
        shutil.copyfile(templates / name, output / name)
        record["templates"][name] = digest(output / name)
    (output / "cohort.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
