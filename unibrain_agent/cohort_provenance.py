"""Validate portable cohort exports against their original fold membership."""

import hashlib
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
TEMPLATES = ("template_img_orig_96.npy", "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy")
SCOPES = {"acceptance_subset_not_full_cohort", "full_fold"}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def validate_cohort(directory, dataset, fold, data_hashes=None, required_scope=None, allow_missing=True):
    directory = Path(directory).resolve()
    path = directory / "cohort.json"
    if not path.is_file():
        if not allow_missing or required_scope:
            raise ValueError("An exported cohort manifest is required")
        return {"status": "unverified", "scope": "unverified_public_sample_or_external_data",
                "reason": "No cohort manifest; original fold provenance is not established", "array_hashes_verified": False}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("version") != 2 or record.get("status") != "complete" or record.get("scope") not in SCOPES:
        raise ValueError("Use a complete version 2 cohort export with original indices and template hashes")
    scope = record["scope"]
    if required_scope and scope != required_scope:
        raise ValueError("The cohort export scope does not match this experiment")
    if record.get("dataset") != dataset or record.get("fold") != fold or not checksum(record.get("master_sha256")):
        raise ValueError("Cohort source dataset, fold or master checksum is missing or inconsistent")
    if (type(record.get("master_count")) is not int or record["master_count"] < 1
            or type(record.get("master_size_bytes")) is not int or record["master_size_bytes"] < 1
            or record.get("source_data_modified") is not False):
        raise ValueError("Master provenance is incomplete")
    originals, selected, subjects, declared, index_hashes, counts = [], [], [], {}, {}, {}
    for split in SPLITS:
        details = record["splits"][split]
        relative = f"source_indices/{split}_indices.npy"
        index_path = directory / relative
        if details.get("original_indices_file") != relative or directory not in index_path.resolve().parents:
            raise ValueError("Original indices must be retained inside the exported cohort")
        index_hash = file_hash(index_path)
        if index_hash != details.get("original_indices_sha256"):
            raise ValueError("Original fold index checksum mismatch")
        original = np.load(index_path, allow_pickle=False)
        chosen = details.get("master_indices", [])
        if (original.ndim != 1 or not np.issubdtype(original.dtype, np.integer)
                or details.get("original_count") != len(original) or not chosen
                or any(type(value) is not int for value in chosen) or len(set(chosen)) != len(chosen)
                or not set(chosen).issubset(set(original.tolist()))):
            raise ValueError("Selected subjects do not preserve original fold membership")
        if scope == "full_fold" and chosen != original.tolist():
            raise ValueError("A full fold must preserve every original index in its original order")
        name = f"adhd_{split}_sample.npy"
        if not checksum(details.get("sha256")):
            raise ValueError("Exported array checksum is missing")
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        volumes = ("raw_img_np", "raw_img_brain_mask_np", "raw_img_gm_mask_np", "raw_img_aal_mask_np")
        if (array.ndim != 1 or array.dtype.names != ("id", "label", *volumes)
                or any(array.dtype.fields[field][0].shape != (96, 96, 96) for field in volumes)):
            raise ValueError("Exported split must retain subject IDs, labels and full 96 cubed imaging arrays")
        ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in array["id"]]
        if (len(array) != details.get("count") or len(array) != len(chosen) or ids != details.get("ids")
                or len(set(ids)) != len(ids) or any(not value.strip() for value in ids)
                or not np.isin(array["label"], (0, 1)).all()):
            raise ValueError("Exported rows do not match their cohort manifest")
        class_counts = {str(label): int(np.count_nonzero(array["label"] == label)) for label in (0, 1)}
        if class_counts != details.get("class_counts"):
            raise ValueError("Class membership does not match the cohort manifest")
        if scope == "acceptance_subset_not_full_cohort" and (type(record.get("per_class")) is not int
                or record["per_class"] < 1 or any(value != record["per_class"] for value in class_counts.values())):
            raise ValueError("Acceptance class membership does not match its declared budget")
        del array
        declared[name], index_hashes[split], counts[split] = details["sha256"], index_hash, len(chosen)
        originals.extend(original.tolist())
        selected.extend(chosen)
        subjects.extend(ids)
    if (sorted(originals) != list(range(record["master_count"])) or len(set(selected)) != len(selected)
            or len(set(subjects)) != len(subjects)):
        raise ValueError("Original or exported splits overlap or do not partition the declared master")
    first_affine = None
    for name in TEMPLATES:
        details = record["templates"][name]
        if not checksum(details.get("sha256")) or not checksum(details.get("source_sha256")):
            raise ValueError("Template checksum or source provenance is missing")
        try:
            affine = np.asarray(details.get("affine"), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Template grid provenance must be numeric") from error
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or abs(np.linalg.det(affine[:3, :3])) < 1e-12
                or first_affine is not None and not np.allclose(first_affine, affine, rtol=0, atol=1e-5)):
            raise ValueError("Template grid provenance is invalid or inconsistent")
        first_affine = affine
        declared[name] = details["sha256"]
    if data_hashes is not None and any(data_hashes.get(name) != expected for name, expected in declared.items()):
        raise ValueError("Exported data or template checksum mismatch")
    return {"status": "verified" if data_hashes is not None else "metadata_verified", "scope": scope,
            "array_hashes_verified": data_hashes is not None, "dataset": dataset, "fold": fold,
            "manifest_sha256": file_hash(path), "master_sha256": record["master_sha256"],
            "original_index_hashes": index_hashes, "split_counts": counts, "declared_data_hashes": declared,
            "data_hashes": data_hashes,
            "boundary": "Checksum agreement and preserved fold membership do not establish clinical label accuracy"}


def validate_site(directory, dataset, split_id, data_hashes=None):
    directory = Path(directory).resolve()
    path = directory / "site_split.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("version") != 1 or record.get("status") != "complete" or record.get("scope") != "site_holdout"
            or record.get("dataset") != dataset or record.get("split_id") != split_id
            or record.get("held_out_site") != split_id or not isinstance(split_id, str) or not split_id.strip()):
        raise ValueError("A complete site export with the declared dataset and split ID is required")
    if record.get("membership_file") != "site_membership.json" or record.get("metadata_file") != "source_site_metadata.csv":
        raise ValueError("Site membership and source metadata must be retained inside the export")
    sources = record.get("sources", {})
    if not checksum(sources.get("master_sha256")) or file_hash(directory / "source_site_metadata.csv") != sources.get("metadata_sha256"):
        raise ValueError("Site master or source metadata checksum is invalid")
    if file_hash(directory / "site_membership.json") != record.get("membership_sha256"):
        raise ValueError("Site membership checksum mismatch")
    metadata = {}
    with (directory / "source_site_metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "site"}.issubset(reader.fieldnames or []):
            raise ValueError("Retained site metadata lacks id and site columns")
        for row in reader:
            if not row["id"].strip() or row["id"] in metadata or not row["site"].strip():
                raise ValueError("Retained site metadata contains invalid or duplicate rows")
            metadata[row["id"]] = row["site"].strip()
    membership = json.loads((directory / "site_membership.json").read_text(encoding="utf-8"))
    if not membership or len(membership) != record.get("master_count"):
        raise ValueError("Site membership does not cover the declared master")
    master_ids = [row["id"] for row in membership]
    if len(set(master_ids)) != len(master_ids) or any(metadata.get(row["id"]) != row["site"] for row in membership):
        raise ValueError("Site membership disagrees with the source metadata")
    declared, all_indices, all_ids, counts, test_ids = {}, [], [], {}, set()
    for split in SPLITS:
        details = record["splits"][split]
        relative = f"{split}_indices.npy"
        if details.get("indices_file") != relative or file_hash(directory / relative) != details.get("indices_sha256"):
            raise ValueError("Site split index checksum mismatch")
        indices = np.load(directory / relative, allow_pickle=False)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices)
                or indices.tolist() != details.get("master_indices") or len(indices) != details.get("count")
                or np.any(indices < 0) or np.any(indices >= len(membership))):
            raise ValueError("Site indices do not match the declared membership")
        ids = [master_ids[index] for index in indices]
        sites = Counter(metadata[value] for value in ids)
        if ids != details.get("ids") or dict(sites) != details.get("site_counts"):
            raise ValueError("Site subject IDs or site counts disagree")
        if split == "test":
            if set(sites) != {split_id}:
                raise ValueError("Test must contain only the held out site")
            test_ids = set(ids)
        elif split_id in sites:
            raise ValueError("Held out site leaked into training or validation")
        name = f"adhd_{split}_sample.npy"
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        volumes = ("raw_img_np", "raw_img_brain_mask_np", "raw_img_gm_mask_np", "raw_img_aal_mask_np")
        if (array.ndim != 1 or len(array) != len(ids) or array.dtype.names != ("id", "label", *volumes)
                or any(array.dtype.fields[field][0].shape != (96, 96, 96) for field in volumes)):
            raise ValueError("Site arrays must preserve complete 96 cubed imaging rows")
        observed = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in array["id"]]
        classes = dict(Counter(str(int(value)) for value in array["label"]))
        if observed != ids or classes != details.get("class_counts") or not np.isin(array["label"], (0, 1)).all():
            raise ValueError("Site array subjects or diagnosis labels disagree")
        if split != "test" and set(classes) != {"0", "1"}:
            raise ValueError("Site training and validation require both diagnosis classes")
        del array
        if not checksum(details.get("sha256")):
            raise ValueError("Site data checksum is missing")
        declared[name], counts[split] = details["sha256"], len(ids)
        all_indices.extend(indices.tolist())
        all_ids.extend(ids)
    if (sorted(all_indices) != list(range(len(membership))) or len(set(all_ids)) != len(all_ids)
            or test_ids != {row["id"] for row in membership if row["site"] == split_id}):
        raise ValueError("Site splits overlap, omit subjects or lose held out site members")
    origin = record.get("template_provenance", {})
    if not str(origin.get("origin", "")).strip() or origin.get("scope") not in {"external_public", "source_training_only"}:
        raise ValueError("Site templates need declared leakage controls and provenance")
    for name in TEMPLATES:
        expected = record.get("templates", {}).get(name)
        if not checksum(expected):
            raise ValueError("Site template checksum is missing")
        template = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        if template.shape != (96, 96, 96) or not np.isfinite(template).all():
            raise ValueError("Site templates must contain finite 96 cubed volumes")
        del template
        declared[name] = expected
    if data_hashes is not None and any(data_hashes.get(name) != expected for name, expected in declared.items()):
        raise ValueError("Site data or template checksum mismatch")
    return {"status": "verified" if data_hashes is not None else "metadata_verified", "scope": "site_holdout",
            "dataset": dataset, "fold": None, "split_id": split_id, "manifest_sha256": file_hash(path),
            "master_sha256": sources["master_sha256"], "metadata_sha256": sources["metadata_sha256"],
            "split_counts": counts, "declared_data_hashes": declared, "data_hashes": data_hashes,
            "array_hashes_verified": data_hashes is not None,
            "boundary": "Site isolation is verified; template training provenance remains a declared constraint"}
