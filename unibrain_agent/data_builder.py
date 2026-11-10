import json
import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .data import FIELDS, digest, load_split, templates
from .data_contract import model_contract

TEMPLATES = {"image": "template_img_orig_96.npy", "tissue": "template_img_gm_mask_orig_96.npy",
             "atlas": "template_img_aal_mask_orig_96.npy"}
INPUTS = dict(zip(("image", "brain", "tissue", "atlas"), FIELDS[2:]))


def audit_dataset(folder):
    folder = Path(folder)
    templates(folder)
    vocabulary = np.unique(np.load(folder / TEMPLATES["atlas"], allow_pickle=False))
    seen, images, splits, warnings = {}, {}, {}, []
    for split in ("train", "val", "test"):
        array = load_split(folder, split)
        for row in array:
            identifier = str(row["id"])
            if identifier in seen:
                raise ValueError("Subject identity overlaps between dataset splits")
            seen[identifier] = split
            if not np.isin(np.unique(row[FIELDS[-1]]), vocabulary).all():
                raise ValueError("Sample atlas labels differ from the template vocabulary")
            fingerprint = hashlib.sha256(np.asarray(row["raw_img_np"], dtype=np.float32).tobytes()).hexdigest()
            if fingerprint in images and images[fingerprint] != split:
                raise ValueError("Identical image content occurs in different splits")
            images[fingerprint] = split
        splits[split] = {"count": len(array), "classes": dict(Counter(map(str, array["label"].tolist())))}
        if split == "train" and len(splits[split]["classes"]) < 2:
            warnings.append("The training split has one diagnosis class; classification performance cannot be established.")
    return {"passed": True, "splits": splits, "warnings": warnings,
            "checks": ["Structured fields and 96 cubed finite arrays", "Diagnosis and mask label ranges",
                       "Background and 116 template atlas labels", "Sample and template atlas vocabulary",
                       "Disjoint subject IDs", "No identical image content across splits"],
            "files": {p.name: digest(p) for p in sorted(folder.glob("*.npy"))},
            "physical_geometry": "See geometry.json for converted data; existing NPY arrays contain no affine."}


def image_info(path):
    import nibabel as nib
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError("Expected a 3D volume, not a time series or singleton axis")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid spatial affine")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Explicit millimetre spatial units are required")
    return image


def inspect_sources(paths):
    items = []
    for entry in paths:
        root = Path(entry).expanduser().resolve()
        if not root.exists():
            items.append({"path": str(root), "kind": "missing", "status": "blocked", "reason": "Path does not exist on the backend"})
            continue
        if root.is_dir() and (root / "adhd_train_sample.npy").exists():
            try:
                audit = audit_dataset(root)
                items.append({"path": str(root), "kind": "UniBrain NPY", "status": "ready", "audit": audit})
            except (ValueError, OSError) as error:
                items.append({"path": str(root), "kind": "UniBrain NPY", "status": "blocked", "reason": str(error)})
            continue
        files = root.rglob("*") if root.is_dir() else [root]
        for index, path in enumerate(files):
            if index >= 5000:
                items.append({"path": str(root), "kind": "scan limit", "status": "blocked", "reason": "Split this source into smaller folders"})
                break
            if not path.is_file():
                continue
            if path.name.endswith((".nii", ".nii.gz")):
                try:
                    volume = image_info(path)
                    candidate = "t1w" in path.name.lower()
                    items.append({"path": str(path), "kind": "NIfTI", "status": "candidate" if candidate else "review",
                                  "shape": list(volume.shape), "reason": "T1w filename candidate; confirm modality and target mapping" if candidate else "Confirm modality in the manifest"})
                except (ValueError, OSError) as error:
                    items.append({"path": str(path), "kind": "NIfTI", "status": "blocked", "reason": str(error)})
            elif path.suffix.lower() in {".dcm", ".ima"}:
                if not any(item.get("path") == str(path.parent) for item in items):
                    items.append({"path": str(path.parent), "kind": "DICOM series", "status": "review",
                                  "reason": "Select one T1 series per manifest entry; dcm2niix is required for conversion"})
    return {"items": items, "counts": dict(Counter(item["status"] for item in items)),
            "dcm2niix_available": bool(shutil.which("dcm2niix")),
            "scope": "Header and filename triage; anatomical quality and modality still require review."}


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("subjects"), list):
        raise ValueError("Use a version 1 data manifest with a subjects list")
    if value.get("intensity", "preserve") not in {"preserve", "brain_zscore"}:
        raise ValueError("Unknown intensity policy")
    labels = value.get("labels", {})
    if (set(labels.get("diagnosis", {})) != {"0", "1"} or set(labels.get("tissue", {})) != {"0", "1", "2", "3"}
            or labels.get("atlas") != "AAL116"):
        raise ValueError("Declare diagnosis and tissue label meanings, and the AAL116 atlas convention")
    for section in ("diagnosis", "tissue"):
        if any(not isinstance(name, str) or not name.strip() for name in labels[section].values()):
            raise ValueError("Label meanings cannot be empty")
    def resolve(item):
        result = dict(item)
        for key in (*INPUTS, "dicom"):
            if result.get(key):
                p = Path(result[key]).expanduser()
                result[key] = str((p if p.is_absolute() else path.parent / p).resolve())
        return result
    value["subjects"] = [resolve(row) for row in value["subjects"]]
    value["templates"] = resolve(value.get("templates", {}))
    return value


def eligible_subjects(manifest):
    accepted, rejected, seen = [], [], set()
    for row in manifest["subjects"]:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128:
            raise ValueError("Supply a stable subject ID of at most 128 characters")
        if identifier in seen:
            raise ValueError("Repeated subject ID: resolve repeated visits and split leakage before building")
        seen.add(identifier)
        reasons = []
        if row.get("include", True) is not True:
            reasons.append("Excluded by the manifest")
        if row.get("modality") != "T1w":
            reasons.append("This model requires confirmed T1w imaging")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            reasons.append("Missing or invalid diagnosis target")
        if row.get("split") not in {"train", "val", "test"}:
            reasons.append("Missing subject level split assignment")
        if row.get("supervision") not in {"manual", "pseudo", "mixed"}:
            reasons.append("Declare manual, pseudo or mixed supervision provenance")
        for key in INPUTS:
            supplied = row.get("dicom") if key == "image" and not row.get("image") else row.get(key)
            if not supplied or not Path(supplied).exists():
                reasons.append("Missing " + key)
        if reasons:
            rejected.append({"id": identifier, "reasons": reasons})
        else:
            accepted.append(row)
    return accepted, rejected


def convert_dicom(path, folder):
    executable = shutil.which("dcm2niix")
    if not executable:
        raise ValueError("Install dcm2niix on the backend before converting DICOM")
    folder.mkdir(parents=True)
    result = subprocess.run([executable, "-b", "n", "-ba", "y", "-z", "y", "-f", "image", "-o", str(folder), str(path)],
                            capture_output=True, timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    images = list(folder.glob("*.nii*"))
    if result.returncode or len(images) != 1:
        raise ValueError("DICOM conversion requires exactly one selected series and one output volume")
    return images[0]


def resample_bundle(row, work, intensity="preserve"):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    source = Path(row["image"]) if row.get("image") else convert_dicom(row["dicom"], work / "dicom")
    original = image_info(source)
    reference = nib.as_closest_canonical(original)
    scale = np.eye(4)
    scale[:3, :3] = np.diag((np.array(reference.shape) - 1) / 95)
    affine = reference.affine @ scale
    target = ((96, 96, 96), affine)
    arrays, provenance = {}, {}
    for key in INPUTS:
        if not row.get(key) and key != "image":
            continue
        path = source if key == "image" else Path(row[key])
        image = original if key == "image" else image_info(path)
        raw = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite image or annotation values")
        if key != "image" and (np.any(raw < 0) or not np.equal(raw, np.rint(raw)).all()):
            raise ValueError("Annotations must use nonnegative integer labels")
        sampled = resample_from_to(image, target, order=1 if key == "image" else 0)
        arrays[key] = sampled.get_fdata(dtype=np.float32)
        provenance[key] = {"source": str(path), "sha256": digest(path), "shape": list(image.shape),
                           "affine": image.affine.tolist(), "interpolation": "linear" if key == "image" else "nearest"}
    if np.ptp(arrays["image"]) <= 1e-7:
        raise ValueError("The resampled image is constant")
    for key in ("brain", "tissue", "atlas"):
        if key in arrays and np.count_nonzero(arrays[key]) == 0:
            raise ValueError("Empty " + key + " target after spatial resampling; check geometry")
    if "brain" in arrays and not np.isin(arrays["brain"], [0, 1]).all():
        raise ValueError("Brain mask must be binary")
    if "tissue" in arrays and not np.isin(arrays["tissue"], [0, 1, 2, 3]).all():
        raise ValueError("Tissue target must use labels 0 through 3")
    qc = {}
    if "brain" in arrays:
        mask = arrays["brain"] > 0
        bounds = np.array(np.where(mask))
        qc["brain_fraction"] = float(mask.mean())
        qc["brain_bounds_voxels"] = [bounds.min(axis=1).tolist(), bounds.max(axis=1).tolist()]
        for key in ("tissue", "atlas"):
            if key in arrays:
                ratio = float(mask[arrays[key] > 0].mean())
                qc[key + "_inside_brain"] = ratio
                if ratio < .5:
                    raise ValueError("Most " + key + " annotations lie outside the brain mask; review alignment")
    if intensity == "brain_zscore":
        if "brain" not in arrays:
            raise ValueError("Brain z score normalization requires a template brain mask too")
        values = arrays["image"][arrays["brain"] > 0]
        if values.std() <= 1e-7:
            raise ValueError("Brain intensities have zero variance")
        arrays["image"] = (arrays["image"] - values.mean()) / values.std()
    return arrays, {"target_affine": affine.tolist(), "source_files": provenance, "intensity": intensity, "qc": qc,
                    "operation": "Canonical orientation and physical FOV resampling; no anatomical registration"}


def build_dataset(manifest, destination, emit=lambda message: None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    accepted, rejected = eligible_subjects(manifest)
    if {row["split"] for row in accepted} != {"train", "val", "test"}:
        raise ValueError("Every split needs an eligible subject; review missing targets and exclusions")
    intensity = manifest.get("intensity", "preserve")
    template_spec = manifest["templates"]
    if any(not template_spec.get(key) for key in TEMPLATES):
        raise ValueError("Provide image, tissue and AAL116 template volumes")
    emit("Resampling the templates in physical coordinates")
    template, geometry = resample_bundle(template_spec, destination / "template_work", intensity)
    for key, name in TEMPLATES.items():
        np.save(destination / name, template[key], allow_pickle=False)
    templates(destination)
    vocabulary = np.unique(template["atlas"])
    dtype = np.dtype([("id", "U128"), ("label", "i8"), *[(key, "f4", (96, 96, 96)) for key in FIELDS[2:]]])
    records = []
    for split in ("train", "val", "test"):
        rows = [row for row in accepted if row["split"] == split]
        array = np.lib.format.open_memmap(destination / f"adhd_{split}_sample.npy", mode="w+", dtype=dtype, shape=(len(rows),))
        for index, row in enumerate(rows):
            emit(f"Converting {split} volume {index + 1}/{len(rows)}")
            arrays, info = resample_bundle(row, destination / f"{split}_{index}", intensity)
            if not np.isin(arrays["atlas"], vocabulary).all():
                raise ValueError("Subject atlas contains labels absent from the template")
            array[index]["id"], array[index]["label"] = row["id"], row["label"]
            for key, field in INPUTS.items():
                array[index][field] = arrays[key]
            records.append({"id": row["id"], "split": split, "supervision": row["supervision"], **info})
        array.flush()
        del array
    (destination / "geometry.json").write_text(json.dumps({"template": geometry, "subjects": records}, indent=2), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "excluded.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    emit("Checking target labels, subject separation and duplicate images")
    audit = audit_dataset(destination)
    audit["excluded"] = len(rejected)
    (destination / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
