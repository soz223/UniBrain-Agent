import hashlib
from pathlib import Path

import numpy as np
import torch

FIELDS = ("id", "label", "raw_img_np", "raw_img_brain_mask_np",
          "raw_img_gm_mask_np", "raw_img_aal_mask_np")


class OriginalDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.array = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.array.ndim != 1 or self.array.dtype.names != FIELDS or not len(self.array):
            raise ValueError("Expected a nonempty UniBrain structured array")
        if any(self.array.dtype[name].shape != (96, 96, 96) for name in FIELDS[2:]):
            raise ValueError("Original UniBrain requires 96 cubed images")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        row = self.array[index]
        if row["label"] not in (0, 1) or any(not np.isfinite(row[name]).all() for name in FIELDS[2:]):
            raise ValueError("Invalid diagnosis label or nonfinite image values")
        return tuple(np.array(row[name], copy=True) if name in FIELDS[2:] else row[name] for name in FIELDS)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_split(folder, split):
    path = Path(folder) / f"adhd_{split}_sample.npy"
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.dtype.names != FIELDS or data.ndim != 1 or len(data) == 0:
        raise ValueError(f"Expected a nonempty UniBrain structured array: {path}")
    for start in range(0, len(data), 8):
        chunk = data[start:start + 8]
        for name in FIELDS[2:]:
            value = chunk[name]
            if value.shape[1:] != (96, 96, 96) or not np.isfinite(value).all():
                raise ValueError(f"Invalid shape or nonfinite values: {name}")
        for name, maximum in [(FIELDS[3], 1), (FIELDS[4], 3), (FIELDS[5], 100000)]:
            labels = chunk[name]
            if np.any(labels != np.rint(labels)) or labels.min() < 0 or labels.max() > maximum:
                raise ValueError(f"Invalid label values: {name}")
    if not set(data["label"].tolist()) <= {0, 1}:
        raise ValueError("This prototype expects binary diagnosis labels")
    if len(set(data["id"].tolist())) != len(data):
        raise ValueError(f"Duplicate subject identifiers in {split}")
    return data


def batch(row, device="cpu"):
    result = {name: torch.from_numpy(np.array(row[name], dtype=np.float32, copy=True))
              .unsqueeze(0).unsqueeze(0).to(device) for name in FIELDS[2:]}
    result["id"] = str(row["id"])
    result["label"] = torch.tensor([int(row["label"])], device=device)
    return result


def templates(folder, device="cpu"):
    names = {"image": "template_img_orig_96.npy", "tissue": "template_img_gm_mask_orig_96.npy",
             "atlas": "template_img_aal_mask_orig_96.npy"}
    result = {}
    for key, filename in names.items():
        array = np.load(Path(folder) / filename, allow_pickle=False)
        if array.shape != (96, 96, 96) or not np.isfinite(array).all():
            raise ValueError(f"Invalid template: {filename}")
        if key != "image":
            maximum = 3 if key == "tissue" else 100000
            if np.any(array != np.rint(array)) or array.min() < 0 or array.max() > maximum:
                raise ValueError(f"Invalid template labels: {filename}")
        if key == "atlas":
            values = np.unique(array)
            if len(values) != 117 or values[0] != 0:
                raise ValueError("Expected background and 116 atlas regions")
            result["atlas_label_values"] = values.astype(int).tolist()
            array = np.searchsorted(values, array)
        result[key] = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    return result


def inspect_data(folder, train, val):
    atlas_values = np.unique(np.load(Path(folder) / "template_img_aal_mask_orig_96.npy", allow_pickle=False))
    for array in (train, val):
        for start in range(0, len(array), 8):
            if not np.isin(np.unique(array["raw_img_aal_mask_np"][start:start + 8]), atlas_values).all():
                raise ValueError("Sample atlas labels are not in the template label vocabulary")
    overlap = sorted(set(train["id"].tolist()) & set(val["id"].tolist()))
    if overlap:
        raise ValueError(f"Training and validation identifiers overlap: {overlap}")
    warnings = ["Small public sample: functional validation only; no clinical performance claim.",
                "NPY files contain no voxel spacing or orientation affine; only array-grid consistency is checked.",
                "Test samples are not used for training, tuning, or candidate selection."]
    if len(np.unique(train["label"])) < 2:
        warnings.append("The training sample contains only one diagnosis class; classification quality cannot be established.")
    splits = {}
    for name, array in [("train", train), ("val", val)]:
        labels, counts = np.unique(array["label"], return_counts=True)
        splits[name] = {"count": len(array), "ids": array["id"].tolist(),
                        "class_counts": {str(k): int(v) for k, v in zip(labels, counts)},
                        "sha256": digest(Path(folder) / f"adhd_{name}_sample.npy")}
    return {"format": "UniBrain structured NPY", "shape": [96, 96, 96], "splits": splits,
            "atlas_label_values": atlas_values.astype(int).tolist(),
            "warnings": warnings, "test_used": False}
