"""Score a frozen, reviewed agent implementation on the validation subjects of a preserved fold."""

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def images(rows, destination):
    array = np.lib.format.open_memmap(destination, mode="w+", dtype="float32", shape=(len(rows), 96, 96, 96))
    for start in range(0, len(rows), 8):
        batch = rows["raw_img_np"][start:start + 8]
        if batch.shape[1:] != (96, 96, 96) or not np.isfinite(batch).all():
            raise ValueError("Invalid MRI")
        array[start:start + len(batch)] = batch
    array.flush()
    del array
    return np.load(destination, mmap_mode="r")


def evaluate(model_file, expected_hash, fold, destination):
    model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if model_hash != expected_hash:
        raise ValueError("Implementation differs from reviewed file")
    rows = [np.load(fold / f"adhd_{split}_sample.npy", mmap_mode="r") for split in ("train", "val")]
    train, val = rows
    ids = [set(x["id"].tolist()) for x in rows]
    if ids[0] & ids[1] or any(len(s) != len(x) for s, x in zip(ids, rows)):
        raise ValueError("Duplicate or shared subjects")
    if any(set(np.unique(x["label"])) != {0, 1} for x in rows):
        raise ValueError("Expected two diagnosis classes in both splits")
    spec = importlib.util.spec_from_file_location("frozen_classifier", model_file)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    with tempfile.TemporaryDirectory() as work:
        train_x, val_x = [images(x, Path(work) / f"x{i}.npy") for i, x in enumerate(rows)]
        prob = np.array(model.fit_predict(train_x, np.asarray(train["label"], dtype=int).copy(), val_x), copy=True)
        del train_x, val_x
    if prob.shape != (len(val),) or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Invalid prediction probabilities")
    truth = val["label"]
    pred = prob >= 0.5
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    record = dict(n_train=len(train), n_val=len(val), model_sha256=model_hash, split="val",
                  auc=roc_auc_score(truth, prob), accuracy=accuracy_score(truth, pred),
                  balanced_accuracy=balanced_accuracy_score(truth, pred),
                  sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)))
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "validation_probabilities.npy", prob)
    (destination / "validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("fold", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.model, args.sha256, args.fold, args.output)
