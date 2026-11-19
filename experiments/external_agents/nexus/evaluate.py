"""Evaluate a reviewed, hash-locked, NEXUS-generated classifier on one fixed fold."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits


def evaluate(fold, generated, output, threshold):
    provenance = json.loads((generated / "generation.json").read_text())
    model_path = generated / "diagnosis.py"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != provenance["model_sha256"]:
        raise ValueError("Model source does not match generation provenance")
    spec = importlib.util.spec_from_file_location("nexus_generated_model", model_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    train, val = [np.load(fold / f"adhd_{split}_sample.npy", mmap_mode="r", allow_pickle=False)
                  for split in ("train", "val")]
    for group in (train, val):
        if len(set(group["id"].tolist())) != len(group) or set(group["label"].tolist()) != {0, 1}:
            raise ValueError("Invalid participant identifiers or diagnoses")
        if group["raw_img_np"].shape != (len(group), 96, 96, 96):
            raise ValueError("Unexpected image dimensions")
    if set(train["id"].tolist()) & set(val["id"].tolist()):
        raise ValueError("Training and validation subjects overlap")
    started = time.monotonic()
    with threadpool_limits(limits=1):
        scores = np.asarray(model.fit_predict(train["raw_img_np"], train["label"], val["raw_img_np"]), dtype=float)
    if scores.shape != (len(val),) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per validation subject")
    predicted = scores >= threshold
    target = val["label"]
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "validation_scores.npy", scores)
    result = {"scope": provenance["scope"], "model_sha256": digest, "upstream": provenance["upstream"],
              "phase": "validation", "fold": fold.name, "n_train": len(train), "n_val": len(val),
              "seconds": time.monotonic() - started, "decision_threshold": threshold,
              "validation": {"auc": roc_auc_score(target, scores), "accuracy": accuracy_score(target, predicted),
                             "balanced_accuracy": balanced_accuracy_score(target, predicted),
                             "macro_f1": f1_score(target, predicted, average="macro"),
                             "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)}}
    (output / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    evaluate(args.fold, args.generated, args.output, args.threshold)
