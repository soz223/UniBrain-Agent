"""Fit validation-only structural volume controls on a preserved UniBrain fold."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(rows, atlas):
    samples = []
    for row in rows:
        brain = np.count_nonzero(row["raw_img_brain_mask_np"])
        if not brain:
            raise ValueError("Empty brain mask")
        labels, counts = np.unique(row["raw_img_aal_mask_np"], return_counts=True)
        if not np.isin(labels, atlas).all():
            raise ValueError("Sample contains an unknown atlas label")
        regions = np.zeros(len(atlas))
        regions[np.searchsorted(atlas, labels)] = counts / brain
        samples.append([np.log(brain), *regions[1:]])
    return np.asarray(samples)


def run(folder, output):
    atlas = np.unique(np.load(folder / "template_img_aal_mask_orig_96.npy"))
    if atlas[0] != 0 or len(atlas) != 117:
        raise ValueError("Expected the AAL116 template")
    splits = [np.load(folder / f"adhd_{part}_sample.npy", mmap_mode="r") for part in ("train", "val")]
    ids = [set(rows["id"].tolist()) for rows in splits]
    if ids[0] & ids[1] or any(len(keys) != len(rows) for keys, rows in zip(ids, splits)):
        raise ValueError("Duplicate or overlapping subjects")
    train, val = splits
    x, v = [features(rows, atlas) for rows in splits]
    results = {}
    for name, columns in (("brain_size", slice(0, 1)), ("brain_size_and_aal116", slice(None))):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
        model.fit(x[:, columns], train["label"])
        prediction = model.predict(v[:, columns])
        tn, fp, fn, tp = confusion_matrix(val["label"], prediction, labels=[0, 1]).ravel()
        results[name] = dict(auc=roc_auc_score(val["label"], model.predict_proba(v[:, columns])[:, 1]),
                             accuracy=accuracy_score(val["label"], prediction),
                             balanced_accuracy=balanced_accuracy_score(val["label"], prediction),
                             sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))
    record = dict(data=str(folder), n_train=len(train), n_val=len(val), validation=results,
                  features="Log brain voxel count and AAL counts divided by brain voxel count on the supplied grid")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.fold, args.output)
