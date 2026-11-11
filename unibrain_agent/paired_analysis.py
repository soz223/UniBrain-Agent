"""Paired diagnosis comparisons for matched, complete held-out predictions."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .data import digest


def paired_cases(pairs):
    observed, provenance, keys, signatures, fold_hashes = {}, [], set(), None, {}
    for pair in pairs:
        records = [json.loads(Path(pair[key]).read_text(encoding="utf-8")) for key in ("a", "b")]
        for record in records:
            config = record.get("config", {})
            if record.get("status") != "complete" or not record.get("test"):
                raise ValueError("Each input must be a complete trainer report with held-out test predictions")
            if config.get("max_steps") or config.get("eval_limit"):
                raise ValueError("Bounded smoke runs cannot enter scientific comparisons")
            if record["test"]["sample_count"] != record["test"]["full_split_count"]:
                raise ValueError("Test evaluation must cover the complete held-out split")
            if not config.get("dataset") or config.get("fold") is None or config.get("seed") is None:
                raise ValueError("Reports must declare dataset, fold and seed")
        identity = tuple(records[0]["config"][key] for key in ("dataset", "fold", "seed"))
        settings = ("method", "setting", "initialization", "trainable", "cut_extraction", "connect_paths",
                    "cut_boundary", "repair_strategy", "undo_repair", "loss_scale", "regularization", "epochs",
                    "pretrain_epochs", "patience", "composition_mode")
        current = [{key: record["config"].get(key) for key in settings} for record in records]
        if any(not row["method"] or not row["setting"] for row in current):
            raise ValueError("Reports must identify the method and setting")
        if signatures is not None and current != signatures:
            raise ValueError("A comparison branch cannot mix methods, budgets or interventions across runs")
        signatures = current
        if identity in keys or any(records[0]["config"][key] != records[1]["config"][key] for key in ("dataset", "fold", "seed")):
            raise ValueError("Each dataset, fold and seed needs exactly one matched pair")
        keys.add(identity)
        hashes = [record.get("audit", {}).get("data_hashes", {}).get("adhd_test_sample.npy") for record in records]
        if not hashes[0] or hashes[0] != hashes[1]:
            raise ValueError("Compared runs must have the same test data hash")
        previous = fold_hashes.setdefault(identity[:2], hashes[0])
        if previous != hashes[0]:
            raise ValueError("Test data changed across seeds within one fold")
        indexed = [{str(case["id"]): case for case in record["test"]["cases"]} for record in records]
        if any(len(indexed[i]) != records[i]["test"]["sample_count"] for i in range(2)) or indexed[0].keys() != indexed[1].keys():
            raise ValueError("Test subjects must be unique and exactly matched")
        for subject, a in indexed[0].items():
            b = indexed[1][subject]
            if a["label"] != b["label"] or a["label"] not in (0, 1):
                raise ValueError("Matched diagnosis labels disagree")
            scores = [a["probability"], b["probability"]]
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores):
                raise ValueError("Finite class one probabilities in [0, 1] are required")
            key = (identity[0], subject)
            row = observed.setdefault(key, {"label": a["label"], "scores": []})
            if row["label"] != a["label"]:
                raise ValueError("A subject has inconsistent labels across runs")
            row["scores"].append(scores)
        provenance.append({"dataset": identity[0], "fold": identity[1], "seed": identity[2],
                           "a_sha256": digest(Path(pair["a"])), "b_sha256": digest(Path(pair["b"]))})
    if len({key[0] for key in observed}) != 1:
        raise ValueError("Analyze each dataset separately")
    seed_sets = [{key[2] for key in keys if key[:2] == fold} for fold in fold_hashes]
    if any(values != seed_sets[0] for values in seed_sets):
        raise ValueError("All included folds must have the same set of matched seeds")
    labels, scores, repeats = [], [], []
    for key in sorted(observed):
        row = observed[key]
        labels.append(row["label"])
        scores.append(np.mean(row["scores"], axis=0))
        repeats.append(len(row["scores"]))
    if len(set(labels)) != 2:
        raise ValueError("Both diagnosis classes are required")
    return np.asarray(labels), np.asarray(scores), provenance, repeats


def measures(labels, scores):
    predictions = scores > .5
    return np.asarray([roc_auc_score(labels, scores), np.mean(predictions == labels),
                       balanced_accuracy_score(labels, predictions), f1_score(labels, predictions, average="macro", zero_division=0)])


def compare(pairs, repetitions=2000, seed=17):
    if repetitions < 100:
        raise ValueError("At least 100 paired bootstrap samples are required")
    labels, scores, provenance, repeats = paired_cases(pairs)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    point = [measures(labels, scores[:, column]) for column in (0, 1)]
    differences = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        differences.append(measures(labels[indices], scores[indices, 0]) - measures(labels[indices], scores[indices, 1]))
    bounds = np.percentile(differences, [2.5, 97.5], axis=0)
    return {"status": "complete", "subjects": len(labels), "matched_run_pairs": len(pairs), "seed": seed,
            "included_folds": sorted({row["fold"] for row in provenance}, key=str),
            "included_seeds": sorted({row["seed"] for row in provenance}),
            "bootstrap_repetitions": repetitions, "repeat_range_per_subject": [min(repeats), max(repeats)],
            "metrics": {name: {"a": float(point[0][i]), "b": float(point[1][i]),
                                "difference_a_minus_b": float(point[0][i] - point[1][i]),
                                "paired_subject_ci95": bounds[:, i].tolist()}
                        for i, name in enumerate(("auc", "accuracy", "balanced_accuracy", "macro_f1"))},
            "inputs": provenance,
            "estimand": "Pooled held-out subject scores, averaging matched seeds and repeated held-out appearances before scoring",
            "uncertainty": "Class-stratified paired subject bootstrap conditional on the fitted models; not uncertainty over model retraining",
            "boundary": "This pooled estimand differs from the mean of fold metrics. Site clustering and multiplicity are not modeled. No automatic significance claim."}
