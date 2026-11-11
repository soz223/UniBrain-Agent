"""Evaluate a pinned UniBrain checkpoint on the untouched test split."""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unibrain_agent.data import OriginalDataset, digest
from unibrain_agent.native import verify_original


def original_config(source):
    tree = ast.parse((source / "main.py").read_text(encoding="utf-8"))
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_Unibrain")
    result = {}
    for node in entry.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def evaluate(source, data_dir, checkpoint, output, limit=None, threads=4):
    verify_original(source)
    torch.set_num_threads(threads)
    if not torch.cuda.is_available():
        raise RuntimeError("Original UniBrain evaluation requires CUDA")
    sys.path.insert(0, str(source))
    from model import UniBrain
    from utils import compute_label_dice_v2, f1_loss, mutual_information, normalized_cross_correlation

    config = original_config(source)
    keys = ("img_size", "ext_stage", "reg_stage", "if_pred_aal", "enc_nf", "dec_nf", "enc_affine",
            "enc_seg", "dec_seg", "enc_par", "dec_par", "feature_net_hidden", "feature_dim", "gnn_hidden")
    model = UniBrain(*(config[key] for key in keys)).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=True), strict=True)
    model.eval()
    dataset = OriginalDataset(data_dir / "adhd_test_sample.npy")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    fixed = torch.from_numpy(np.load(data_dir / "template_img_orig_96.npy")).cuda().view(1, 1, 96, 96, 96).float()
    tissue = torch.from_numpy(np.load(data_dir / "template_img_gm_mask_orig_96.npy")).cuda().view(1, 1, 96, 96, 96).float()
    atlas = torch.from_numpy(np.load(data_dir / "template_img_aal_mask_orig_96.npy")).cuda().view(1, 1, 96, 96, 96).float()
    cases = []
    with torch.no_grad():
        for index, row in enumerate(loader):
            if limit and index >= limit:
                break
            identifier, label, image, brain_mask, tissue_mask, atlas_mask = row
            image = image.cuda().view(1, 1, 96, 96, 96).float()
            brain_mask = brain_mask.cuda().view(1, 1, 96, 96, 96).float()
            tissue_mask = tissue_mask.cuda().view(1, 1, 96, 96, 96).float()
            atlas_mask = atlas_mask.cuda().view(1, 1, 96, 96, 96).float()
            extracted, masks, warped, transforms, _, tissue_logits, warped_tissue, _, warped_atlas, _, logits = model(
                fixed, image, tissue, atlas, if_train=False)
            combined_mask = torch.prod(torch.cat(masks), 0).unsqueeze(0)
            registered_tissue = F.grid_sample(tissue_mask, F.affine_grid(transforms[-1], tissue_mask.size(), align_corners=True),
                                              mode="nearest", align_corners=True, padding_mode="zeros")
            probability = torch.softmax(logits, dim=1)[0, 1].item()
            case = {"id": identifier[0], "label": int(label[0]), "probability": probability,
                    "prediction": int(logits.argmax(dim=1)[0]),
                    "brain_dice": float(f1_loss(combined_mask, brain_mask)),
                    "registration_mi": float(mutual_information(warped[-1], fixed)),
                    "registration_cc": float(normalized_cross_correlation(warped[-1], fixed)),
                    "registration_dice": float(compute_label_dice_v2(tissue, registered_tissue, "gm")),
                    "tissue_dice": float(compute_label_dice_v2(tissue_logits.argmax(dim=1), tissue_mask, "gm")),
                    "atlas_dice": float(compute_label_dice_v2(warped_atlas, atlas_mask, "aal"))}
            if not all(np.isfinite(value) for key, value in case.items() if key not in {"id", "label", "prediction"}):
                raise ValueError("Nonfinite test metric")
            cases.append(case)
            print(json.dumps({"completed": len(cases), "total": min(limit or len(dataset), len(dataset))}), flush=True)
    labels = [case["label"] for case in cases]
    predictions = [case["prediction"] for case in cases]
    scores = [case["probability"] for case in cases]
    metrics = {"accuracy": accuracy_score(labels, predictions), "macro_f1": f1_score(labels, predictions, average="macro"),
               "auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else None}
    for name in ("brain_dice", "registration_mi", "registration_cc", "registration_dice", "tissue_dice", "atlas_dice"):
        metrics[name] = float(np.mean([case[name] for case in cases]))
    result = {"split": "test", "sample_count": len(cases), "full_split_count": len(dataset),
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "source": verify_original(source), "metrics": metrics, "cases": cases}
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive")
    result = evaluate(args.source, args.data_dir, args.checkpoint, args.output, args.limit, args.threads)
    print(json.dumps({"sample_count": result["sample_count"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
