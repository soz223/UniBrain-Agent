#!/usr/bin/env python3
"""Self-contained smoke adapter for UniBrain ADHD-vs-control training.

Constraints (enforced/observed):
- Keeps ./source immutable; imports by path.
- Uses original architecture, forward graph, losses, augmentation, and hyperparameters from source/main.py & source/train.py.
- Uses exactly `--steps` genuine optimizer updates (default caller uses 2).
- Runs at least one validation forward pass (here: up to first two per class from val set).
- Never loads/copies the entire training array: uses np.load(..., mmap_mode='r') and slices rows.
- Never touches any test file.
- Writes output/report.json with required evidence; fails loudly otherwise.

This script is designed to be run as:
  python adapter.py --source PATH --fold PATH --output NEW_DIR --steps 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ---- import provided source (no module name) ----

def _import_source_modules(source_dir: str):
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"--source does not exist or is not a directory: {source_dir}")
    sys.path.insert(0, source_dir)
    try:
        import model as model_mod  # type: ignore
        import utils as utils_mod  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed importing provided source modules from {source_dir}: {e}") from e
    return model_mod, utils_mod


# ---- device-safe patches (do not edit source files) ----

def _am_1d_2_Nd_torch_device(am, device):
    # Mirrors source/model.py and source/utils.py am_1d_2_Nd_torch but without hard-coded .cuda().
    import torch

    am = am.squeeze()
    mask_label = torch.unique(am)

    multi_d = torch.zeros((len(mask_label), 96, 96, 96), device=device)
    for i in range(len(mask_label)):
        label_idx = mask_label[i]
        multi_d[i][am == label_idx] = 1.0
    return multi_d.unsqueeze(0).float()


def _patch_model_for_device(model_mod, device):
    import torch

    # get_batch_identity_theta_* and am_1d_2_Nd_torch in model.py hardcode CUDA.
    def get_batch_identity_theta_4_4(batch_size: int):
        theta = torch.tensor(
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1],
            dtype=torch.float,
            device=device,
        )
        return theta.view(4, 4).unsqueeze(0).repeat(batch_size, 1, 1)

    def get_batch_identity_theta_3_4(batch_size: int):
        theta = torch.tensor(
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0],
            dtype=torch.float,
            device=device,
        )
        return theta.view(3, 4).unsqueeze(0).repeat(batch_size, 1, 1)

    def am_1d_2_Nd_torch(am):
        return _am_1d_2_Nd_torch_device(am, device)

    model_mod.get_batch_identity_theta_4_4 = get_batch_identity_theta_4_4
    model_mod.get_batch_identity_theta_3_4 = get_batch_identity_theta_3_4
    model_mod.am_1d_2_Nd_torch = am_1d_2_Nd_torch

    # NCC.loss hardcodes sum_filt to "cuda"; we patch it device-safe (even if not used in this smoke).
    if hasattr(model_mod, "NCC"):
        NCC = model_mod.NCC

        orig_loss = NCC.loss

        def loss(self, y_true, y_pred):
            import torch.nn.functional as F
            import torch

            I = y_true
            J = y_pred
            ndims = len(list(I.size())) - 2
            if ndims not in (1, 2, 3):
                raise AssertionError(f"volumes should be 1 to 3 dimensions. found: {ndims}")

            win = [9] * ndims if getattr(self, "win", None) is None else self.win
            sum_filt = torch.ones([1, 1, *win], device=device)
            pad_no = math.floor(win[0] / 2)
            if ndims == 1:
                stride = (1,)
                padding = (pad_no,)
            elif ndims == 2:
                stride = (1, 1)
                padding = (pad_no, pad_no)
            else:
                stride = (1, 1, 1)
                padding = (pad_no, pad_no, pad_no)

            conv_fn = getattr(F, f"conv{ndims}d")
            I2 = I * I
            J2 = J * J
            IJ = I * J
            I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = float(np.prod(win))
            u_I = I_sum / win_size
            u_J = J_sum / win_size
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
            cc = cross * cross / (I_var * J_var + 1e-5)
            return -torch.mean(cc)

        NCC.loss = loss  # type: ignore


def _train_aug_device(img, label, img_size, degree, voxel, scale_min, scale_max, device):
    # Mirrors utils.train_aug but without hard-coded CUDA.
    import torch
    import torch.nn.functional as F

    # Reuse utils functions exactly by reimplementing the same logic here.
    def get_translation_3D(tx, ty, tz):
        t = torch.zeros(1, 4, 4, device=device)
        t[:, :, :4] = torch.tensor(
            [[1, 0, 0, tx],
             [0, 1, 0, ty],
             [0, 0, 1, tz],
             [0, 0, 0, 1]],
            device=device,
        )
        return t

    def get_rotate_3D_x(angle):
        r_x = torch.zeros(1, 4, 4, device=device)
        angle = np.radians(angle)
        r_x[:, :, :4] = torch.tensor(
            [[1, 0, 0, 0],
             [0, np.cos(angle), -np.sin(angle), 0],
             [0, np.sin(angle), np.cos(angle), 0],
             [0, 0, 0, 1]],
            device=device,
        )
        return r_x

    def get_rotate_3D_y(angle):
        r_y = torch.zeros(1, 4, 4, device=device)
        angle = np.radians(angle)
        r_y[:, :, :4] = torch.tensor(
            [[np.cos(angle), 0, np.sin(angle), 0],
             [0, 1, 0, 0],
             [-np.sin(angle), 0, np.cos(angle), 0],
             [0, 0, 0, 1]],
            device=device,
        )
        return r_y

    def get_rotate_3D_z(angle):
        r_z = torch.zeros(1, 4, 4, device=device)
        angle = np.radians(angle)
        r_z[:, :, :4] = torch.tensor(
            [[np.cos(angle), -np.sin(angle), 0, 0],
             [np.sin(angle), np.cos(angle), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            device=device,
        )
        return r_z

    def get_scale_3D(sx, sy, sz):
        t = torch.zeros(1, 4, 4, device=device)
        t[:, :, :4] = torch.tensor(
            [[sx, 0, 0, 0],
             [0, sy, 0, 0],
             [0, 0, sz, 0],
             [0, 0, 0, 1]],
            device=device,
        )
        return t

    def random_affine_3D(img_size_, degree_, voxel_, scale_min_, scale_max_):
        angle_x = random.uniform(-degree_, degree_)
        angle_y = random.uniform(-degree_, degree_)
        angle_z = random.uniform(-degree_, degree_)
        tx = random.uniform(-voxel_, voxel_)
        ty = random.uniform(-voxel_, voxel_)
        tz = random.uniform(-voxel_, voxel_)
        sx = random.uniform(scale_min_, scale_max_)
        sy = random.uniform(scale_min_, scale_max_)
        sz = random.uniform(scale_min_, scale_max_)
        c = get_translation_3D(-img_size_ // 2, -img_size_ // 2, -img_size_ // 2)
        c_inv = torch.linalg.inv(c)
        r_x = get_rotate_3D_x(angle_x)
        r_y = get_rotate_3D_y(angle_y)
        r_z = get_rotate_3D_z(angle_z)
        t = get_translation_3D(tx, ty, tz)
        s = get_scale_3D(sx, sy, sz)
        A = c_inv @ t @ r_x @ r_y @ r_z @ s @ c
        return A

    def param2theta(param, x, y, z):
        param = np.linalg.inv(param)
        theta = np.zeros([3, 4], dtype=np.float32)
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * y / x
        theta[0, 2] = param[0, 2] * z / x
        theta[0, 3] = theta[0, 0] + theta[0, 1] + theta[0, 2] + 2 * param[0, 3] / x - 1

        theta[1, 0] = param[1, 0] * x / y
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * z / y
        theta[1, 3] = theta[1, 1] + theta[1, 2] + 2 * param[1, 3] / y + theta[1, 0] - 1

        theta[2, 0] = param[2, 0] * x / z
        theta[2, 1] = param[2, 1] * y / z
        theta[2, 2] = param[2, 2]
        theta[2, 3] = theta[2, 2] + 2 * param[2, 3] / z + theta[2, 0] + theta[2, 1] - 1
        return theta

    A = random_affine_3D(img_size, degree, voxel, scale_min, scale_max)
    param = A.detach().cpu().numpy().reshape(4, 4)
    theta = param2theta(param, img_size, img_size, img_size)
    theta_torch = torch.from_numpy(theta).view(-1, 3, 4).to(device=device, dtype=torch.float32)
    grid = F.affine_grid(theta_torch, torch.Size([1, 1, img_size, img_size, img_size]), align_corners=True)

    img_transformed = F.grid_sample(img, grid, mode='bilinear', align_corners=True, padding_mode="zeros")
    label_transformed = F.grid_sample(label, grid, mode='nearest', align_corners=True, padding_mode="zeros")
    return img_transformed, label_transformed


# ---- data access ----

@dataclass(frozen=True)
class Sample:
    label: int
    raw_img: np.ndarray
    brain_mask: np.ndarray
    gm_mask: np.ndarray
    aal_mask: np.ndarray


class NpyRowDataset:
    """Row-wise access to saved NPY without loading the whole array."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        # mmap_mode='r' is required by the task; if the array cannot be memory-mapped (e.g. object array), we fail.
        arr = np.load(path, mmap_mode='r', allow_pickle=True)
        self._path = path
        self._arr = arr
        # Heuristic: memmap if base is memmap or instance is memmap.
        is_memmap = isinstance(arr, np.memmap) or isinstance(getattr(arr, 'base', None), np.memmap)
        if not is_memmap:
            raise RuntimeError(
                f"Dataset NPY is not memory-mappable (would load entire array), refusing: {path}. "
                f"Loaded type={type(arr)}, dtype={getattr(arr, 'dtype', None)}"
            )

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def get_row(self, idx: int):
        return self._arr[idx]


def _parse_row_to_sample(row: Any) -> Sample:
    # Matches utils.MyDataset_no_fix_v2.__getitem__ ordering
    # Returns label + 4 tensors used in training; id is ignored to avoid participant leakage.
    # row may be tuple/list or structured; we rely on index access as in original.
    label = int(row[1])
    raw_img = row[2]
    brain_mask = row[3]
    gm_mask = row[4]
    aal_mask = row[5]
    return Sample(label=label, raw_img=raw_img, brain_mask=brain_mask, gm_mask=gm_mask, aal_mask=aal_mask)


def _select_indices_by_class(ds: NpyRowDataset, per_class: int, classes: Sequence[int] = (0, 1)) -> List[int]:
    picked: Dict[int, List[int]] = {c: [] for c in classes}
    out: List[int] = []
    for i in range(len(ds)):
        row = ds.get_row(i)
        lbl = int(row[1])
        if lbl in picked and len(picked[lbl]) < per_class:
            picked[lbl].append(i)
            out.append(i)
        if all(len(picked[c]) >= per_class for c in classes):
            break
    return out


def _finite(x: float) -> bool:
    return (not math.isnan(x)) and (not math.isinf(x))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to provided source directory (contains 4 .py files)")
    ap.add_argument("--fold", required=True, help="Path to fold directory containing adhd_train_sample.npy, adhd_val_sample.npy, and template NPYs")
    ap.add_argument("--output", required=True, help="Output directory to create")
    ap.add_argument("--steps", required=True, type=int, help="Number of genuine optimizer updates to run")
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=False)

    # Determinism: preserve deterministic ordering and augmentation randomness.
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    model_mod, utils_mod = _import_source_modules(args.source)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    _patch_model_for_device(model_mod, device)

    # Original hyperparameters from source/main.py
    img_size = 96
    ext_stage = 1
    reg_stage = 5

    # Original flags/hparams
    reg_loss_name = "GCC"
    weight_decay = 0.000001
    if_train_aug = True
    if_pred_aal = False
    lambda_ext = 1
    lambda_reg = 0.1
    lambda_seg = 1
    lambda_cls = 1
    learning_rate = 0.00001

    batch_size = 1  # required to make exactly `steps` genuine updates, one sample per update

    # Build model exactly as in main.py
    model = model_mod.UniBrain(
        img_size,
        ext_stage,
        reg_stage,
        if_pred_aal,
        [1, 16, 32, 32, 64, 64, 128, 128],
        [128, 64, 64, 32, 32, 32, 16, 1],
        [2, 16, 32, 64, 128, 256, 512],
        [1, 128, 256, 256, 512, 512],
        [512, 256, 256, 256, 128, 4],
        [1, 128, 256, 256, 512, 512],
        [512, 256, 256, 256, 128, 117],
        256,
        128,
        128,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Losses as in train.py
    ext_loss_func = torch.nn.BCELoss()
    seg_loss_func = torch.nn.CrossEntropyLoss()
    if reg_loss_name == "NCC":
        reg_loss_func = model_mod.NCC().loss
    elif reg_loss_name == "GCC":
        reg_loss_func = model_mod.GCC().to(device)
    elif reg_loss_name == "MSE":
        reg_loss_func = torch.nn.MSELoss().to(device)
    else:
        raise ValueError(f"Unknown reg_loss_name: {reg_loss_name}")
    classifier_loss_func = torch.nn.CrossEntropyLoss()

    # Load templates referenced by train.py, but from --fold (not cwd ./dataset).
    def load_template(name: str) -> torch.Tensor:
        p = os.path.join(args.fold, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required template NPY in --fold: {p}")
        arr = np.load(p)
        t = torch.from_numpy(arr).to(device).view(-1, 1, img_size, img_size, img_size).float()
        return t

    fixed_data = load_template("template_img_orig_96.npy")
    fixed_label_am = load_template("template_img_gm_mask_orig_96.npy")
    fixed_label_aal = load_template("template_img_aal_mask_orig_96.npy")

    # Datasets (must be memory-mapped)
    train_path = os.path.join(args.fold, "adhd_train_sample.npy")
    val_path = os.path.join(args.fold, "adhd_val_sample.npy")
    train_ds = NpyRowDataset(train_path)
    val_ds = NpyRowDataset(val_path)

    # Select exactly `steps` distinct training samples, in deterministic dataset order.
    # We do not shuffle; we pick first occurrence indices (could be same class; allowed; still distinct rows).
    train_indices: List[int] = list(range(min(args.steps, len(train_ds))))
    if len(set(train_indices)) != len(train_indices):
        raise RuntimeError("Training indices are not distinct; this should never happen")
    if len(train_indices) < args.steps:
        raise RuntimeError(f"Requested steps={args.steps} but train dataset has only {len(train_ds)} rows")

    # Validation: up to first two per class, deterministic.
    val_indices = _select_indices_by_class(val_ds, per_class=2, classes=(0, 1))
    if len(val_indices) == 0:
        raise RuntimeError("No validation samples available")

    report: Dict[str, Any] = {
        "task": "UniBrain ADHD vs control smoke integration",
        "steps_requested": args.steps,
        "device": str(device),
        "source_dir": os.path.abspath(args.source),
        "fold_dir": os.path.abspath(args.fold),
        "output_dir": os.path.abspath(args.output),
        "hyperparameters": {
            "img_size": img_size,
            "ext_stage": ext_stage,
            "reg_stage": reg_stage,
            "reg_loss_name": reg_loss_name,
            "weight_decay": weight_decay,
            "if_train_aug": if_train_aug,
            "if_pred_aal": if_pred_aal,
            "lambda_ext": lambda_ext,
            "lambda_reg": lambda_reg,
            "lambda_seg": lambda_seg,
            "lambda_cls": lambda_cls,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        "adaptations": [],
        "updates": [],
        "validation": {},
    }

    report["adaptations"].append(
        "Patched source/model.py CUDA-hardcoded helpers at runtime (identity-theta and am_1d_2_Nd_torch) to be device-aware; forward computation graph preserved aside from device placement."
    )
    report["adaptations"].append(
        "Reimplemented utils.train_aug logic locally to remove hard-coded CUDA, preserving the same affine augmentation math."
    )
    report["adaptations"].append(
        "Loaded templates from --fold rather than ./dataset to keep integration self-contained and avoid touching other filesystem paths."
    )

    # Track one parameter tensor for evidence of change.
    tracked_name = None
    tracked_param = None
    for n, p in model.named_parameters():
        if p.requires_grad:
            tracked_name = n
            tracked_param = p
            break
    if tracked_param is None:
        raise RuntimeError("No trainable parameters found")

    # ---- training: exactly args.steps optimizer updates ----
    model.train()
    updates_done = 0

    for step_i, row_idx in enumerate(train_indices):
        row = train_ds.get_row(row_idx)
        s = _parse_row_to_sample(row)

        # Convert sample arrays to torch tensors on device.
        moving_data = torch.from_numpy(np.asarray(s.raw_img)).to(device).view(-1, 1, img_size, img_size, img_size).float()
        moving_label_bm = torch.from_numpy(np.asarray(s.brain_mask)).to(device).view(-1, 1, img_size, img_size, img_size).float()
        label = torch.tensor([s.label], device=device, dtype=torch.long)

        if if_train_aug:
            degree = 2
            voxel = 2
            scale_min = 0.98
            scale_max = 1.02
            moving_data, moving_label_bm = _train_aug_device(
                moving_data, moving_label_bm, img_size, degree, voxel, scale_min, scale_max, device
            )

        optimizer.zero_grad(set_to_none=True)

        # Forward: preserve original call signature and outputs.
        (striped_list, mask_list, warped_list, theta_list, theta_list_inv,
         am_mov_pred, am_fix_2_moving, aal_mov_pred, aal_fix_2_moving, ajc_m, y_predicted) = model(
            fixed_data, moving_data, fixed_label_am, fixed_label_aal, if_train=True
        )

        # Losses: match train.py
        ext_loss = ext_loss_func(mask_list[-1], moving_label_bm)

        am_fix_2_moving_Nd = _am_1d_2_Nd_torch_device(am_fix_2_moving, device)
        seg_loss = seg_loss_func(am_mov_pred, am_fix_2_moving_Nd)

        sim_loss = reg_loss_func(warped_list[-1], fixed_data) if callable(reg_loss_func) else reg_loss_func(warped_list[-1], fixed_data)

        classifier_loss = classifier_loss_func(y_predicted, label)

        total_loss = lambda_reg * sim_loss + lambda_ext * ext_loss + lambda_seg * seg_loss + lambda_cls * classifier_loss

        # Evidence: param snapshot pre-update
        pre_param = tracked_param.detach().clone()

        total_loss.backward()

        # Evidence: gradient norm (nonzero)
        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            grad_sq += float((g * g).sum().item())
        grad_norm = float(math.sqrt(grad_sq))

        optimizer.step()
        updates_done += 1

        post_param = tracked_param.detach().clone()
        delta = float((post_param - pre_param).abs().max().item())

        # Record per-update losses
        update_rec = {
            "update_index": updates_done,
            "train_row_index": int(row_idx),
            "loss_total": float(total_loss.item()),
            "loss_components": {
                "lambda_reg": lambda_reg,
                "lambda_ext": lambda_ext,
                "lambda_seg": lambda_seg,
                "lambda_cls": lambda_cls,
                "sim_loss": float(sim_loss.item()) if hasattr(sim_loss, "item") else float(sim_loss),
                "ext_loss": float(ext_loss.item()),
                "seg_loss": float(seg_loss.item()),
                "classifier_loss": float(classifier_loss.item()),
            },
            "evidence": {
                "grad_norm_l2": grad_norm,
                "tracked_param_name": tracked_name,
                "tracked_param_max_abs_delta": delta,
            },
        }

        # Validate required finiteness and evidence
        if not _finite(update_rec["loss_total"]):
            raise RuntimeError(f"Non-finite total loss at update {updates_done}: {update_rec['loss_total']}")
        for k, v in update_rec["loss_components"].items():
            if isinstance(v, float) and (k.endswith("loss") or k in ("sim_loss", "ext_loss", "seg_loss", "classifier_loss")):
                if not _finite(v):
                    raise RuntimeError(f"Non-finite component loss {k} at update {updates_done}: {v}")
        if not (grad_norm > 0.0 and _finite(grad_norm)):
            raise RuntimeError(f"No/invalid gradient observed at update {updates_done}: grad_norm={grad_norm}")
        if not (delta > 0.0 and _finite(delta)):
            raise RuntimeError(f"No tracked parameter change observed at update {updates_done}: delta={delta}")

        report["updates"].append(update_rec)

    if updates_done != args.steps:
        raise RuntimeError(f"Expected exactly {args.steps} optimizer updates, got {updates_done}")

    # ---- validation forward(s) ----
    model.eval()
    val_probs: List[float] = []
    val_labels: List[int] = []
    with torch.no_grad():
        for vi in val_indices:
            row = val_ds.get_row(vi)
            s = _parse_row_to_sample(row)

            moving_data = torch.from_numpy(np.asarray(s.raw_img)).to(device).view(-1, 1, img_size, img_size, img_size).float()
            label = torch.tensor([s.label], device=device, dtype=torch.long)

            out = model(fixed_data, moving_data, fixed_label_am, fixed_label_aal, if_train=False)
            y_predicted_val = out[-1]
            probs = torch.nn.functional.softmax(y_predicted_val, dim=1)[:, 1].detach().cpu().numpy().tolist()
            val_probs.extend([float(p) for p in probs])
            val_labels.extend([int(label.item())])

    if len(val_probs) < 1:
        raise RuntimeError("Validation produced no probabilities")

    report["validation"] = {
        "val_samples_forwarded": int(len(val_probs)),
        "class1_probabilities": [float(p) for p in val_probs],
        "labels": [int(x) for x in val_labels],
        "note": "No participant IDs recorded; only labels and continuous class-1 probabilities.",
    }

    # Final required checks
    if not report["updates"]:
        raise RuntimeError("No optimizer updates were recorded")

    out_path = os.path.join(args.output, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
