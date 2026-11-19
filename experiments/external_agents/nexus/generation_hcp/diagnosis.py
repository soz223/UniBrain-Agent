"""diagnosis.py

Pure-Python baseline model for sMRI diagnosis.

Implements:
    fit_predict(x_train, y_train, x_eval) -> scores

Requirements / constraints satisfied:
- numpy/scipy/sklearn only
- deterministic (fixed random_state)
- single CPU thread
- <8GB RAM by streaming subjects and avoiding 4D dense copies
- preprocessing fit on training only; eval transformed independently
- no CV / tuning; fixed hyperparameters
- iterate over subjects for feature extraction
- safe handling for memmap/strided arrays and NaNs
- no file I/O (other than this module definition)

Method summary:
- For each 96x96x96 whole-head volume, compute a robust per-subject intensity normalization
  (median/MAD on a foreground mask derived from high quantile threshold).
- Extract a compact structural representation using multi-scale block means within a
  fixed central crop (preserves head-size/position differences, not MNI-aligned GM maps):
    * Scale 1: crop 80^3 then 8x8x8 blocks -> 1000 features
    * Scale 2: crop 72^3 then 6x6x6 blocks -> 1728 features
  Total ~2728 features per subject.
- Apply StandardScaler fit on training features.
- Fit LogisticRegression with elastic-net penalty (saga solver), class_weight='balanced'.
- Return positive-class probabilities for eval.

Input expectations:
- x_train: array-like of shape (n_train, 96, 96, 96) or iterable yielding 3D arrays.
           Can be numpy array, memmap, or list.
- y_train: array-like of shape (n_train,), binary labels {0,1} (positive class=1).
- x_eval:  array-like of shape (n_eval, 96, 96, 96) or iterable yielding 3D arrays.

The function does not modify inputs and avoids holding all volumes in memory.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np

# sklearn only; avoid any other dependencies
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _set_single_thread_determinism() -> None:
    """Force single-thread execution for common numeric backends.

    This must be called before heavy numpy / sklearn computation.
    """

    # These env vars are best-effort; not all are honored in all environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _as_float32_view(vol: np.ndarray) -> np.ndarray:
    """Return a float32 array for computations.

    Uses a view/copy only if needed; also ensures C-contiguity for predictable speed.
    """
    a = np.asarray(vol)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    # Ensure contiguous to make subsequent reshapes cheap.
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a


def _nan_to_num_inplace(a: np.ndarray) -> np.ndarray:
    """In-place NaN/inf handling (safe for float arrays)."""
    # np.nan_to_num supports in-place via out= starting numpy 1.17+
    np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _central_crop(vol: np.ndarray, out_size: int) -> np.ndarray:
    """Central cubic crop."""
    assert vol.ndim == 3
    d = vol.shape[0]
    if d == out_size:
        return vol
    start = (d - out_size) // 2
    end = start + out_size
    return vol[start:end, start:end, start:end]


def _robust_normalize(vol: np.ndarray) -> np.ndarray:
    """Robustly normalize intensities per subject.

    - Build a foreground mask by thresholding at the 80th percentile of finite voxels.
      This keeps a mix of head tissues and reduces air/background influence.
    - Compute median and MAD over foreground; standardize.

    Returns a float32 array (same shape) with NaNs/Infs replaced by 0.
    """
    v = _as_float32_view(vol)
    _nan_to_num_inplace(v)

    flat = v.reshape(-1)
    # Quantile on all voxels; for T1w whole-head, 80th percentile is usually inside head.
    # If volume is near-constant, quantile may be equal to most values.
    q80 = np.quantile(flat, 0.80)
    fg = flat > q80
    if fg.sum() < 1024:
        # Fallback: use all nonzero voxels (or all voxels if still too few)
        fg = flat != 0
        if fg.sum() < 1024:
            fg = np.ones_like(flat, dtype=bool)

    x = flat[fg]
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # Scale factor for MAD to approximate std under Gaussian; plus epsilon for stability
    scale = (1.4826 * mad) + 1e-6

    v = (v - med) / scale
    # Clip extreme values that can occur in corrupted scans; keeps linear model stable.
    np.clip(v, -10.0, 10.0, out=v)
    _nan_to_num_inplace(v)
    return v


def _block_means(cube: np.ndarray, n_blocks: int) -> np.ndarray:
    """Compute block means for a cubic array.

    cube: shape (S,S,S), S must be divisible by n_blocks.
    Returns flattened array of length n_blocks^3.
    """
    S = cube.shape[0]
    assert cube.ndim == 3 and cube.shape[1] == S and cube.shape[2] == S
    assert S % n_blocks == 0
    b = S // n_blocks
    # Reshape into blocks: (nb, b, nb, b, nb, b) then mean across block dims
    x = cube.reshape(n_blocks, b, n_blocks, b, n_blocks, b)
    feat = x.mean(axis=(1, 3, 5))
    return feat.reshape(-1).astype(np.float32, copy=False)


def _extract_features_from_volume(vol: np.ndarray) -> np.ndarray:
    """Extract fixed-length feature vector from one volume.

    Multi-scale central crops + block means after robust per-subject normalization.
    """
    v = _robust_normalize(vol)

    # Use two scales to capture both coarse morphology and more detailed patterns.
    # Keeping central crop also reduces influence of varying FoV while preserving
    # head/brain size differences (not enforcing MNI GM alignment).
    c80 = _central_crop(v, 80)
    f1 = _block_means(c80, n_blocks=10)  # 10^3 = 1000

    c72 = _central_crop(v, 72)
    f2 = _block_means(c72, n_blocks=12)  # 12^3 = 1728

    # Add a few global robust stats to help with scanner/intensity shifts.
    flat = v.reshape(-1)
    # Use percentiles on the full volume (already clipped/normalized)
    p = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99]).astype(np.float32)
    m = np.array(
        [
            float(flat.mean()),
            float(flat.std()),
            float(np.mean(np.abs(flat))),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate([f1, f2, p, m], axis=0)
    # Ensure finite
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat


def _iter_subjects(x: object) -> Iterable[np.ndarray]:
    """Iterate over subjects without assuming a specific container type."""
    if isinstance(x, np.ndarray) and x.ndim == 4:
        for i in range(x.shape[0]):
            yield x[i]
    else:
        # Assume iterable of 3D arrays
        for vol in x:  # type: ignore[assignment]
            yield np.asarray(vol)


def _compute_feature_matrix(x: object) -> np.ndarray:
    """Compute features for a dataset by streaming subjects."""
    feats = []
    for vol in _iter_subjects(x):
        f = _extract_features_from_volume(vol)
        feats.append(f)
    X = np.vstack(feats).astype(np.float32, copy=False)
    return X


def fit_predict(x_train: object, y_train: object, x_eval: object) -> np.ndarray:
    """Fit on training data and output positive-class probabilities for eval.

    Parameters
    ----------
    x_train : array-like
        Training volumes (n_train, 96, 96, 96) or iterable of 3D arrays.
    y_train : array-like
        Binary labels, positive class must be 1.
    x_eval : array-like
        Eval volumes (n_eval, 96, 96, 96) or iterable of 3D arrays.

    Returns
    -------
    scores : np.ndarray, shape (n_eval,)
        Positive-class probabilities in (0,1), finite.
    """

    _set_single_thread_determinism()

    y = np.asarray(y_train)
    # Accept {0,1} or {False,True}
    y = y.astype(np.int32, copy=False)

    # --- Feature extraction (streaming) ---
    Xtr = _compute_feature_matrix(x_train)
    Xev = _compute_feature_matrix(x_eval)

    # --- Preprocessing fit on training only ---
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xev_s = scaler.transform(Xev)

    # --- Regularized linear classifier ---
    # Elastic-net improves stability in high-dimensional but correlated features.
    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.10,
        C=0.40,
        class_weight="balanced",
        max_iter=400,
        tol=1e-4,
        random_state=0,
        n_jobs=1,
        verbose=0,
    )
    clf.fit(Xtr_s, y)

    # Predict proba; ensure finite and clipped away from exact 0/1.
    proba = clf.predict_proba(Xev_s)[:, 1].astype(np.float64, copy=False)
    proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
    eps = 1e-6
    np.clip(proba, eps, 1.0 - eps, out=proba)
    return proba
