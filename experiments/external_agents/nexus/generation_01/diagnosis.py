"""ADHD-vs-control sMRI classifier (deterministic, CPU-only, low-memory).

Public API
----------
fit_predict(x_train, y_train, x_eval) -> np.ndarray of shape (len(x_eval),)
    Returns one finite continuous score per eval subject (higher => more likely y=1).

Design goals / constraints satisfied
-----------------------------------
- Deterministic: fixed RNG seed, deterministic ops only.
- One CPU thread: uses threadpoolctl when available; also sets common env vars.
- <8GB RAM: iterates over subjects for feature extraction; does not copy full 4D.
- No filesystem I/O: no reads/writes except caller-managed and final model file not used.
- No metadata/affines used.
- Operates directly on provided intensities (no GM template imposing).
- Preprocessing fit on training only; eval transformed independently.
- Numpy/Scipy/Scikit-learn only.
- Compatible with memmapped/strided arrays: uses np.asarray(view) on per-subject.

Method summary
--------------
Representation: Multi-scale blockwise moments + light frequency content + global stats.
- Downsample into a coarse 3D grid via average pooling (non-overlapping blocks).
- For each pooled voxel: use mean and log-variance pooled within the block.
- Add simple 3D DCT low-frequency energy features computed on a small pooled volume.
- Add global robust intensity summaries.

Model: LogisticRegression (saga) with StandardScaler; C via lightweight CV.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Tuple

import numpy as np


# -------------------------
# Threading / determinism
# -------------------------
SEED = 1337

# Best-effort environment thread caps (safe if already set by caller)
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")


@contextmanager
def _one_thread():
    """Context manager to force single-threaded BLAS/OpenMP where possible."""
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        with threadpool_limits(limits=1):
            yield
    except Exception:
        # threadpoolctl not available or failed; fall back to env vars only
        yield


# -------------------------
# Feature extraction utils
# -------------------------

def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def _global_stats(vol: np.ndarray) -> np.ndarray:
    """Compute robust-ish global summaries without copying the whole volume."""
    v = vol.ravel()
    # Use float64 accumulations for numeric stability
    mean = float(np.mean(v, dtype=np.float64))
    std = float(np.std(v, dtype=np.float64))
    vmin = float(np.min(v))
    vmax = float(np.max(v))

    # Quantiles: numpy computes with partial sorting; operates on 1D view
    q = np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q = q.astype(np.float32, copy=False)

    # Intensity tail mass (sensitive to tissue contrast / artifacts)
    # thresholds relative to median and IQR
    med = float(q[3])
    iqr = float(q[4] - q[2])
    if iqr <= 1e-8:
        hi_thr = med + 1.0
        lo_thr = med - 1.0
    else:
        hi_thr = med + 1.5 * iqr
        lo_thr = med - 1.5 * iqr
    frac_hi = float(np.mean(v > hi_thr))
    frac_lo = float(np.mean(v < lo_thr))

    # L2 energy
    l2 = float(np.sqrt(np.mean((v.astype(np.float64) ** 2), dtype=np.float64)))

    return np.array(
        [mean, std, vmin, vmax, l2, frac_hi, frac_lo, *q.tolist()],
        dtype=np.float32,
    )


def _avgpool3d(vol: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """Non-overlapping average pooling from (96,96,96) to out_shape.

    Assumes vol shape is exactly (96,96,96) but does not rely on affine.
    Uses reshape-based pooling (view when possible), minimal extra memory.
    """
    D, H, W = vol.shape
    od, oh, ow = out_shape
    bd, bh, bw = D // od, H // oh, W // ow
    # Crop to multiples (should be exact for 96 divisible by common factors)
    D2, H2, W2 = od * bd, oh * bh, ow * bw
    v = vol[:D2, :H2, :W2]
    # reshape into blocks and average
    v = v.reshape(od, bd, oh, bh, ow, bw)
    return v.mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32, copy=False)


def _block_moments(vol: np.ndarray, grid: Tuple[int, int, int]) -> np.ndarray:
    """Blockwise mean and log-variance features."""
    D, H, W = vol.shape
    gd, gh, gw = grid
    bd, bh, bw = D // gd, H // gh, W // gw
    D2, H2, W2 = gd * bd, gh * bh, gw * bw
    v = vol[:D2, :H2, :W2]
    # reshape into blocks: (gd,bd,gh,bh,gw,bw)
    vb = v.reshape(gd, bd, gh, bh, gw, bw)
    m = vb.mean(axis=(1, 3, 5), dtype=np.float64)
    # variance: E[x^2] - E[x]^2
    m2 = (vb.astype(np.float64) ** 2).mean(axis=(1, 3, 5))
    var = np.maximum(m2 - m * m, 0.0)
    # flatten
    m = m.astype(np.float32, copy=False).ravel()
    lv = _safe_log1p(var.astype(np.float32, copy=False)).ravel()
    return np.concatenate([m, lv], axis=0)


def _dct_lowfreq_energy(pooled: np.ndarray, k: int = 6) -> np.ndarray:
    """Low-frequency 3D DCT energy features from a small pooled volume.

    pooled: small 3D array, e.g., (24,24,24) float32.
    k: keep k lowest frequencies along each axis -> k^3 coefficients.

    Uses scipy.fftpack.dct if available; otherwise scipy.fft.dct.
    """
    # Import lazily to keep module import lightweight
    try:
        from scipy.fft import dctn  # type: ignore

        coeff = dctn(pooled.astype(np.float64, copy=False), type=2, norm="ortho")
    except Exception:
        from scipy.fftpack import dct  # type: ignore

        x = pooled.astype(np.float64, copy=False)
        x = dct(x, axis=0, type=2, norm="ortho")
        x = dct(x, axis=1, type=2, norm="ortho")
        coeff = dct(x, axis=2, type=2, norm="ortho")

    kk = min(k, coeff.shape[0], coeff.shape[1], coeff.shape[2])
    low = coeff[:kk, :kk, :kk]
    # Use signed coefficients and log-energy variants
    low_flat = low.ravel().astype(np.float32)
    energy = np.log1p((low * low).ravel()).astype(np.float32)
    return np.concatenate([low_flat, energy], axis=0)


def _subject_features(vol: np.ndarray) -> np.ndarray:
    """Extract features for one subject volume (96,96,96)."""
    # Ensure we only hold one subject; accept memmap/strided
    v = np.asarray(vol, dtype=np.float32)

    feats = []
    feats.append(_global_stats(v))

    # Multi-scale block moments
    # coarse grid captures regional atrophy patterns; variance captures texture/contrast
    feats.append(_block_moments(v, (12, 12, 12)))  # 1728*2=3456
    feats.append(_block_moments(v, (8, 8, 8)))     # 512*2=1024

    # DCT low-frequency content on a pooled representation
    pooled = _avgpool3d(v, (24, 24, 24))
    feats.append(_dct_lowfreq_energy(pooled, k=6))  # 216*2=432

    out = np.concatenate(feats, axis=0)
    # Replace any non-finite (shouldn't happen) deterministically
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


def _iter_features(x: np.ndarray) -> Iterable[np.ndarray]:
    """Yield per-subject feature vectors."""
    n = x.shape[0]
    for i in range(n):
        yield _subject_features(x[i])


def _build_feature_matrix(x: np.ndarray) -> np.ndarray:
    """Construct (n_samples, n_features) matrix by iterating subjects."""
    n = x.shape[0]
    # First subject to determine dimensionality
    it = _iter_features(x)
    first = next(it)
    p = first.shape[0]
    X = np.empty((n, p), dtype=np.float32)
    X[0] = first
    row = 1
    for f in it:
        X[row] = f
        row += 1
    return X


# -------------------------
# Modeling
# -------------------------

def _cv_select_C(X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> float:
    """Lightweight CV selection for inverse regularization C."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Small grid; works well in high-dim settings
    C_grid = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    best_C = float(C_grid[0])
    best_score = -np.inf

    for C in C_grid:
        fold_scores = []
        for tr, va in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(
                    penalty="l2",
                    solver="saga",
                    C=float(C),
                    max_iter=2000,
                    tol=1e-3,
                    random_state=SEED,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            )
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])[:, 1]
            # roc_auc_score is deterministic
            auc = roc_auc_score(y[va], p)
            fold_scores.append(float(auc))
        m = float(np.mean(fold_scores))
        if m > best_score + 1e-6:
            best_score = m
            best_C = float(C)

    return best_C


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Fit on training data and return positive-class scores for x_eval."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(SEED)

    # Validate minimal assumptions without inspecting data distributions
    if x_train.ndim != 4 or x_eval.ndim != 4:
        raise ValueError("x_train and x_eval must be 4D arrays (N,96,96,96)")
    if x_train.shape[1:] != (96, 96, 96) or x_eval.shape[1:] != (96, 96, 96):
        raise ValueError("Expected volumes of shape (96,96,96)")

    y = np.asarray(y_train).astype(np.int64, copy=False)
    if y.shape[0] != x_train.shape[0]:
        raise ValueError("y_train length must match x_train")

    with _one_thread():
        # Feature extraction (iterating subjects)
        Xtr = _build_feature_matrix(x_train)
        Xev = _build_feature_matrix(x_eval)

        # CV choose C on training only
        C_best = _cv_select_C(Xtr, y, rng)

        # Fit final model on all training
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                penalty="l2",
                solver="saga",
                C=float(C_best),
                max_iter=4000,
                tol=1e-4,
                random_state=SEED,
                n_jobs=1,
                class_weight="balanced",
            ),
        )
        model.fit(Xtr, y)

        scores = model.predict_proba(Xev)[:, 1]

    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    # Ensure finite continuous outputs
    scores = np.clip(scores, 0.0, 1.0)
    return scores
