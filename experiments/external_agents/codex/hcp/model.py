import numpy as np
from sklearn.linear_model import LogisticRegression


def _pool(image, width):
    side = 96 // width
    blocks = image.reshape(side, width, side, width, side, width)
    return blocks.mean(axis=(1, 3, 5), dtype=np.float32)


def _features(image):
    sample = image[::2, ::2, ::2]
    corners = [image[i:i+6, j:j+6, k:k+6].ravel()
               for i in (0, 90) for j in (0, 90) for k in (0, 90)]
    background = np.median(np.concatenate(corners))
    signal = sample[sample > background]
    if signal.size:
        scale = float(np.percentile(signal, 95)) - float(background)
    else:
        scale = 1.0
    intensity = np.subtract(image, background, dtype=np.float32)
    intensity /= max(scale, np.finfo(np.float32).tiny)
    np.clip(intensity, 0, 1.5, out=intensity)

    fine = _pool(intensity, 4)
    groups = [fine.ravel(), _pool(intensity, 8).ravel()]
    profiles = []
    volumes = []
    for threshold in (0.15, 0.50, 0.75):
        occupancy = _pool(intensity > threshold, 8)
        groups.append(occupancy.ravel())
        profiles.extend(occupancy.mean(axis=axes) for axes in ((1, 2), (0, 2), (0, 1)))
        volumes.append(occupancy.mean())
    groups.extend((np.concatenate(profiles), np.asarray(volumes)))
    return groups


def fit_predict(train_images, train_labels, eval_images):
    """Estimate male-class probabilities in evaluation row order."""
    y = np.asarray(train_labels).reshape(-1)
    if not len(y) or len(y) != len(train_images) or not np.isin(y, (0, 1)).all():
        raise ValueError("Expected one binary label per training image.")
    if not len(eval_images):
        return np.empty(0, dtype=np.float64)
    if np.all(y == y[0]):
        return np.full(len(eval_images), (y.sum() + 1) / (len(y) + 2))

    rows = [_features(image) for image in train_images]
    widths = [group.size for group in rows[0]]
    train = np.stack([np.concatenate(groups) for groups in rows])
    del rows
    center = train.mean(axis=0)
    spread = train.std(axis=0)
    # Balance feature groups and limit the influence of low-variance voxels.
    start = 0
    for width in widths:
        block = slice(start, start + width)
        spread[block] = np.maximum(spread[block], 0.02)
        active = np.sum(np.minimum(train[:, block].var(axis=0) / spread[block]**2, 1))
        spread[block] *= np.sqrt(max(active, 1.0))
        start += width
    train -= center
    train /= spread
    model = LogisticRegression(C=4.0, solver="liblinear", dual=True,
                               random_state=0, max_iter=500)
    model.fit(train, y)
    weights = model.coef_[0] / spread
    intercept = model.intercept_[0] - center @ weights
    scores = np.array([np.concatenate(_features(image)) @ weights + intercept
                       for image in eval_images])
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
