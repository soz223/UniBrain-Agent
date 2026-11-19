import numpy as np


def _features(image):
    sample = image[::2, ::2, ::2]
    sample = sample[np.isfinite(sample) & (sample != 0)]
    if sample.size == 0:
        sample = image[np.isfinite(image) & (image != 0)]
    low, high = np.percentile(sample, [2, 98]) if sample.size else (0, 1)
    foreground = np.isfinite(image) & (image != 0)
    volume = np.zeros_like(image, subok=False)
    if high > low:
        values = (image[foreground].astype(np.float64) - low) / (high - low)
        volume[foreground] = 0.1 + 0.9 * np.clip(values, 0, 1)
    else:
        volume[foreground] = 1
    blocks = volume.reshape(24, 4, 24, 4, 24, 4)
    fine = blocks.mean(axis=(1, 3, 5))
    coarse = fine.reshape(12, 2, 12, 2, 12, 2).mean(axis=(1, 3, 5))
    return np.concatenate((fine.ravel(), coarse.ravel()))


def _matrix(images):
    matrix = np.empty((len(images), 24**3 + 12**3), dtype=np.float32)
    for row, image in enumerate(images):
        matrix[row] = _features(image)
    return matrix


def fit_predict(train_images, train_labels, eval_images):
    labels = np.asarray(train_labels).reshape(-1)
    if labels.size != len(train_images) or not labels.size:
        raise ValueError("Expected one label per training image")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary labels encoded as 0 and 1")
    if len(eval_images) == 0 or np.all(labels == labels[0]):
        return np.full(len(eval_images), labels.mean())

    train = _matrix(train_images)
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 0.05)
    # Give each resolution comparable weight in the regularization penalty.
    scale[:24**3] *= np.sqrt(24**3)
    scale[24**3:] *= np.sqrt(12**3)
    train -= center
    train /= scale
    # Solve ridge regression in subject space to keep the system small.
    gram = (train @ train.T).astype(np.float64)
    gram.flat[::len(train) + 1] += 0.5
    prevalence = labels.mean()
    weights = train.T @ np.linalg.solve(gram, labels - prevalence)
    evaluation = _matrix(eval_images)
    evaluation -= center
    evaluation /= scale
    scores = prevalence + evaluation @ weights
    return np.clip(scores, 0, 1)
