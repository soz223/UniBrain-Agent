"""Restrict partial tissue-reference metrics to annotated voxels."""

import numpy as np


def initial_state_digest(model):
    import hashlib
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def attach_validity(datasets, directory):
    for split, dataset in datasets.items():
        array = np.load(directory / f'tissue_valid_{split}.npy', mmap_mode='r', allow_pickle=False)
        if array.dtype != np.bool_ or array.shape != (len(dataset), 96, 96, 96):
            raise ValueError('Invalid tissue validity array')
        dataset.tissue_validity = array


def validity_for(dataset, index, device):
    import torch
    if isinstance(dataset, torch.utils.data.Subset):
        return validity_for(dataset.dataset, dataset.indices[index], device)
    values = getattr(dataset, 'tissue_validity', None)
    if values is None:
        return None
    mask = np.array(values[index], copy=True)
    if not mask.any():
        raise ValueError('Tissue reference has no annotated voxels')
    return torch.as_tensor(mask, device=device).reshape(1, 1, *mask.shape)
