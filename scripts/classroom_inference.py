"""Run one full UniBrain forward on isolated, administrator-selected demo assets."""

import argparse
import base64
import hashlib
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_image(path):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.dtype.names:
        if array.shape != (1,) or 'raw_img_np' not in array.dtype.names:
            raise ValueError('The classroom asset must contain exactly one image')
        array = array[0]['raw_img_np']
    if array.shape != (96, 96, 96) or not np.isfinite(array).all():
        raise ValueError('Expected one finite 96 cubed image')
    return np.array(array, dtype=np.float32, copy=True)


def install_cpu_affines(module):
    # Only allocation changes; the pinned source and affine equations stay intact.
    module.get_batch_identity_theta_4_4 = lambda batch: torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    module.get_batch_identity_theta_3_4 = lambda batch: torch.eye(4)[:3].unsqueeze(0).repeat(batch, 1, 1)


def png_data(array, categorical=False):
    plane = np.rot90(np.asarray(array)[:, :, 48])
    if categorical:
        palette = np.array([[8, 15, 24], [63, 193, 193], [100, 119, 231], [239, 193, 102]], dtype=np.uint8)
        pixels = palette[plane.astype(np.int64) % len(palette)]
    else:
        low, high = np.percentile(plane, [1, 99])
        gray = np.clip((plane - low) / max(float(high - low), 1e-8) * 255, 0, 255).astype(np.uint8)
        pixels = np.repeat(gray[:, :, None], 3, axis=2)
    pixels = np.repeat(np.repeat(pixels, 3, axis=0), 3, axis=1)

    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value) & 0xffffffff)

    rows = b''.join(b'\x00' + row.tobytes() for row in pixels)
    content = b'\x89PNG\r\n\x1a\n'
    content += chunk(b'IHDR', struct.pack('>IIBBBBB', pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0))
    content += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(content).decode('ascii')


def infer(args):
    from unibrain_agent.paper_original import build_paper_model
    from unibrain_agent.training_control import normalize_template

    if not 1 <= args.threads <= 4:
        raise ValueError('Choose between one and four CPU threads')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_path = args.checkpoint.resolve()
    inputs = [checkpoint_path, args.data.resolve()] + [args.templates.resolve() / name for name in (
        'template_img_orig_96.npy', 'template_img_gm_mask_orig_96.npy', 'template_img_aal_mask_orig_96.npy')]
    if args.output.resolve() in inputs or args.output.exists():
        raise ValueError('Use a new result path outside the input assets')
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    identity = checkpoint.get('identity', {})
    if identity.get('paper_original') is not True or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('A documented paper_original checkpoint is required')
    normalization = identity.get('registration_normalization', 'stored')
    scaling = identity.get('template_normalization', 'none')
    model, _, architecture = build_paper_model(args.source, identity.get('paper_registration_weight', .1), normalization)
    install_cpu_affines(sys.modules['model'])
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.eval().requires_grad_(False)

    image_array = load_image(args.data)
    arrays = [load_image(path) for path in inputs[2:]]
    arrays[0], _ = normalize_template(arrays[0], scaling)
    if not np.isin(arrays[1], [0, 1, 2, 3]).all():
        raise ValueError('The tissue template must have labels zero through three')
    labels = np.unique(arrays[2])
    if len(labels) != 117 or labels[0] != 0 or not np.equal(labels, np.rint(labels)).all():
        raise ValueError('The atlas template must contain AAL116 and background')
    image = torch.from_numpy(image_array).reshape(1, 1, 96, 96, 96)
    templates = [torch.from_numpy(array).reshape_as(image) for array in arrays]
    stages, starts, handles = [], {}, []
    names = {'ext_net': 'Brain extraction', 'reg_net': 'Registration', 'seg_net': 'Tissue segmentation',
             'feature_net': 'ROI feature extraction', 'graph_gen': 'Network generation', 'gnn_pred': 'Classification'}

    def before(name):
        def hook(module, values):
            starts[name] = time.perf_counter()
        return hook

    def after(name):
        def hook(module, values, result):
            value = result
            while isinstance(value, (tuple, list)):
                value = value[-1]
            if not torch.isfinite(value).all():
                raise ValueError('A model stage returned nonfinite values')
            stages.append({'name': names[name], 'seconds': round(time.perf_counter() - starts[name], 3),
                           'output_shape': list(value.shape)})
        return hook

    for name in names:
        module = getattr(model, name)
        handles.extend([module.register_forward_pre_hook(before(name)), module.register_forward_hook(after(name))])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(templates[0], image, templates[1], templates[2], if_train=False)
    seconds = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    extracted, masks, warped, _, _, tissue, _, _, atlas, graph, logits = output
    if logits.shape != (1, 2) or not all(torch.isfinite(value).all() for value in (logits, tissue, atlas, graph)):
        raise ValueError('The full model returned invalid output')
    probabilities = logits.softmax(1)[0].tolist()
    tissue_labels = tissue.argmax(1).reshape(96, 96, 96).numpy()
    atlas_labels = atlas.reshape(96, 96, 96).numpy()
    stages.insert(3, {'name': 'Atlas parcellation', 'output_shape': list(atlas.shape),
                      'observed_regions': int(torch.unique(atlas).numel() - int((atlas == 0).any()))})
    stages[0]['mean_brain_probability'] = round(float(masks[-1].mean()), 4)
    stages[2]['label_counts'] = [int((tissue_labels == label).sum()) for label in range(4)]
    visuals = [('Input MRI', image_array, False), ('Extracted brain', extracted[-1].reshape(96, 96, 96).numpy(), False),
               ('Tissue segmentation', tissue_labels, True), ('Atlas parcellation', atlas_labels, True)]
    result = {
        'sample_id': 'demo',
        'summary': 'Full UniBrain inference completed on one fixed demonstration image.',
        'prediction': {'class_index': int(logits.argmax(1)[0]), 'probabilities': probabilities,
                       'class_names': args.class_names},
        'stages': stages,
        'visualization': [{'title': title, 'image': png_data(array, categorical)} for title, array, categorical in visuals],
        'evidence': {'profile': architecture['profile'], 'parameters': architecture['parameters'],
                     'input_shape': list(image.shape), 'checkpoint_sha256': checkpoint_hash,
                     'device': 'cpu', 'inference_seconds': round(seconds, 3),
                     'registration_normalization': normalization, 'template_normalization': scaling,
                     'training_updates': 0, 'source_commit': 'f9f8d906b1725046d532f862a28876482aca2cfa',
                     'cpu_adaptation': 'Two affine identity allocation helpers use CPU tensors'},
        'notice': args.notice,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'data', 'templates', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--class-names', nargs=2, default=['Class 0', 'Class 1'])
    parser.add_argument('--notice', default='One demonstration sample; no cohort accuracy estimate or clinical interpretation.')
    result = infer(parser.parse_args())
    print(json.dumps({'status': 'complete', 'evidence': result['evidence']}))


if __name__ == '__main__':
    main()
