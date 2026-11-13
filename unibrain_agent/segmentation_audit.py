"""Read-only diagnosis of tissue supervision, geometry, metrics and gradients."""

import argparse
import itertools
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def label_metrics(prediction, reference):
    prediction, reference = np.asarray(prediction).squeeze(), np.asarray(reference).squeeze()
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError('Tissue arrays must share a three dimensional grid')
    for value in (prediction, reference):
        if not np.isfinite(value).all() or not np.isin(value, range(4)).all():
            raise ValueError('Expected finite tissue labels 0, 1, 2, 3')
    confusion = np.bincount((reference.astype(int) * 4 + prediction.astype(int)).ravel(), minlength=16).reshape(4, 4)
    denominator = confusion.sum(0) + confusion.sum(1)
    dice = (2 * confusion.diagonal() + 1e-5) / (denominator + 1e-5)
    permutations = []
    for permutation in itertools.permutations((1, 2, 3)):
        mapped = np.array((0, *permutation))
        intersection = np.array([confusion[mapped[p], p] for p in range(4)])
        sizes = confusion.sum(0) + confusion.sum(1)[mapped]
        permutations.append((float(((2 * intersection + 1e-5) / (sizes + 1e-5))[1:].mean()), permutation))
    best, permutation = max(permutations)
    return {'foreground_macro_dice': float(dice[1:].mean()), 'per_class_dice': dice.tolist(),
            'confusion_reference_rows': confusion.tolist(), 'prediction_counts': confusion.sum(0).tolist(),
            'reference_counts': confusion.sum(1).tolist(), 'best_permuted_dice': best,
            'best_permutation': list(permutation), 'permutation_is_diagnostic_only': True}


def center(mask):
    coordinates = np.argwhere(mask)
    return coordinates.mean(0).tolist() if len(coordinates) else None


def subject_geometry(subjects, row):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    folder = subjects / str(row[0])
    fields = ('raw_img_np', 'raw_img_brain_mask_np', 'raw_img_gm_mask_np', 'raw_img_aal_mask_np')
    images = [nib.load(folder / (name + '.nii.gz')) for name in fields]
    result = {'same_affine': all(np.allclose(image.affine, images[0].affine, atol=1e-5) for image in images),
              'same_shape': all(image.shape == images[0].shape == (96, 96, 96) for image in images),
              'axis_codes': [list(nib.aff2axcodes(image.affine)) for image in images],
              'master_matches_nifti': [bool(np.allclose(image.get_fdata(), row[index + 2], atol=1e-6)) for index, image in enumerate(images)]}
    qc = json.loads((folder / 'qc.json').read_text())
    source = subjects.parent / qc['source_dseg']
    native = nib.load(folder / 'tissue_native_grid_2p7mm.nii.gz')
    resampled = resample_from_to(nib.load(source), (native.shape, native.affine), order=0).get_fdata()
    if not np.isin(resampled, range(4)).all():
        raise ValueError('Source tissue mask contains unexpected labels')
    remapped = np.array([0, 2, 3, 1])[resampled.astype(int)]
    result['source_remapping_matches_native_mask'] = bool(np.array_equal(remapped, native.get_fdata()))
    result['source_to_target_label_map'] = [0, 2, 3, 1]
    result['boundary'] = 'Resampling and shared-grid consistency; not expert anatomical validation'
    return result


def gradient_summary(loss, module):
    import torch
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    observed = [gradient for gradient in gradients if gradient is not None]
    return {'parameter_tensors': len(parameters), 'connected_tensors': len(observed),
            'nonzero_tensors': sum(bool(torch.count_nonzero(gradient)) for gradient in observed),
            'finite': all(bool(torch.isfinite(gradient).all()) for gradient in observed),
            'l2': float(sum(gradient.double().square().sum() for gradient in observed).sqrt()) if observed else 0.}


def collect(args):
    import torch
    from torch.nn import functional as F
    from types import SimpleNamespace
    from unibrain_agent.data import OriginalDataset, digest
    from unibrain_agent.full_model import build_full_model
    from unibrain_agent.replacement_training import finalize_pipeline, original_losses
    from unibrain_agent.replacement_evaluation import code_audit

    run = args.run.resolve()
    report = json.loads((run / 'report.json').read_text())
    if report['status'] not in {'complete', 'budget_exhausted'}:
        raise ValueError('Audit only a stopped run; active experiments are not modified')
    config = report['config']
    if config['method'] not in {'original', 'synthstrip'} or config.get('secondary_extraction'):
        raise ValueError('This probe currently supports original and SynthStrip full hosts')
    checkpoint = run / 'best.pt'
    checkpoint_hash = digest(checkpoint)
    expected = report.get('test', {}).get('checkpoint_sha256')
    if expected and checkpoint_hash != expected:
        raise ValueError('Best checkpoint differs from the evaluated checkpoint')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    implementation = code_audit(payload['identity'])
    if payload['identity']['method'] != config['method'] or payload['progress']['best_epoch'] != report['selection']['best_epoch']:
        raise ValueError('Checkpoint and report disagree')
    source = args.source or Path(config['source'])
    adapter = args.adapter or (Path(config['adapter']) if config.get('adapter') else None)
    if adapter and digest(adapter) != report['architecture']['generated_adapter_sha256']:
        raise ValueError('Adapter differs from training')
    model, _, architecture = build_full_model(source, adapter)
    model = finalize_pipeline(model, architecture, SimpleNamespace(**config))
    model.load_state_dict(payload['model'], strict=True)
    del payload
    model.to(args.device).eval()
    torch.set_num_threads(args.threads)
    from utils import compute_label_dice_v2

    data = args.data or Path(config['data'])
    template_names = ('template_img_orig_96.npy', 'template_img_gm_mask_orig_96.npy', 'template_img_aal_mask_orig_96.npy')
    templates = [torch.from_numpy(np.load(data / name)).float().to(args.device).reshape(1, 1, 96, 96, 96) for name in template_names]
    dataset = OriginalDataset(args.master or data / 'adhd_val_sample.npy')
    membership = np.load(args.indices, allow_pickle=False) if args.indices else np.arange(len(dataset))
    evidence = {'version': 1, 'run_id': run.name, 'dataset': config['dataset'], 'fold': config['fold'],
                'checkpoint_sha256': checkpoint_hash, 'checkpoint_epoch': report['selection']['best_epoch'],
                'device': args.device, 'optimizer_steps': 0, 'model_mode': 'eval with autograd enabled',
                'implementation': implementation,
                'training_target': 'Template tissue labels transferred into subject space',
                'evaluation_target': 'Subject automated tissue reference',
                'physical_geometry': 'Not verified by NPY arrays; check source NIfTI affine and resampling provenance',
                'template_counts': np.bincount(templates[1].detach().cpu().numpy().astype(int).ravel(), minlength=4).tolist(),
                'cases': [], 'saved_prediction_check': {}}
    from unibrain_agent.full_connectivity import transfer
    evidence['operator_evidence'] = {function.__qualname__: {'source': inspect.getsource(function),
        'sha256': hashlib.sha256(inspect.getsource(function).encode()).hexdigest()}
        for function in (transfer, type(model).forward, original_losses)}
    indices = np.linspace(0, len(membership) - 1, min(args.samples, len(membership)), dtype=int)
    for index in indices:
        row = dataset[int(membership[index])]
        image, brain, tissue = [torch.tensor(row[i], dtype=torch.float32, device=args.device).reshape(1, 1, 96, 96, 96) for i in (2, 3, 4)]
        label = torch.tensor([int(row[1])], device=args.device)
        output = model(templates[0], image, templates[1], templates[2], if_train=False)
        logits = output[5][0] if isinstance(output[5], list) else output[5]
        target = output[6]
        if target.shape[1] != 4:
            target = F.one_hot(target[:, 0].long(), 4).movedim(-1, 1).float()
        segmentation = original_losses(output, brain, label)['segmentation']
        prediction, reference, transferred = [value.detach().cpu().numpy().squeeze() for value in
                                              (logits.argmax(1), tissue, target.argmax(1))]
        actual = label_metrics(prediction, reference)
        grid = model.last_inverse_grid if hasattr(model, 'last_inverse_grid') else F.affine_grid(output[4][-1], image.shape, align_corners=True)
        predicted_to_training = label_metrics(prediction, transferred)
        training_to_reference = label_metrics(transferred, reference)
        direct_template = label_metrics(templates[1].detach().cpu().numpy(), reference)
        cross_entropy = F.cross_entropy(logits, tissue[:, 0].long(), reduction='none')
        foreground = tissue[:, 0] != 0
        foreground_ce = float(cross_entropy[foreground].mean()) if foreground.any() else None
        axes = []
        for axis in range(3):
            axes.append(label_metrics(np.flip(prediction, axis), reference)['foreground_macro_dice'])
        row_result = {'validation_index': int(index), 'segmentation_loss': float(segmentation.detach()),
                      'subject_reference_ce': float(cross_entropy.mean().detach()), 'subject_foreground_ce': foreground_ce,
                      'prediction_to_reference': actual, 'prediction_to_training_target': predicted_to_training,
                      'training_target_to_reference': training_to_reference, 'identity_template_to_reference': direct_template,
                      'axis_flip_dice_diagnostic_only': axes,
                      'upstream_dice': float(compute_label_dice_v2(logits.argmax(1), tissue, 'gm')),
                      'target_channel_sum_min': float(target.sum(1).min().detach()),
                      'target_channel_sum_max': float(target.sum(1).max().detach()),
                      'target_requires_grad': target.requires_grad,
                      'inverse_grid_outside_fraction': float((grid.abs() > 1).any(-1).float().mean().detach()),
                      'affine_determinant': float(torch.linalg.det(output[3][-1][:, :, :3]).detach().item()),
                      'prediction_foreground_center': center(prediction > 0), 'reference_foreground_center': center(reference > 0),
                      'reference_outside_brain_fraction': float(((reference > 0) & (brain.cpu().numpy().squeeze() < .5)).sum() / max(1, (reference > 0).sum()))}
        row_result['metric_absolute_error'] = abs(actual['foreground_macro_dice'] - row_result['upstream_dice'])
        outside = (grid.abs() > 1).any(-1)
        row_result['subject_tissue_outside_grid_fraction'] = float(outside[foreground].float().mean()) if foreground.any() else None
        brain_foreground = brain[:, 0] > .5
        row_result['subject_brain_outside_grid_fraction'] = float(outside[brain_foreground].float().mean()) if brain_foreground.any() else None
        bottom = torch.tensor([[[0., 0., 0., 1.]]], device=args.device)
        affine = torch.cat((output[3][-1], bottom), dim=1)
        inverse = torch.cat((output[4][-1], bottom), dim=1)
        row_result['affine_inverse_roundtrip_max_error'] = float((affine @ inverse - torch.eye(4, device=args.device)).abs().max().detach())
        target_ce = F.cross_entropy(logits, target, reduction='none')
        target_foreground = target.argmax(1) != 0
        row_result['target_foreground_fraction'] = float(target_foreground.float().mean())
        row_result['target_foreground_ce'] = float(target_ce[target_foreground].mean().detach()) if target_foreground.any() else None
        row_result['target_background_ce'] = float(target_ce[~target_foreground].mean().detach()) if (~target_foreground).any() else None
        if args.subjects:
            row_result['physical_geometry'] = subject_geometry(args.subjects, row)
            evidence['physical_geometry'] = 'Source remapping, shared NIfTI geometry and master-array consistency checked for sampled validation cases'
        row_result['reference_class_intensity_means'] = [float(row[2][reference == label].mean()) if (reference == label).any() else None for label in range(4)]
        if int(index) == int(indices[0]):
            row_result['segmentation_gradient'] = {name: gradient_summary(segmentation, getattr(model, name)) for name in ('seg_net', 'reg_net', 'ext_net')}
            detached_target_loss = F.cross_entropy(logits, target.detach())
            row_result['detached_target_gradient'] = {name: gradient_summary(detached_target_loss, getattr(model, name)) for name in ('seg_net', 'reg_net', 'ext_net')}
        evidence['cases'].append(row_result)
        print(json.dumps({'validation_index': int(index), 'loss': row_result['segmentation_loss'],
                          'dice_subject': actual['foreground_macro_dice'], 'dice_target': predicted_to_training['foreground_macro_dice'],
                          'target_dice_subject': training_to_reference['foreground_macro_dice']}), flush=True)
        del output, logits, target, segmentation, cross_entropy, target_ce
        if int(index) == int(indices[0]):
            del detached_target_loss
    row = dataset[int(membership[indices[0]])]
    image, brain = [torch.tensor(row[i], dtype=torch.float32, device=args.device).reshape(1, 1, 96, 96, 96) for i in (2, 3)]
    model.train()
    torch.manual_seed(config['seed'])
    output = model(templates[0], image, templates[1], templates[2], if_train=True)
    segmentation = original_losses(output, brain, torch.tensor([int(row[1])], device=args.device))['segmentation']
    gradients = {name: gradient_summary(segmentation, getattr(model, name)) for name in ('seg_net', 'reg_net', 'ext_net')}
    detached_output = list(output)
    detached_output[6] = output[6].detach()
    detached = original_losses(detached_output, brain, torch.tensor([int(row[1])], device=args.device))['segmentation']
    evidence['training_mode_probe'] = {'scope': 'One unaugmented validation input; training-mode backward only; no optimizer',
                                       'segmentation_loss': float(segmentation.detach()), 'segmentation_gradient': gradients,
                                       'detached_target_gradient': {name: gradient_summary(detached, getattr(model, name)) for name in ('seg_net', 'reg_net', 'ext_net')}}
    saved = []
    for folder in sorted((run / 'predictions').glob('case_*')):
        prediction = np.load(folder / 'tissue_prediction.npy', allow_pickle=False)
        reference = np.load(folder / 'tissue_reference.npy', allow_pickle=False)
        saved.append(label_metrics(prediction, reference))
    if saved:
        evidence['saved_prediction_check'] = {'count': len(saved), 'foreground_macro_dice': float(np.mean([r['foreground_macro_dice'] for r in saved])),
                                             'per_class_dice': np.mean([r['per_class_dice'] for r in saved], axis=0).tolist(),
                                             'reported_test_dice': report.get('test', {}).get('metrics', {}).get('tissue_dice')}
    evidence['historical_gradients'] = report.get('gradient_verification')
    evidence['status'] = 'complete'
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--master', type=Path)
    parser.add_argument('--indices', type=Path)
    parser.add_argument('--subjects', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if bool(args.master) != bool(args.indices):
        parser.error('Provide both master and validation indices')
    if args.samples < 1 or args.threads < 1 or args.output.exists():
        parser.error('Use positive budgets and a new output file')
    if args.run.resolve() in args.output.resolve().parents:
        parser.error('Keep diagnostic artifacts outside the training run')
    evidence = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
