"""Full-cohort classification and site diagnostics without optimizer updates."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def summarize(cases, seed=17):
    from sklearn.metrics import roc_auc_score, log_loss

    labels = np.array([c['label'] for c in cases])
    probabilities = np.array([c['probability'] for c in cases])
    if not len(labels):
        return {'n': 0, 'auc': None}
    result = {'n': len(labels), 'positive': int(labels.sum()),
              'accuracy': float(((probabilities >= .5) == labels).mean()),
              'classification_loss': float(log_loss(labels, probabilities, labels=[0, 1])),
              'brier': float(np.mean((probabilities - labels) ** 2)),
              'confident_error_fraction': float(np.mean(((probabilities >= .5) != labels) &
                                                        (np.maximum(probabilities, 1 - probabilities) >= .9)))}
    if len(np.unique(labels)) != 2:
        return {**result, 'auc': None, 'auc_ci95': None}
    result['auc'] = float(roc_auc_score(labels, probabilities))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = []
    for _ in range(500):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(roc_auc_score(labels[indices], probabilities[indices]))
    result['auc_ci95'] = np.quantile(values, [.025, .975]).tolist()
    return result


def diagnose_cases(train, validation, probe_ids):
    probe_ids = set(probe_ids)
    summaries = {'train_full': summarize(train), 'validation': summarize(validation),
                 'train_probe': summarize([c for c in train if c['id'] in probe_ids]),
                 'train_complement': summarize([c for c in train if c['id'] not in probe_ids])}
    sites = sorted({c['site'] for c in train + validation})
    per_site = {s: {name: summarize([c for c in rows if c['site'] == s])
                    for name, rows in [('train', train), ('validation', validation)]} for s in sites}
    counts = {s: [c['label'] for c in train if c['site'] == s] for s in sites}
    global_rate = (sum(c['label'] for c in train) + 1) / (len(train) + 2)
    site_predictions = [{**c, 'probability': (sum(counts[c['site']]) + 1) / (len(counts[c['site']]) + 2)
                         if counts[c['site']] else global_rate} for c in validation]
    summaries['site_only_validation'] = summarize(site_predictions)
    train_auc, val_auc = summaries['train_full']['auc'], summaries['validation']['auc']
    gap = None if train_auc is None or val_auc is None else train_auc - val_auc
    return {'summaries': summaries, 'sites': per_site, 'auc_gap': gap,
            'interpretation': 'Training-set fitting gap; causation is not identified by this audit',
            'recommended_experiment': 'cross_subject_training_reward' if gap is not None and gap > .1 else 'continue_observation',
            'validation_used_for_policy_reward': False, 'test_evaluated': False}


def audit(args):
    import torch
    from .data import OriginalDataset, digest
    from .replacement_training import save_json
    from .training_control import training_probe_indices, normalize_template

    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {s: OriginalDataset(args.data / f'adhd_{s}_sample.npy') for s in ('train', 'val')}
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    sites = {r['id']: r['raw_img'].replace('\\', '/').split('/')[1] for r in manifest}
    ids = {s: list(map(str, d.array['id'])) for s, d in datasets.items()}
    overlap = sorted(set(ids['train']) & set(ids['val']))
    if overlap or any(len(set(values)) != len(values) for values in ids.values()):
        raise ValueError('Duplicate subjects or train-validation overlap')
    missing = sorted(set(ids['train'] + ids['val']) - sites.keys())
    if missing:
        raise ValueError('Site manifest lacks subjects')
    metadata = {s: {'n': len(d), 'positive': int(d.array['label'].sum()),
                    'sites': dict(Counter(sites[v] for v in ids[s]))} for s, d in datasets.items()}
    save_json(args.output / 'metadata.json', {'splits': metadata, 'overlap': overlap,
              'manifest_sha256': digest(args.manifest), 'test_evaluated': False})
    if args.metadata_only:
        return metadata
    if args.checkpoint is None or args.source is None:
        raise ValueError('Checkpoint and pinned source are required')
    from .full_replacement_worker import available_gpu, selected_device
    device = selected_device(args.gpu)
    available_gpu(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    torch.set_num_threads(2)
    checkpoint_hash = digest(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    identity = state['identity']
    if not identity.get('paper_original') or identity['method'] != 'original':
        raise ValueError('This audit requires the complete original paper model')
    from .replacement_evaluation import code_audit
    code = code_audit(identity)
    for name in [f'adhd_{s}_sample.npy' for s in datasets] + [
            'template_img_orig_96.npy', 'template_img_gm_mask_orig_96.npy', 'template_img_aal_mask_orig_96.npy']:
        if digest(args.data / name) != identity['data_hashes'][name]:
            raise ValueError('Checkpoint data mismatch: ' + name)
    from .paper_original import build_paper_model
    model, _, architecture = build_paper_model(args.source, identity.get('paper_registration_weight', .1))
    model.load_state_dict(state['model'], strict=True)
    model.cuda().eval()
    arrays = [np.load(args.data / name) for name in (
        'template_img_orig_96.npy', 'template_img_gm_mask_orig_96.npy', 'template_img_aal_mask_orig_96.npy')]
    arrays[0], _ = normalize_template(arrays[0], identity.get('template_normalization', 'none'))
    templates = [torch.as_tensor(a, dtype=torch.float32, device='cuda').reshape(1, 1, 96, 96, 96) for a in arrays]
    cases = {s: [] for s in datasets}
    with torch.no_grad(), (args.output / 'cases.jsonl').open('w', encoding='utf-8') as stream:
        for split, dataset in datasets.items():
            for index in range(len(dataset)):
                row = dataset[index]
                image = torch.as_tensor(row[2], dtype=torch.float32, device='cuda').reshape(1, 1, 96, 96, 96)
                logits = model(templates[0], image, templates[1], templates[2], if_train=False)[-1]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite classification output')
                case = {'id': str(row[0]), 'label': int(row[1]), 'site': sites[str(row[0])],
                        'probability': float(logits.softmax(1)[0, 1])}
                cases[split].append(case)
                stream.write(json.dumps({'split': split, **case}) + '\n')
                if index % 16 == 0:
                    stream.flush()
                    save_json(args.output / 'progress.json', {'split': split, 'completed': index + 1, 'total': len(dataset)})
    indices = training_probe_indices(datasets['train'].array['label'], identity['train_probe_size'], identity['seed'])
    result = diagnose_cases(cases['train'], cases['val'], [ids['train'][i] for i in indices])
    result.update(status='complete', checkpoint_sha256=checkpoint_hash, step=state['progress']['steps'],
                  epoch=state['progress']['epoch'], architecture=architecture, code_audit=code,
                  optimizer_updates=0, evaluation_mode='model.eval; if_train=False; identical train and validation path')
    if digest(args.checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation')
    save_json(args.output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--metadata-only', action='store_true')
    result = audit(parser.parse_args())
    print(json.dumps(result.get('summaries', result)))


if __name__ == '__main__':
    main()
