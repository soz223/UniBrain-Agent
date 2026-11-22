"""Compare fixed UniBrain checkpoints with registration BatchNorm ablated."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unibrain_agent.anchored_taskforce import classification_probe
from unibrain_agent.data import OriginalDataset
from unibrain_agent.normalization_diagnostics import sample_statistics
from unibrain_agent.paper_original import build_paper_model
from unibrain_agent.training_control import normalize_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if state['identity'].get('template_normalization') != 'minmax':
        raise ValueError('This audit requires a minmax template checkpoint')
    model, _, _ = build_paper_model(args.source)
    model.load_state_dict(state['model'], strict=True)
    model.cuda().eval()
    dataset = OriginalDataset(args.data / 'adhd_val_sample.npy')
    arrays = [np.load(args.data / name) for name in
              ('template_img_orig_96.npy', 'template_img_gm_mask_orig_96.npy', 'template_img_aal_mask_orig_96.npy')]
    arrays[0], _ = normalize_template(arrays[0], 'minmax')
    templates = [torch.as_tensor(a, dtype=torch.float32, device='cuda').reshape(1, 1, 96, 96, 96) for a in arrays]
    report = {'scope': 'Diagnostic validation ablation; no training or test-set access',
              'epoch': state['progress']['epoch'], 'subjects': len(dataset), 'parameter_updates': 0,
              'checkpoint_identity': state['identity'],
              'normalization_scope': 'One image at a time; registration BatchNorm only'}
    cohort_path = args.data / 'cohort.json'
    if cohort_path.is_file():
        cohort = json.loads(cohort_path.read_text())
        report['data'] = {key: cohort.get(key) for key in ('dataset', 'fold', 'master_sha256')}
    report['stored_statistics'] = classification_probe(model, dataset, templates)
    print('stored_statistics', report['stored_statistics'], flush=True)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    with sample_statistics(model.reg_net):
        report['registration_sample_statistics'] = classification_probe(model, dataset, templates)
    report['buffers_unchanged'] = all(torch.equal(buffers[n], b) for n, b in model.named_buffers())
    if not report['buffers_unchanged']:
        raise RuntimeError('Diagnostic forward changed persistent buffers')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('registration_sample_statistics', report['registration_sample_statistics'], flush=True)


if __name__ == '__main__':
    main()
