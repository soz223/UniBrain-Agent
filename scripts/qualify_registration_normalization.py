"""Check the production normalization in a complete, unchanged checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unibrain_agent.data import OriginalDataset
from unibrain_agent.paper_original import build_paper_model
from unibrain_agent.registration_normalization import ImageBatchNorm3d
from unibrain_agent.normalization_diagnostics import sample_statistics
from unibrain_agent.training_control import normalize_template
from unibrain_agent.replacement_training import evaluate, sample_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'data', 'checkpoint', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model, _, _ = build_paper_model(args.source, registration_normalization='per_image')
    model.load_state_dict(state['model'], strict=True)
    model.cuda().eval()
    data = OriginalDataset(args.data / 'adhd_val_sample.npy')
    arrays = [np.load(args.data / name) for name in
              ('template_img_orig_96.npy', 'template_img_gm_mask_orig_96.npy', 'template_img_aal_mask_orig_96.npy')]
    arrays[0], _ = normalize_template(arrays[0], state['identity']['template_normalization'])
    templates = [torch.as_tensor(a, device='cuda', dtype=torch.float32).reshape(1, 1, 96, 96, 96) for a in arrays]
    image, _, _, _, label = sample_tensors(data[0])
    image.requires_grad_(True)
    model.train()
    output = model(templates[0], image, templates[1], templates[2], if_train=True)
    gradient = torch.autograd.grad(torch.nn.functional.cross_entropy(output[-1], label), image)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    del output
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(templates[0], image.detach(), templates[1], templates[2], if_train=False)[-1].clone()
    layers = [layer for layer in model.reg_net.modules() if isinstance(layer, ImageBatchNorm3d)]
    for layer in layers:
        layer.__class__ = torch.nn.BatchNorm3d
    with sample_statistics(model.reg_net), torch.no_grad():
        diagnostic = model(templates[0], image.detach(), templates[1], templates[2], if_train=False)[-1]
    torch.testing.assert_close(expected, diagnostic, rtol=0, atol=0)
    for layer in layers:
        layer.__class__ = ImageBatchNorm3d
    result = {'diagnostic_matches_production': True, 'input_gradient_finite_nonzero': True,
              'parameter_updates': 0, 'normalization': 'per_image',
              'validation': evaluate(model, data, templates)['metrics']}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
