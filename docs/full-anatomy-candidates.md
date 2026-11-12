# Full anatomy candidate modules

`build_anatomy_modules` constructs complete candidate networks from the locked
source definitions. Every public module consumes the unchanged single channel
96 cubed tensor used by the full UniBrain host. No preliminary resizing or slice
sampling is performed. The separate host integration controls objectives,
classification connections, initialization and checkpoint provenance.

FastSurfer uses the classic `FastSurferCNN` core with 64 filters, 5 by 5 kernels,
four encoder and decoder stages, competitive dense blocks and seven neighboring
slices. All 96 central slices are evaluated in ordered minibatches. Boundary
neighbors repeat the edge slice. The adapter uses the first tensor spatial axis;
NPY arrays do not justify calling it an anatomical axial view. It is one full
view network, not the official three view ensemble, VINN or surface pipeline.
The classifier has 117 AAL outputs and requires training on the current task.
Official FastSurfer label indices are not converted into AAL labels.
The locked class documents the classic 64 filter architecture; current upstream
configuration defaults target the separate 71 filter VINN model.

HighResNet uses the pinned nobrainer PyTorch registry implementation with
16, 32 and 64 channels, three residual blocks in each stage and dilation rates
1, 2 and 4. Its full volume convolutions preserve the 96 cubed field. Only the
output class count is set to 117 for AAL training. This identifies the exact
registry implementation; it does not claim compatibility with every historical
HighRes3DNet implementation or a pretrained official AAL checkpoint.

FastSurfer and HighResNet use activation recomputation. BatchNorm buffers are
restored after recomputation so each original forward contributes statistics
once. FastSurfer slice minibatch size is an explicit training parameter because
it affects BatchNorm statistics. The native class structure and parameter widths
remain unchanged by this memory optimization.

ERNet replaces both extraction and affine registration. It retains five
extraction stages, five registration stages, gamma 10, the source default
extraction U-Net widths and the full 512 channel affine encoder with its native
128 unit localization head. The extraction module is loaded directly from the
source. The registration adapter preserves cumulative left multiplication and
sampling with `align_corners=True`, and returns affine and inverse matrices in
the interface required by UniBrain. Its coordinate tensors follow the input
device. Native ERNet originally returns warped volumes and grids only; the
additional inverse matrices are a differentiable interface adaptation.

Source definitions are verified against `catalog/sources.json` before building.
Random initialization is reported explicitly. An official checkpoint with a
different output head must not be presented as a compatible AAL model. Source
parsing, small interface tests and full candidate CPU checks establish distinct
levels of evidence; completed full host training remains a separate check.

Run the complete candidate CPU checks sequentially to bound peak memory:

```sh
python scripts/check_anatomy_candidates.py --method ernet --output /path/to/new/ernet.json
python scripts/check_anatomy_candidates.py --method nobrainer --output /path/to/new/nobrainer.json
python scripts/check_anatomy_candidates.py --method fastsurfer --output /path/to/new/fastsurfer.json
```

Each check compares the adapted forward with the native core at full spatial
size, checks finite input and parameter gradients and performs one SGD update.
The tensors are synthetic; this verifies execution, not clinical performance.

Source references:

* [FastSurferCNN](https://github.com/Deep-MI/FastSurfer/blob/6dcd0e949a2dc13ef7e940f33b2842a2ebec1477/FastSurferCNN/models/networks.py)
* [nobrainer HighResNet](https://github.com/neuronets/nobrainer/blob/3109460d048bfa60d8c4e056e86deadd47568f68/nobrainer/models/highresnet.py)
* [ERNet](https://github.com/ERNetERNet/ERNet/blob/f0708bbf13de32ef7e977afd2468f1597477c08e/model.py)
