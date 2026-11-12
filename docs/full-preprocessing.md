# Full preprocessing components

`build_preprocessor(method)` in `unibrain_agent.full_preprocessing` constructs a
trainable component with a floating tensor input and output of shape
`[1, 1, 96, 96, 96]`. The locked source hashes are verified before construction.
Network widths come from the source defaults or its checked configuration.
No volume is reduced to 16 or 32 cubed. Models expose `core`, `architecture`,
`provenance` and `adaptations` for integration and checkpoint inspection.

| Method | Retained component | Parameters | Integration scope |
| :--- | :--- | ---: | :--- |
| `redwgan` | Original `GeneratorNet`, widths 32, 64, 128, 256 | 2,326,273 | Denoising generator on the full volume; adversarial training is a separate protocol |
| `deepn4` | Original `Synbo_UNet3D`, base width 32 | 19,328,418 | Log field prediction followed by `image * exp(-log_field)` |
| `deepnormalize` | Original preprocessor `UNet`, widths 16, 32, 64, 128, 256 | 4,770,625 | Single T1 input and output; the joint two-modality segmentation system is not reproduced |
| `harmonizing_flows` | Original 2D `Harmonizer` | 816,594 | All 96 slices along tensor axis 2, each zero padded to 224 by 224 and center cropped afterward |

DeepN4 keeps the original log-field interpretation. It does not apply the
compact adapter's tanh bound. Nonfinite values must fail the pipeline checker.
The official CLI's spatial resampling and SciPy smoothing are outside this
tensor component. A pretrained checkpoint requires a separately validated
intensity convention and source provenance.

The Harmonizer uses activation recomputation when gradients are enabled. Every
slice contributes to the output and gradient graph. Padding preserves voxel
resolution but changes the field of view presented to the network. Tensor
axis 2 is declared explicitly and must be matched to the prepared image
orientation. This component does not implement normalizing-flow supervision,
cross-site fitting or test-time adaptation from the complete Harmonizing Flows
method.

All four components initialize randomly unless the caller explicitly loads a
compatible checkpoint. Their presence does not establish denoising, bias
correction or domain harmonization accuracy. Train the selected component
through its task-specific or downstream objective and retain the evaluation
protocol. Full pipeline gradient checks remain necessary after attachment to
UniBrain.

```sh
python -m unittest discover -s tests -p test_full_preprocessing.py -v
```

The CPU checks use the complete 96 cubed input, test finite outputs, input and
parameter gradients, and perform one optimizer update. The Harmonizer check
also requires a nonzero input gradient on every slice. These checks retain the
original architecture and can require substantial CPU memory and several
minutes, especially for the full-resolution RED WGAN generator. Source files
and user data are never modified by these tests.
