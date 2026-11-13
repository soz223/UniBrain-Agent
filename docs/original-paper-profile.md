# Original paper profile

The `--paper-original` training option implements the original UniBrain method using the supplied KDD 2024 manuscript as the specification. The pinned public source remains immutable. This profile is separate from both the public-code reproduction and the connected-host replacement experiments.

The extraction and segmentation convolution widths, shared five-stage affine network, Adam learning rate of 0.00001, weight decay of 0.000001, batch size of one, and image augmentation follow Appendix A.4. ROI features have widths 256, 256, 256. Two GCN layers have width 128, followed by a 128-unit MLP and a binary output. The MLP activation and binary output placement are implementation choices where the manuscript is not explicit.

Tissue and atlas labels are transferred as multi-channel probabilities with trilinear interpolation. The segmentation loss uses the transferred template, including its gradient path to registration. Subject tissue and atlas references are used only for evaluation. The fixed objective weights are extraction 1, registration 0.1, segmentation 1, classification 1. These coefficients are validation-selected hyperparameters in the manuscript, not learnable weights. The upstream 1 minus CC objective differs from the paper's negative CC by a constant and has identical gradients.

The graph uses L2-normalized node similarities and excludes negative edges. Feature and edge perturbation rates are 0.01. Retained edges preserve their similarity weights and gradients. Randomly added edges use unit weight; the manuscript does not specify their weight or symmetry policy. This implementation preserves the public code's directed perturbation convention. Border padding for categorical interpolation also follows the existing integration convention and is an explicit implementation choice.

The unused standalone parcellation predictor from the public source is omitted: the paper defines parcellation through inverse template transfer. No subject segmentation supervision, TaskForce coordination, module replacement, detached target, or added regularization is enabled.

Use the existing dataset folds. Current full cohorts and automatic reference masks differ from the manuscript's cohort sizes and reported reference provenance. Results must be described as paper-method training on the current cohorts, not an exact reproduction of its tables. Select checkpoints using validation only. A short gradient check establishes execution and connectivity, not convergence.

Example:

```sh
python scripts/train_replacement.py --source "$UNIBRAIN_SOURCE" --data "$FOLD_DIRECTORY" --output "$RUN_DIRECTORY" --method original --setting original --paper-original --dataset abide --fold 1 --epochs 20 --patience 0 --gpu 0
```

Run a separate two-update check with `--max-steps 2 --verify-gradients` before full training. Preserve `report.json`, `updates.jsonl`, `best.pt` and `last.pt`. The last checkpoint includes optimizer and RNG state for exact continuation under the recorded configuration.

Validation diagnostics include each foreground tissue Dice, transferred-template versus subject Dice, foreground cross entropy against subject references, inverse-grid coverage of subject foreground, and affine determinant. These measurements do not enter the optimization objective.

On 2026-11-12, the full 271,483,811-parameter profile completed two updates on each preserved-fold acceptance subset from ABIDE and ADHD. Classification gradients reached extraction, registration, feature extraction, classification and the input after the affine initialization warmup. Parameter updates and finite gradients were checked. The ABIDE checkpoint was reloaded in a separate process with strict tensor matching and identical validation classification loss. These are integration checks, not full-cohort accuracy or convergence results.
