# Training fold plans for nnU-Net

The full model candidate uses the official PlainConvUNet architecture specified
by a dataset plan. It requires a `3d_fullres` patch size of 96 cubed. The program
below exports only the current fold's training array and invokes the real
nnU-Net planner. It does not estimate a replacement plan itself.

Export an existing fold directory created for the full trainer:

```sh
python scripts/prepare_nnunet.py export \
  --data /path/to/exported/fold --output /path/to/new/planning/workspace \
  --dataset abide --fold 1 --dataset-id 501 --coordinates normalized_voxel
```

Only `adhd_train_sample.npy` is opened. Validation and test arrays are not read.
The source data remain unchanged. Each case is exported as a NIfTI image and
automatically generated tissue reference, with a private ID mapping, file hashes
and source training array hash in `export.json`. Generic tissue class names
avoid assuming an undocumented class ordering. These are not manual labels.

Structured NPY arrays do not contain native orientation or spacing. The explicit
`normalized_voxel` convention writes unit grid spacing with unknown NIfTI spatial
units. This is a numerical planning coordinate system, not millimetres. If all
images have verified common spacing in their current array axes, use
`--coordinates physical_mm --spacing SX SY SZ --spacing-source DESCRIPTION`.
The affine still has synthetic axis directions and zero origin; this does not
recover native patient orientation. Do not use raw acquisition spacing after
resizing unless it has been transformed correctly.
NIfTI axes are written in reverse array order so SimpleITK reconstructs the
original tensor axes. This permutation is recorded in the manifest. Plans that
require a further axis permutation are currently marked incompatible.

Install the official `nnunetv2` package and its declared dependencies in a
separate Python environment. No installation or download occurs in this script.
The pinned architecture source required by this repository remains managed by
`scripts/bootstrap.py`. The planning environment additionally needs the complete
nnU-Net fingerprinting and planning dependencies, including batchgenerators,
acvl-utils, SimpleITK and dynamic-network-architectures. Planner version and
source hash are recorded on each run.

```sh
python scripts/prepare_nnunet.py plan \
  --output /path/to/new/planning/workspace \
  --python /path/to/nnunet/environment/python --processes 2
```

The command invokes `nnUNetv2`'s `ExperimentPlanner` with integrity checks and
`--no_pp`. It performs CPU fingerprinting and planning, not training or
preprocessing. CUDA visibility is empty for the child process. Logs and failures
are retained under `planner_attempts`. Every exported file is rechecked before
planning, and unexpected files in the raw dataset directory are rejected.

`plan.json` records the generated plan path, hash and compatibility. If the real
planner selects another patch size or architecture, the result is marked
`incompatible` and preserved unchanged. The program never rewrites a plan to
make it appear compatible. Missing dependencies produce `blocked` with a
retained diagnostic. A compatible plan still requires actual candidate and full
pipeline execution checks before it can support a training claim.

Pass the recorded `nnUNetPlans.json` path to `train_replacement.py --plans` and
retain its adjacent `.provenance.json` sidecar. The trainer requires this sidecar,
checks its dataset and fold against `--dataset` and `--fold`, and verifies both
the training array hash and plan hash. Use a separate planning workspace
and dataset ID for each fold. The current integration imports the planned
architecture and uses UniBrain inputs and objectives; it is not a reproduction
of nnU-Net's complete preprocessing, augmentation, loss or training pipeline.
