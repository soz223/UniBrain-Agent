# Training fold plans for nnU-Net

The full model candidate uses the official PlainConvUNet architecture specified
by a dataset plan. Its execution volume is 96 cubed. The native planned patch
may differ: all architecture stages and weights are retained, and the cumulative
native strides must divide 96 along every axis. This explicit full volume
execution adaptation does not rewrite the official plan. The program
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
NIfTI axes are written in reverse array order so official NibabelIO reconstructs
the original tensor axes. NibabelIO supports Windows Unicode paths where older
SimpleITK readers fail. This permutation is recorded in the manifest. Plans that
require a further axis permutation are currently marked incompatible.

Install the official `nnunetv2` package and its declared dependencies in a
separate Python 3.10 environment. The following Linux commands keep the original
training environment unchanged and explicitly select CPU PyTorch wheels:

```sh
python3.10 -m venv .venv-planner
.venv-planner/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv-planner/bin/python -m pip install -r requirements.txt -r requirements-planner.txt
.venv-planner/bin/python -m pip check
```

On Windows, use `py -3.10` to create the environment and replace `bin/python`
with `Scripts/python.exe`. Do not use `--no-deps` for a fresh installation.
No installation or download occurs in the preparation script itself.
The pinned architecture source required by this repository remains managed by
`scripts/bootstrap.py`. The planning environment additionally needs the complete
nnU-Net fingerprinting and planning dependencies, including batchgenerators,
acvl-utils, SimpleITK and dynamic-network-architectures. Planner version and
source hash are recorded on each run.
The local acceptance used Windows Python 3.10.18 with inherited PyTorch
2.5.1+cu121, executed on CPU. Final `pip check` passed after adding imagecodecs
and unittest2's declared argparse, traceback2 and linecache2 dependencies.
The pinned file records the tested numerical and planner package versions;
transitive dependencies are resolved normally. A fresh Linux installation has
not yet been executed locally. Verify `pip check` and run the CPU acceptance
there before claiming that platform has passed.

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

`plan.json` records the generated plan path, hash, native patch, execution shape,
cumulative strides and compatibility. Unsupported architectures, tensor axis
permutations or strides that do not divide the execution shape are marked
`incompatible`. The native plan is preserved unchanged, including its patch
size. Missing dependencies produce `blocked` with a
retained diagnostic. A compatible plan still requires actual candidate and full
pipeline execution checks before it can support a training claim.

Pass the recorded `nnUNetPlans.json` path to `train_replacement.py --plans` and
retain its adjacent `.provenance.json` sidecar. The trainer requires this sidecar,
checks its dataset and fold against `--dataset` and `--fold`, and verifies both
the training array hash and plan hash. Use a separate planning workspace
and dataset ID for each fold. The current integration imports the planned
architecture and uses UniBrain inputs and objectives; it is not a reproduction
of nnU-Net's complete preprocessing, augmentation, loss or training pipeline.

Validate a genuine plan against one training sample with the reproducible CPU
check below. It records input and auxiliary output shapes, finite gradients,
parameter changes and the unchanged native plan hash. It does not establish
complete UniBrain integration or model performance.

```sh
python scripts/check_nnunet_plan.py \
  --plans /path/to/nnUNetPlans.json --data /path/to/exported/fold \
  --dataset abide --fold 1 --output /path/to/new/core_check.json
```
