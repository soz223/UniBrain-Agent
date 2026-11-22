# Transfer and execution runbook

Prepared: **2026-11-22 13:20 EST / 18:20 UTC**.
Experiment status was verified at **13:14 EST / 18:14 UTC**.
Read [HANDOFF.md](../HANDOFF.md) before running commands. This guide describes
the full model research CLI, not the compact UI demonstration.

## 1. Obtain the private assets

GitHub supplies code, source pins, release checks and this plan. The owner must
separately transfer authorized datasets and experiment directories. Obtain the
current `bn_consistent_20261122` pair directories, their `paired_manifest.json`,
release manifest and the archived source checkpoints. HCP uses the
`delivery_family_v3_20261116` cohort. The operational server aliases are MAIR01
and NCSA Delta. Actual account paths and credentials are private operator input.

For existing runs, copy the complete run directories, including `last.pt`,
`best.pt` when present, stable checkpoints, `report.json`, `updates.jsonl`,
supervisor records and retained recovery records. The resume validator checks
the checkpoint against its committed trace and selected model. A weights file
alone is insufficient to continue the same experiment.

Coordinate a graceful stop or a consistent checkpoint snapshot before the final
copy. Retain the old location until transfer verification succeeds. Record
file hashes at both ends. Never run two writers against one output directory.
Do not transfer a live `supervisor.lock` as permission to start another worker;
verify that the old owner and worker have stopped before archiving a stale lock.

HCP references require `tissue_valid_train.npy`, `tissue_valid_val.npy` and
`tissue_valid_test.npy` in addition to the exported image arrays. Keep the
cohort provenance, split protocol, source indices and templates. Preserve the
five original folds and all original data. Do not substitute the earlier
subject split or a smoke subset. No dataset or checkpoint belongs in GitHub.

## 2. Install and verify the model release

These are Linux commands. Set your own workspace and data locations; no old
server path should be copied blindly. A fresh clone has no trained weights.

```bash
git -c core.autocrlf=false clone https://github.com/soz223/UniBrain-Agent.git
cd UniBrain-Agent
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-upstream.txt -r requirements.txt
export UB_PROJECT="$PWD"
export UB_PYTHON="$(command -v python)"
export UB_SOURCES="$UB_PROJECT/sources"
export UNIBRAIN_SOURCES="$UB_SOURCES"
python scripts/bootstrap.py --source reg_unibrain
python scripts/verify_code_release.py "$UB_PROJECT" "$UB_SOURCES/reg_unibrain" \
  deploy/paired_continuation/releases/bn_consistent_20261122.json
```

The tested environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.1,
PyG 2.4.0, NumPy 1.26.4, SciPy 1.15.3 and scikit-learn 1.3.2. Select a GPU with
sufficient free memory; current runs use 48 GB GPUs. Check occupancy immediately
before a local launch. On Slurm use only the GPUs assigned to the allocation.

The release verifier checks code bytes and dependencies. Clone with LF line
endings. Documentation may advance while this model release remains pinned.
If future `main` fails verification, do not overwrite the reference manifest
to make the error disappear. Use the model files from `69d98ae` and the launchers
from `92d3a27`, or explicitly validate and label a new release. A source checkout
without provenance cannot be treated as the same experiment.

Run the focused CPU tests in the installed environment:

```bash
python -m unittest tests.test_registration_normalization \
  tests.test_normalization_diagnostics tests.test_code_release \
  tests.test_replacement_evaluation tests.test_training_diagnosis
```

These tests do not replace a CUDA check on the destination server.

## 3. Verify the data contract

Each exported fold contains `adhd_train_sample.npy`, `adhd_val_sample.npy` and
`adhd_test_sample.npy`. These historical filenames also represent full HCP and
ABIDE folds; the name alone does not identify the cohort or sample size. The
structured fields are `id`, `label`, `raw_img_np`, `raw_img_brain_mask_np`,
`raw_img_gm_mask_np` and `raw_img_aal_mask_np`. Images and references are 96 cubed.
The three template arrays are `template_img_orig_96.npy`,
`template_img_gm_mask_orig_96.npy` and `template_img_aal_mask_orig_96.npy`.

If an exported fold is already supplied, verify and reuse it. To export a new
HCP fold from the authorized master and preserved family protocol:

```bash
# Set UB_COHORT to the complete family delivery and UB_FOLD to a new destination.
python scripts/export_cohort_fold.py "$UB_COHORT" "$UB_FOLD" \
  --dataset hcp --fold 1 --full --split-root "$UB_COHORT"
```

Use the actual dataset and fold for ABIDE or ADHD. Their existing subject
splits are preserved; do not label them as HCP family protocols. Confirm counts,
labels, input checksums, template normalization and reference validity before
training. The trainer's preflight checks the recorded cohort provenance.

## 4. Continue a fixed experiment

Copy both original and Agent run directories to a new private pair directory.
Use each arm's own `last.pt`; do not replace a progressed Agent checkpoint with
the original's weights. Preserve configuration, optimizer, policy, RNG, trace
and validation selection. Update paths in the copied manifest to the new Python
interpreter, repository, source, data, output and release manifest. Keep method,
split, seed and model options unchanged.

The supplied [HCP fold 1 manifest example](../deploy/paired_continuation/hcp_pair.example.json)
matches this dated continuation's configuration. It is not an arbitrary dataset
template. Prefer the actual saved manifest for other datasets or folds.
Its placeholders are not automatically expanded by the training launcher.
Export `UB_PROJECT`, `UB_PYTHON`, `UB_SOURCES`, `UB_FOLD`, `UB_PAIR` and `UB_GPU`
to verified destination values, then render a separate manifest:

```bash
python - "$UB_PAIR/paired_manifest.migrated.json" <<'PY'
import json
import os
import sys
from pathlib import Path
from string import Template

template = Path('deploy/paired_continuation/hcp_pair.example.json')
def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return Template(value).substitute(os.environ) if isinstance(value, str) else value
spec = expand(json.loads(template.read_text()))
with Path(sys.argv[1]).open('x') as stream:
    json.dump(spec, stream, indent=2)
PY
```

Inspect the rendered commands. For each job, execute its command with
`--preflight` appended before the CUDA run. This checks resources, cohort
provenance and files, not actual GPU execution. For a bounded CUDA acceptance
check, use a disposable copy of the complete run directory and set `--max-steps`
to the checkpoint's current step plus two. The limit is an absolute step count,
not a count of additional updates. Retain failed attempts for diagnosis.

After verifying finite outputs, gradients, actual parameter updates and a
successful save/resume, start the pair under a persistent process manager:

```bash
python -u deploy/paired_continuation/launch_pair.py \
  "$UB_PAIR/paired_manifest.migrated.json"
```

Use tmux, a service or a scheduler so the trainer does not depend on a local
terminal staying open. The launcher verifies the shared release before running.
Two arms alternate budgets on one GPU. Preserve the current additional epoch
budget and validation selection rule. A two step check establishes execution,
not convergence or Agent improvement.

## 5. Start a new normalization intervention

This differs from resuming an already fixed run. Start in two new directories
from the same **complete original resume checkpoint**, and keep its source
untouched. Record its SHA256. Use `--paper-original`,
`--registration-normalization per_image` and `--resume-normalization-upgrade`.
The migration accepts audited source versions only. Unsupported historical
checkpoints require investigation, not editing their identity metadata.

The original arm uses `--coordination none`. The current Agent arm uses
`--coordination taskforce --taskforce-profile paper`,
`--taskforce-gradient-normalization anchored --taskforce-action-bound 1`,
`--coordination-reward classification_advantage` and the recorded bootstrap
configuration. It keeps the same model components and original task losses.
Use the exact saved options rather than inventing a new loss configuration.

Migration retains the model and optimizer but resets Agent policy state,
validation history, best-model selection and phase counters. The source epoch
and step are retained in `normalization_migration`. The current 1000 epoch
budget is **additional** to the archived history. Do not report epoch 1 as
training from random initialization. An already fixed checkpoint with matching
identity resumes normally rather than resetting again.

## 6. Slurm operation

The current Delta plan requests one A40, four CPUs, 64 GB host memory and
24 hours per dataset pair. Use an allocation account and partition available
to the new operator. Do not copy another person's authentication or assume
the same partition names exist on a different cluster.

The batch script should execute the following using real deployment paths:

```bash
exec "$UB_PYTHON" -u "$UB_PROJECT/deploy/paired_continuation/run_allocation.py" \
  "$UB_PAIR/paired_manifest.migrated.json" "$UB_BATCH_SCRIPT"
```

Keep `run_allocation.py` and `launch_pair.py` together. The allocation runner
requests checkpointing after 23 hours 45 minutes and submits a dependent
continuation only for this scheduled boundary. Manual cancellation does not
resubmit. The cluster must allow `sbatch` from compute nodes. This rollover
path is implemented but its first real 24 hour cycle was not observed at the
handoff timestamp. Inspect the first rollover rather than assuming it worked.

Useful scheduler checks are `squeue -u "$USER"`, `scontrol show job JOB_ID`
and `sacct -j JOB_ID`. Confirm resource assignment before launch and retain
allocation IDs, logs and failed attempts. Avoid duplicate submissions during
the ownership transfer.

## 7. Inspect results and evaluate

`queue.json` identifies the active arm. Per arm, inspect `health.json`,
`supervisor.json`, `supervised_worker.log`, `updates.jsonl` and `report.json`.
The last file includes full validation results after each completed epoch.
Policy updates and optimizer steps are separate from model optimizer steps.
The W&B records in [HANDOFF.md](../HANDOFF.md) are existing monitoring targets;
a new operator needs their own permitted login and project configuration.
An SSH tunnel is not required for a trainer to continue on a server.

Use the same evaluator and recorded normalization policy for both arms.
For an explicit validation evaluation of a copied run:

```bash
UB_CHECKPOINT_SHA=$(sha256sum "$UB_RUN/last.pt" | cut -d ' ' -f 1)
UB_REPORT_SHA=$(sha256sum "$UB_RUN/report.json" | cut -d ' ' -f 1)
python scripts/evaluate_replacement.py \
  --checkpoint "$UB_RUN/last.pt" --checkpoint-sha256 "$UB_CHECKPOINT_SHA" \
  --training-report "$UB_RUN/report.json" --training-report-sha256 "$UB_REPORT_SHA" \
  --source "$UB_SOURCES/reg_unibrain" --data "$UB_FOLD" \
  --output "$UB_EVALUATION" --dataset hcp --fold 1 --split val --gpu 0
```

Use a stable snapshot with a matching completed or budget boundary report;
do not hash files while a worker is replacing them. Use a new evaluation
directory. The evaluator defaults to test if `--split` is omitted, so always
specify the intended split. Do not repeatedly use test results to tune a method.

Finish matched first fold validation before broader experiments. Later,
freeze the method and validation selection rule, complete all five preserved
folds, evaluate the selected checkpoints on their test splits, and report means
and standard deviations. Current first fold validation values cannot substitute
for those final results.
