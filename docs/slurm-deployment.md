# Slurm deployment

Keep code, the Python environment and pinned sources in project storage. Keep
the immutable master arrays and original fold indices in the site's data area.
Export only the folds needed for the experiment into working storage. Do not
use the home directory for training I/O or copy local provider credentials.

The launcher expects this layout under `UNIBRAIN_ROOT`:

```
code/                         repository checkout
env/bin/python                isolated environment
sources/reg_unibrain/         pinned original source
sources/freesurfer_synthstrip/ pinned SynthStrip source
adapters/synthstrip.py         accepted generated adapter
```

Install `requirements-upstream.txt`. Verify source hashes with the existing
source verifier. Copy the accepted adapter without changing its hash. Preserve
the source commit, environment package list and transfer checksums in the
private deployment record.

`UNIBRAIN_DATA_ROOT` contains `abide_round1` and `adhd_round1`, exported with
`scripts/export_cohort_fold.py --full`. The `abide_smoke` and `adhd_smoke`
directories contain strict acceptance subsets from those same original folds.
The master arrays, original indices and templates remain separate and intact.

Set `UNIBRAIN_ROOT`, `UNIBRAIN_DATA_ROOT` and `UNIBRAIN_RUN_ROOT` before submitting
`scripts/slurm_train.sh`. Supply the site's account, partition, CPU, memory, GPU,
wall time and log path using `sbatch` flags. The launcher relies on Slurm's GPU
visibility and the trainer's occupancy checks; do not override GPU visibility.

```
sbatch --account=ACCOUNT --partition=PARTITION --nodes=1 --ntasks=1 \
  --gpus-per-node=1 --cpus-per-task=4 --mem=64G --time=00:20:00 \
  --output=SMOKE_LOG scripts/slurm_train.sh abide smoke
```

After successful full-model acceptance on the new machine, submit with `full`
and an appropriate time limit. This configuration uses the accepted SynthStrip
extraction replacement, connected gradient paths, joint training from random
initialization, seed 17 and original round 1 membership. It does not invoke a
new literature agent or enable TaskForce coordination. The full run uses every
training subject, full validation, a 100-epoch cap and patience 10 based on
validation diagnosis cross entropy. Test is evaluated only after selection.
An early stop is a selection rule, not proof of convergence or improvement.

The output directory is `UNIBRAIN_RUN_ROOT/DATASET_MODE`. It holds:

* `updates.jsonl`: per-update losses and gradient/update evidence.
* `report.json`: configuration, provenance, validation history and status.
* `last.pt`: resumable optimizer, model, RNG and progress state.
* `best.pt`: validation-selected checkpoint.
* `test.json` and `predictions/`: final held-out results.

Use `python scripts/training_status.py UNIBRAIN_RUN_ROOT` to read current
artifacts. The report is refreshed at epoch boundaries; per-update files are
written after each update. Use `squeue` and `sacct` to distinguish an active job
from stale artifacts after an interruption. Resubmitting the same launcher
resumes from `last.pt` with the same configuration. Completed runs should not
be resubmitted. Failed attempts without a resume checkpoint need a new output
root; retain the old diagnostics.

Batch jobs continue if the local SSH connection closes. A new authenticated
connection is required to submit more work or retrieve logs. Full cohort
training remains separate from acceptance checks and the original baseline.
