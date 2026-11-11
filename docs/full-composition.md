# Full two-stage composition

The composition executor combines a generated full SynthStrip extraction
adapter with full VoxelMorph or full three-level LapIRN registration. It keeps
the five original affine stages and the original segmentation, parcellation,
region feature and classification module widths. It uses the matched training
executor and can apply the same connectivity repairs and diagnostic tests.

This is a controlled composition of accepted modules. It does not by itself
establish that an Agent independently discovered and integrated both methods.
Agent composition success needs a separate integration trial and evaluator.

## Simultaneous replacement

```sh
python scripts/run_composition.py --source PINNED_SOURCE --data FOLD_DATA \
  --output NEW_RUN --method voxelmorph --setting joint \
  --secondary-extraction ACCEPTED_ADAPTER.py --composition-mode simultaneous \
  --connect-paths --max-steps 2 --eval-limit 2
```

Use the same command on PowerShell as one line. Paths resolve through ordinary
command arguments. `--method lapirn` changes the dense core. Explicit candidate
and host checkpoints use the trainer's provenance and SHA256 flags. Defaults
use random initialization and must not be described as pretrained methods.
Simultaneous replacement adds both modules before the requested joint training.

## Sequential replacement

First train the SynthStrip-only full pipeline and retain its checkpoint and
report. Then run the composition command with:

```sh
--composition-mode sequential --previous-checkpoint FIRST_STAGE.pt \
--previous-report FIRST_STAGE_REPORT.json --previous-sha256 ACTUAL_SHA256
```

The first-stage evidence must report at least one optimizer update. Its dataset
and fold must match any declared dataset and fold of the new run. A previous
SynthStrip smoke checkpoint can also be used for structural verification if
its original validation report is supplied; it is not a completed cohort run.

The loader strictly reconstructs and loads the previous extraction pipeline
before introducing dense registration. The generated extraction adapter hash
must match the prior run, preventing a semantic code change under unchanged
parameter shapes. Every old registration tensor is explicitly renamed from
`reg_net.*` to `reg_net.affine.*`. Only `reg_net.core.*` may be newly initialized.
Other missing keys, unexpected keys, shape changes, dtype changes and nonfinite
tensors fail validation. The trained extraction and all remaining host weights
are preserved. No broad `strict=False` load is used.

Sequential mode currently supports only extraction followed by registration.
It rejects a separately supplied host or extraction checkpoint that would
overwrite the prior trained state. The report records the previous checkpoint
and report hashes, prior update count, renamed keys, added keys and module
parameter counts. Previous-stage training costs must be included when comparing
simultaneous and sequential experimental budgets.

## Resource checks and validation

`--preflight` checks files and the experiment protocol without allocating a GPU.
`--build-only` constructs the full model on CPU, verifies the state migration
and writes `build_report.json`; it performs no optimizer update or GPU forward.
This checks structure and checkpoint transfer, not trainability or accuracy.

Actual training uses `train_replacement.py` through the shared trainer, which
checks the selected GPU and retains checkpoints, measurements and its update
budget. `--max-steps 2` bounds the new run; it does not erase the cost of the
previous stage. Without that flag the declared epoch budget is executed.

The suite maps both composition variants to this executable. Configure
`adapters.synthstrip`, matched registration weights and prepared fold paths.
Sequential jobs additionally need:

```json
{
  "sequential_runs": {
    "abide:1:17": {
      "checkpoint": "first_stage/last.pt",
      "report": "first_stage/report.json",
      "sha256": "REPLACE_WITH_ACTUAL_SHA256"
    }
  }
}
```

## Verified scope

CPU verification reconstructed the actual full SynthStrip checkpoint from the
two-update experiment, added the full VoxelMorph core, and retained 256 state
tensors with only 26 new registration-core tensors. The composed model contains
308,118,331 parameters. No GPU training of this two-module composition was
performed during that check. Separate tests reject absent retained keys,
unexpected keys, nonfinite state, wrong checkpoint hashes and mismatched folds.
