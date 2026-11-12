# Saved pipeline evaluation

`scripts/evaluate_replacement.py` evaluates a checkpoint from
`scripts/train_replacement.py`. It reconstructs the recorded full host,
candidate, generated adapter and connection settings, then loads every saved
tensor strictly. It creates no optimizer and performs zero training updates.
This entry point is separate from original upstream checkpoint evaluation.

```sh
python scripts/evaluate_replacement.py --checkpoint private_runs/run/last.pt --checkpoint-sha256 EXPECTED_SHA256 --source sources/reg_unibrain --data data/abide_fold1 --dataset abide --fold 1 --output private_runs/evaluation --split test --gpu 0
```

Use `best.pt` to evaluate the validation selected model. `last.pt` also supports
short acceptance runs that never completed an epoch. Evaluating such a model
does not establish a final test result for the paper. The default evaluates
every subject in the requested split; `--limit 1` is an explicit diagnostic
limit. `--save-predictions` retains per subject predictions locally. IDs,
reference labels and predictions are private runtime artifacts.

The checkpoint must contain its model, identity and progress from replacement
training. Both weights only and resumable checkpoints are accepted. Original
initialization checkpoints and prior composition checkpoints are unnecessary:
the final checkpoint contains the complete trained pipeline. Sequential and
simultaneous compositions share the same topology. The evaluator constructs
that topology without replaying historical initialization, loads the final
state strictly, and records the historical composition mode separately.

The adjacent `report.json` supplies the recorded configuration. Supply
`--training-report` if it was moved, and optionally
`--training-report-sha256` to check an externally recorded report checksum.
The checkpoint's required SHA256 is verified before and after loading and
again after evaluation. Method, setting, seed, data hashes, fold, adapter,
plan and connection settings must agree with checkpoint metadata. Tensor
names, shapes, dtypes and total parameter count must match reconstruction.
Nonfinite checkpoint tensors fail inspection.

Migration uses explicit `--source` and `--data` paths and the existing
`UNIBRAIN_SOURCES` configuration for pinned candidate repositories. Override
relocated artifacts using `--adapter`, `--plans` or `--secondary-extraction`.
Their content hashes must still match. nnU-Net also requires the matching
plan provenance sidecar. Cohort or site manifests and their retained original
membership files are required; a public sample without that provenance cannot
be presented as a preserved fold evaluation.

Required recorded architecture files must match their training hashes. A
change limited to the trainer's warmup or resume bookkeeping does not change
the reconstructed architecture and is reported separately. The evaluation
report retains historical and current implementation hashes. Historical
checkpoints did not hash `data.py`, `upstream.py` or `native.py`; their current
hashes are recorded, with this coverage limit stated explicitly. This is
provenance validation, not a cryptographic signature of training history.

`--preflight` inspects metadata, checkpoint tensors, source and data hashes on
CPU. It does not construct the full model or call CUDA and does not create the
output directory. Successful preflight establishes static readiness. Runtime
evaluation checks GPU availability and records `report.json`, per subject
metrics and whether the entire split was evaluated. Use a new output directory.

CPU tests exercise real small checkpoint serialization, migrated paths,
tamper rejection, strict reconstruction, adapter and connection routing,
timeout accounting and an inference orchestration fixture. Those tests do not
establish GPU inference for a full candidate. Full checkpoint inference must
be checked in the target environment before reporting a migrated experiment.
