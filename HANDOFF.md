# Research handoff

Document updated: **2026-11-22 13:20 EST (America/New_York), 18:20 UTC**.
Experiment status verified: **2026-11-22 13:14 EST, 18:14 UTC**.
This is a dated snapshot, not a live status page.

## Objective

UniBrain Agent adds module selection, code integration, gradient verification and
training coordination to the full UniBrain workflow. The immediate priority is
a stable, controlled comparison of the original loss combination and an Agent
coordinator. Further module replacement experiments are deferred until this
comparison is understood. Better Agent performance is an objective, not a result.

The training coordinator contains local trainable actors and critics, replay,
target networks and measured task gradients. Its audited current variant uses
original loss weights with guarded adjustments and a classification advantage
reward on another training subject. It is adapted from TaskForce, not an exact
TaskForce reproduction. LLM literature research and advisory prompts are separate
from these numerical policy updates. Normal coordinator training requires no
Codex or other LLM account.

## Active release

Model and evaluator commit: `69d98ae`.
Portable continuation launchers: `92d3a27`.
Pinned original UniBrain source: `f9f8d906b1725046d532f862a28876482aca2cfa`.

Both servers passed the same 144 file release check and matched the recorded
NumPy, SciPy, scikit-learn, PyTorch and PyG versions. The shared
[release manifest](deploy/paired_continuation/releases/bn_consistent_20261122.json)
is portable and contains no server paths. Its SHA256 is
`149e7b0173d9aa45ba48cf5d02915dc77b3beadb9aadf27dca923b416ff42a11`.
Later documentation commits do not by themselves imply a new model release.

## Current experiments

1. **HCP, MAIR01, fold 1.** The original arm completed continuation epoch 1:
   646 updates, full validation on 223 subjects, AUC **0.979143**, accuracy
   **0.874439**. The Agent arm was at update **300**, with **236** coordinator
   updates and **944** actor and **944** critic optimizer steps. It had no
   completed validation epoch at the snapshot time. Both arms share one GPU
   and alternate epoch budgets. `budget_exhausted` at a handoff means the arm
   reached its allotted steps, not that the overall experiment ended.
2. **ABIDE, Delta, fold 1.** Job **24376778**, pending for priority. One A40
   allocation alternates the original and Agent arms. Static checks passed for
   both branches: 651 train, 217 validation, 217 test subjects. GPU execution
   of this new release on Delta was not yet observed.
3. **ADHD, Delta, fold 1.** Job **24376779**, pending for priority. The same
   paired plan uses 465 train, 155 validation and 155 test subjects. Static
   checks passed; new Delta GPU execution remains to be verified.

Monitor the [unibrain W&B project](https://forge.coreweave.com/wandb/soz223-lehigh-university/unibrain).
New names contain `bn-consistent`. The live
[HCP original](https://forge.coreweave.com/wandb/soz223-lehigh-university/unibrain/runs/0d5b939d8473fd39)
and [HCP Agent](https://forge.coreweave.com/wandb/soz223-lehigh-university/unibrain/runs/9ce8bf6a6f972c05)
have separate records. Access may require the owner's project permissions.
Do not infer that queued jobs have produced results.

## Why this continuation exists

The five registration stages share an encoder. Changing its inference BatchNorm
statistics, with fixed epoch 190 weights and no training, changed Agent HCP
validation AUC from **0.415436** to **0.968593**. Applying the same diagnostic to
the original model changed AUC from **0.964778** to **0.978900**. These are
diagnostic validation results, not the new continuation's outcomes.

The production fix uses each current image's spatial statistics in both training
and evaluation. It preserves trainable affine parameters and checkpoint tensor
keys. The supported forward contains one subject. See
[normalization behavior and migration](docs/registration-normalization.md).
Thirty focused tests, full model input and parameter gradient checks, measured
parameter updates and save/resume checks passed. The production calculation
matched the diagnostic calculation. This does not establish anatomical accuracy
or long term training convergence.

Each pair starts from a shared original checkpoint. HCP uses the retained epoch
190 snapshot, SHA256
`1bb6418d3695be5878fb843569fd337789314c674eff76e1d86c6c09aa0cb197`.
Model weights and optimizer state survive; the Agent policy is initialized again.
New counters and validation selection begin at zero. The budget is up to
**1000 additional epochs**, with early stopping disabled. This is a matched
continuation with a normalization fix, not an untouched original paper baseline
or a pure normalization ablation of the previous Agent policy.

Old inconsistent UniBrain jobs were stopped. Their logs and checkpoints remain
in their original directories. Other external baseline experiments were not
cancelled. Legacy HCP used a different reward strategy and must not be merged
with this release under one method label.

## Data and evidence boundaries

HCP contains 1113 subjects and 445 family components. Five preserved family
folds were independently checked against restricted relationship metadata.
Cross split subject, Family_ID, shared parent, connected family and exact image
duplicate overlaps were zero. Actual fold 1 arrays matched the master, and
MAIR01 and Delta data file hashes matched. HCP labels are 0 female, 1 male.
This is not a claim of temporal or external site generalization.

ABIDE contains 1085 subjects and ADHD contains 775. Keep their existing five
fold indices. All image and reference arrays are 96 cubed. Segmentation and AAL
references are automatically generated; HCP tissue references have partial
validity masks. Do not describe these as manual ground truth.

The frozen Codex image classifier achieved HCP fold 1 validation AUC
**0.997160**, accuracy **0.964126**. A second server reproduced its probabilities
within `1.7e-15`. It uses image features and logistic regression, with fitting
restricted to training subjects. Twenty training label permutations gave mean
AUC **0.515574**. No direct label or family leakage was found in that audit.
This is a single validation fold, not five fold test performance; acquisition
confounding and anatomical shortcuts remain possible. Do not use it as evidence
that UniBrain or its Agent is superior.

## Next actions

1. Verify live scheduler and worker status before launching anything elsewhere.
   Avoid duplicate owners or two writers to one experiment directory.
2. Finish the first Agent validation cycle and compare both arms at matched
   continuation epochs. Inspect AUC, accuracy, calibration, anatomical metrics,
   gradient health, policy updates and compute cost together.
3. Confirm the Delta jobs execute the verified release and survive a checkpoint
   resume. The scheduled allocation rollover mechanism is implemented but its
   first 24 hour rollover has not yet been observed.
4. If the Agent remains worse, retain the result and diagnose the coordinator.
   Do not silently change learning rates, disable gradient checks or select on
   test labels. A revised method gets a new configuration and experiment name.
5. After the first fold comparison is stable, schedule the remaining preserved
   folds. Current active experiments cover fold 1 only. Freeze the method and
   validation selection rule before final test evaluation and report fold means
   with standard deviations.

## Reading order

Read the [handoff runbook](docs/handoff-runbook.md), then the
[normalization contract](docs/registration-normalization.md) and
[TaskForce mechanism audit](docs/taskforce-core-audit.md).
The broader [experiment suite](docs/experiment-suite.md) is a program inventory,
not a record that every experiment has been completed.
Data, checkpoints, credentials and machine specific paths are transferred
privately by the owner; they are intentionally absent from GitHub.
An aggregate [status snapshot](docs/status/2026-11-22.json) records the measured
values and their scope for later comparison.
