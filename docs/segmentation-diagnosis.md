# Segmentation diagnosis

The diagnostic worker inspects a stopped original or SynthStrip full-model run.
It restores the saved model, performs bounded forward and backward probes, and
never creates an optimizer or writes into the training run. Active runs are rejected.
Keep the recorded architecture and pinned sources available when relocating it.

```sh
python -m unibrain_agent.segmentation_audit \
  --run /experiments/completed_run \
  --output /diagnostics/segmentation/report.json \
  --device cuda --samples 3 --threads 4
```

Use `--source`, `--adapter`, and `--data` to relocate pinned code, the generated
adapter, and templates. `--master` plus `--indices` selects validation subjects
from a master array instead of the materialized fold. `--subjects` enables
NIfTI affine, orientation, array equivalence and source tissue remapping checks.
Verify copied fold membership and sampled arrays before interpreting a relocation.
Select an idle device or request an independent scheduler allocation first.

The report separates predictions against transferred template supervision from
predictions against the automated subject reference. It records foreground and
background CE, per-class Dice, class counts, grid coverage, affine determinant,
target probability mass, and segmentation-specific gradients. Training and
inference modes are distinct: the SynthStrip adapter thresholds its mask only at
inference. A detached-target probe isolates gradients through target generation;
it does not change the trained checkpoint or establish a validated repair.

```sh
python scripts/review_training_diagnosis.py \
  --report /diagnostics/segmentation/report.json \
  --output /diagnostics/segmentation/review \
  --backend configured --qualify
```

The configured provider receives aggregate diagnostics, not image arrays, subject
IDs or private file paths. `--backend rules` is explicit deterministic screening,
not an LLM qualification. Six synthetic cases check target disagreement, target
probability mass, metric disagreement, geometry, missing segmentation gradients,
and absent evidence. Passing them does not establish general autonomous repair.
Evidence citations are validated; no review action executes a training change.

Set `diagnosis_report` in backend runtime configuration to the audit JSON to make
it available to the existing Training workspace role review. It carries its own
dataset, fold and checkpoint identity. Segmentation and registration roles receive
the diagnostic prompt. A coordinator gate prevents weight balancing while the
attached diagnostic has unresolved screening flags. Detach this report when it
is no longer applicable; it must not be treated as evidence for another checkpoint.

## ABIDE diagnostic observation, 12 November 2026

The first, middle and last validation subjects were probed using the best epoch 9
checkpoint from the full SynthStrip replacement run. This is a three-subject
diagnosis, not a cohort-level re-evaluation or an original UniBrain baseline.

* Tissue label remapping, shared 96-cubed NIfTI affines and array equivalence passed
  on all three subjects. Label permutation and single-axis flips did not improve
  their overlap. This is consistency evidence, not expert anatomical validation.
* Independently recomputed foreground Dice matched the evaluator within 1e-8.
* Template-supervised CE was 0.0294 to 0.0596; subject-reference CE was 0.427 to
  0.563. Transferred template versus subject-reference Dice was 0.217 to 0.251.
* The transferred target was 96.1% to 96.3% background. Background CE was 0.0042
  to 0.0058, while target foreground CE was 0.639 to 1.512. Whole-volume averaging
  therefore obscured poor tissue performance.
* About 78.4% to 79.7% of inverse-grid coordinates were outside the template
  domain. This includes background and is a warning, not proof that the same
  fraction of brain tissue was lost. The transferred tissue volume was small.
  Restricting the check to subject tissue gave 1.7%, 20.0%, and 8.3% outside-grid
  sampling. Forward and inverse affine roundtrip errors were below 1.2e-7; this
  rules out a gross numerical inverse error, not anatomical registration error.
* In one training-mode backward probe, all 42 segmentation, 28 registration and
  54 extraction parameter tensors had finite nonzero segmentation gradients.
  Detaching the target removed its registration and extraction paths, retaining
  segmentation gradients. This establishes a moving-target gradient path, not
  a causal proof of harmful optimization across training.

The next controlled experiment should compare supervision and registration
constraints on an isolated copy. Do not silently modify the ongoing ADHD run,
rename the modified method as original UniBrain, or tune against held-out test
performance. A longer epoch budget alone does not address these observations.
