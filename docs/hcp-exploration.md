# HCP learning check

Use the current quality-reviewed HCP delivery and preserve its original fold
indices. The present task is sex classification: female 0, male 1. It is not a
disease diagnosis benchmark. Family-disjoint membership has not been verified;
results from these provisional subject splits cannot establish family-independent
generalization.

Export with `scripts/export_cohort_fold.py COHORT OUTPUT --dataset hcp --fold 1
--full`. This preserves all selected rows and exports each subject's partial
tissue validity mask in matching order. Masks and their source checksums are bound
to the cohort and training checkpoint. Tissue Dice, Jaccard and tissue-based
registration overlap exclude unannotated voxels. The original template-derived
segmentation training loss remains unchanged. Brain and propagated AAL references
are automated reference labels, not manual annotations.

The initial exploration uses the complete paper architecture, seed 17, original
Adam configuration, 20 epochs, patience 0 and a fixed 64-subject training probe.
The original model and cross-subject anchored Agent start from scratch. A hash of
the initialized model state must match before progressing beyond two update checks.
Test inference remains disabled. All training subjects are retained; the two-step
check bounds optimizer updates rather than replacing the full cohort with a toy set.

`python -m unibrain_agent.paired_training MANIFEST` alternates one epoch per job
on an available GPU. A manifest declares `steps_per_epoch`, `epochs`, and two
supervisor-compatible `jobs`, each with `name`, `command`, `output`, `cwd`, and
`environment`. Paths and GPU assignment are deployment configuration. Both jobs
must expose `initial_state_sha256` in their architecture report. The queue first
checks two updates per job, then uses complete epoch budgets. Resume checkpoints,
optimizer state, policy state and recovery counters persist across turns.

A randomly initialized classifier may predict one class. The HCP Agent explicitly
uses `--coordination-bootstrap-epochs 3`. Before qualification, a finite training
probe loss below five can continue for at most three epochs despite single-class
predictions. It is not saved as a safe recovery checkpoint. Once qualified, the
usual collapse criteria apply immediately; the allowance cannot be introduced into
an existing checkpoint because it is part of the experiment identity. Nonfinite
loss, excessive training loss and gradient checks remain active from the first step.
