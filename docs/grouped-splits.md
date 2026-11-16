# HCP split protocols

Reported exploratory HCP results used one round of an existing subject-level
five-fold partition. They are not a five-fold average or a family-independent
estimate. Preserve those runs and their original indices.

`scripts/build_grouped_folds.py COHORT NEW_OUTPUT --family-csv METADATA` creates
two separate five-fold protocols. The subject protocol groups all scans of one
person and identical input arrays. The family protocol additionally joins
Family_ID, Mother_ID and Father_ID through connected components, including
half siblings and transitive links. Missing family information blocks the family
protocol rather than treating unknown people as unrelated. Family identifiers
are private metadata; never publish them or the subject-level assignment files.

Each round uses one bucket for test, the next for validation, and the remaining
three for training. Each subject is tested exactly once across five rounds.
Sex stratification is approximate because families cannot be split. The seed
is fixed before training. Do not optimize the seed using model performance.

Export a new dataset with:

```sh
python scripts/export_cohort_fold.py COHORT NEW_DATA --dataset hcp --fold 1 --full --split-root NEW_OUTPUT/family_grouped
```

The exporter retains the protocol, verifies image/master and index checksums,
and binds it to the exported data. The training validator checks that group
memberships cannot cross train, validation and test. Image hashes detect exact
copies, not every possible resampled or near-duplicate scan. Source subject IDs
remain the primary protection against derived-image leakage.

All visits, repeat acquisitions, modalities, slices, crops and augmentations
inherit their subject's group. Relative session day and time of day are not
calendar dates; they cannot establish prospective temporal generalization.
A separate chronological holdout needs trustworthy dates and must retain
family and subject isolation. A single-site HCP analysis also cannot establish
generalization to new sites or scanners.

Fit population normalization, harmonization, feature selection, calibration and
data-derived templates on training subjects only. Per-image normalization is
allowed. Document independent atlas/template and pretrained-weight provenance.
Audit any overlap with pretraining cohorts separately; split checks do not
establish that upstream training data are independent.

Retrain both methods from fresh initialization on the new folds. Existing HCP
checkpoints have seen subjects assigned differently under the new protocol.
Use validation for selection and reserve test for the frozen comparison.
Prior exposure to exploratory results must be disclosed. Changing the split
does not create a previously unseen external cohort.
