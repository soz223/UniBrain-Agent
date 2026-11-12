# Bounded acceptance and trajectory checks

These scripts verify actual model execution on small, auditable samples from
the existing cohort folds. They retain the full architecture and 96 cubed
images. Their results are separate from the complete cohort baseline and from
scientific comparisons of accuracy or convergence.

## Export a version 2 subset

The exporter is standalone and needs only NumPy and nibabel. Run it where the
complete master and original indices are available:

```sh
python scripts/export_acceptance_subset.py ABIDE_COHORT NEW_ABIDE_SUBSET --dataset abide --fold 1 --per-class 1 --seed 17
python scripts/export_acceptance_subset.py ADHD_COHORT NEW_ADHD_SUBSET --dataset adhd --fold 1 --per-class 1 --seed 17
```

Each split contains the requested number of subjects per diagnosis class,
selected only from that split's original indices. The exporter retains copies
of all original train, validation and test index arrays. `cohort.json` records
the full master checksum, source index checksums, selected master rows, subject
IDs, class counts, template provenance and exported file checksums. Template
affines must be valid and agree. Every image array remains 96 cubed. A new
output directory outside the original cohort is required. Failed partial
exports remain marked `failed` and cannot be used for acceptance.

Copy the complete subset directory, including `source_indices`, `cohort.json`
and the templates, when migrating. Keep these artifacts private. Version 1
subsets are retained as historical evidence; export a new version 2 directory
for this runner. The full master checksum requires reading the source file
once and can take time on large storage systems.

## Configure and inspect

```json
{
  "source": "../sources/reg_unibrain",
  "datasets": {"abide": "../acceptance/abide", "adhd": "../acceptance/adhd"},
  "methods": ["voxelmorph"],
  "fold": 1,
  "seed": 17,
  "gpu": 0,
  "coordination": "none"
}
```

Paths resolve relative to this configuration file. Add `adapter` for SynthStrip
and per-dataset `plans` for nnU Net. These plans must be generated from the
acceptance training split with their matching provenance sidecar. Configuring
an unsupported or missing resource does not cause a fallback to another model.

By default, both datasets run every configured method. `job_subset` can select
an explicit nonempty list such as `["abide:voxelmorph", "adhd:lapirn"]`. Every
entry must name a configured method and dataset, and duplicates are rejected.
The result records the selected jobs and their denominator. A selected subset
does not imply that the omitted combinations passed.

```sh
python scripts/run_acceptance_matrix.py --config PRIVATE_CONFIG.json --output NEW_MATRIX
python scripts/run_acceptance_matrix.py --config PRIVATE_CONFIG.json --output NEW_MATRIX --execute
```

The first command checks data provenance and prints the intended commands.
Only the second executes training. Acceptance requests exactly two updates per
dataset and selected method. The runner checks the child preflight status,
completion status, exact update count, method, fold, seed, data hashes,
per-update parameter changes and saved checkpoint. `budget_exhausted` is an
expected successful bounded status when all this evidence agrees. A zero exit
code alone cannot pass acceptance.

Every new child receives `--verify-gradients` and must report gradient protocol
2. Total input and candidate parameter gradients must be finite and nonzero.
Classification gradients must reach each required replacement and the input
image. This boundary detects a module that learns its own parameters while
detaching its inputs. The auxiliary nnU Net tissue head declares its
absent direct classification path explicitly; its total training gradient
must still be nonzero. The original model is retained as its own diagnostic
reference. This distinction is recorded in the validation evidence.

Older evidence can be revalidated under its original protocol. It is marked
`legacy_boundary` with `strict_current_gradient_verified=false`; it cannot
satisfy a new protocol 2 job. Original and auxiliary nnU Net classification
input paths remain explicitly diagnostic rather than required paths.

Acceptance defaults to `checkpoint_mode: weights`, retaining the complete
model, identity and progress while omitting optimizer state. These checkpoints
support evaluation or provenance checked initialization, not training resume.
Choose `resume` explicitly when that state is needed. Trajectory mode requires
`resume`. The runner checks the reported checkpoint policy against the request.

Existing output directories are preserved and rejected. A timeout, occupied
GPU, CUDA resource failure or interruption halts later jobs. The runner never
terminates another experiment or silently retries on another GPU. Model-specific
validation failures remain recorded; the selected resource is checked again
by the next training child. Logs and failed outputs are retained.

Before the matrix and each child, free space must exceed a 10 GiB reserve plus
the configured checkpoint budget. The default budget is 8 GiB per current job,
including the temporary peak of atomic checkpoint writes. A shortage records
`blocked_disk_space` and stops subsequent launches. The manifest also reports
the sum of planned budgets. This is an estimate, not a disk reservation; other
jobs can consume space concurrently. Checkpoints are retained automatically.

## Optional longer probes and coordination

`--phase trajectory` uses explicit `trajectory_methods` and `trajectory_updates`
from the configuration, with a maximum of 100 updates. Its default is eight
updates and it runs only when explicitly executed. A small validation trace is
not evidence of cohort convergence or improved generalization.

`coordination: taskforce` enables the trainer's coordinator. The matrix records
weight decisions separately from actor, critic and policy optimizer counts.
Two steps can demonstrate decisions while all policy update counts remain
zero. A trajectory of at least five steps must show actual actor and critic
updates to pass that policy-learning check. Counts are checked against the
per-step trace. Learning an effective policy still requires a separate
scientific evaluation.

```sh
python -m unittest discover -s tests -p test_acceptance_matrix.py -v
```

The tests use CPU protocol fixtures and temporary data. They do not run a GPU
model or convert fixture outcomes into experimental results.

`unibrain_agent.cohort_provenance.validate_cohort` provides the shared export
validator. Passing the caller's six already-computed array hashes verifies
their agreement without reading those large files again. Omitting hashes
checks metadata and marks array hashes unchecked. Public sample folders with
no manifest can be retained as explicitly unverified inputs by a caller that
allows them. The acceptance matrix and formal suite both require their exact
declared export scopes.

## Public evidence export

After execution, revalidate passed jobs and export a table suitable for code
review or sharing:

```sh
python scripts/summarize_acceptance.py --matrix PRIVATE_MATRIX/matrix.json --output NEW_EVIDENCE
```

The exporter checks report, update trace and checkpoint hashes again. It
retains aggregate counts, measured gradient statistics, method definitions
and statuses in JSON and CSV. Subject records, local paths, prompts and
credentials are excluded. Failed, blocked and pending jobs remain visible
in the denominator and do not become successful measurements.
