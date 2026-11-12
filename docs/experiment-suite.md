# Paper experiment suite

The suite keeps the intended paper matrix separate from runtime readiness and
completed evidence. A configured command is not a validated model. Missing
weights, datasets, integrations or implementations remain visible as blocked
jobs. Planning never launches training, changes a backend service or attaches
another dataset to the running dashboard.

## Coverage

| Family | Planned scope | Execution |
| :--- | :--- | :--- |
| Checker fixtures | Known tensor, gradient and spatial faults | CPU benchmark command; not a real candidate integration benchmark |
| Integration | 14 candidates, coding agents, checks as tools, full workflow, all six functional ablations and alternate LLMs | One configured case and provider per job through `benchmark_integration.py` |
| Main training | ABIDE and ADHD, five preserved folds, explicit seeds, original and three settings for four replacements | `train_replacement.py`; prerequisites and supported interventions are checked |
| Systems | Human or OpenClaw execution, NeuroPilot, NEXUS, NEURA, MLE STAR, coding agents and ours | Explicit external executor registrations |
| Pipeline ablations | Gradient cuts, repair strategies, trained components, initialization, loss scaling and composition | Full training and composition commands; resource checks apply |
| Site generalization | Named site splits from ABIDE | Native original and SynthStrip independent or joint training through `run_site_experiment.py` |
| Longer training | One case and fold under original, independent and joint settings | Explicit `long_epochs` budget |
| Composition | Simultaneous or sequential SynthStrip and VoxelMorph replacement | `run_composition.py`; sequential runs require a trained first-stage checkpoint and strict state transfer |
| Case study | Observed failure and repair trace | `summarize_case_study.py` audits a retained full replacement run |

Only diagnosis is a downstream task. The plan contains neither DeepAtlas nor
external manually annotated datasets. The original full stage quality metrics
remain part of training evaluation. Default planning retains three seeds and
three integration repetitions; these defaults are proposed protocol values,
not completed experiments or deterministic LLM seeds.

## Setup and planning

```sh
python scripts/experiment_suite.py init runs/paper/config.json
python scripts/experiment_suite.py plan --config runs/paper/config.json --output runs/paper/plan.json
```

Edit the generated configuration with local source paths, prepared fold
directories and accepted adapters. Paths resolve relative to the configuration
file on Windows and Linux. The Python interpreter defaults to the active one.
The output root must be private and excluded from version control. Never place
API keys in the configuration; providers use the established local account
configuration or environment.

`datasets.abide.folds.1` refers to an exported full fold containing the original
train, validation and test NPY files and templates. Export with
the complete master and existing split indices:

```sh
python scripts/export_cohort_fold.py COHORT_DIRECTORY NEW_FOLD_DIRECTORY --dataset abide --fold 1 --full
```

The exporter uses a memory-mapped master and writes one subject at a time.
It preserves every original index in its original order, keeps copies of the
original indices, and records full source and output checksums. It requires
enough disk space for the exported cohort plus a 10 GiB reserve. Exported fold
arrays occupy additional disk space; several folds can require substantial
storage even though memory use is bounded. A new destination outside the
source cohort is required. Both exporter scripts must be copied together when
running the full exporter outside this repository.

Set `datasets.abide.folds.1` to `NEW_FOLD_DIRECTORY`, and repeat only for the
required dataset and fold. The suite requires a complete version 2 `full_fold`
manifest and validates its membership, quantities and original index hashes.
At result ingestion, the actual training data hashes must agree with the
export manifest. No complete cohort was exported as part of the program tests.
Do not replace full fold paths with smoke subsets. `datasets.abide.sites` maps
explicit held out site names to separately prepared directories. Create each
directory with the actual site export program:

```sh
python scripts/prepare_site_split.py --master MASTER.npy --metadata SUBJECT_SITES.csv --templates TEMPLATE_DIR --dataset abide --held-out-site SITE_NAME --template-origin "Published atlas release" --template-scope external_public --output NEW_SITE_DIRECTORY
```

The metadata CSV needs exact `id` and `site` columns. Every subject from the
declared site enters the test split. Subjects from the other sites are split
by diagnosis into training and validation. The original five folds remain
unchanged. The exported `site_split.json` records source hashes, indices, IDs,
class counts and site counts. The source CSV, normalized membership table and
split indices are retained for portable auditing; keep them private. Supplied templates and any fitted
normalization must originate from external references or source training
subjects. Declare `--template-origin` and `--template-scope external_public` or
`source_training_only`. The latter must refer to the exported training subjects,
excluding validation subjects. A declaration and file hashes do not establish
this absence of information leakage.
Review template provenance and freeze model selection before evaluating the
held out site. The native runner verifies source metadata, subject membership,
site isolation, complete test membership and all six input file hashes. Site
experiments use `dataset=abide`, a named `split_id` and `fold=null`. The numeric
fold exporter and its provenance checks are separate from this site protocol.

```sh
python scripts/run_site_experiment.py --source PINNED_SOURCE --data NEW_SITE_DIRECTORY --output NEW_RUN --dataset abide --split-id SITE_NAME --method original --setting original --epochs 20 --preflight
python scripts/run_site_experiment.py --source PINNED_SOURCE --data NEW_SITE_DIRECTORY --output NEW_RUN --dataset abide --split-id SITE_NAME --method synthstrip --setting joint --adapter ACCEPTED_ADAPTER.py --connect-paths --epochs 20 --evaluate-test
```

`independent_finetuned` trains SynthStrip on source training subjects for the
explicit `pretrain_epochs` budget, freezes it, and trains the remaining host.
`joint` trains the candidate and host together. The suite uses matching random
candidate initialization across these two settings by default. Set
`site_initialization: official` to use the configured official SynthStrip
checkpoint and its recorded provenance in both settings. Numeric fold
checkpoints cannot initialize a site experiment. Additional standalone updates
remain visible in the independent setting's budget. Explicit external
executors remain available as overrides. Site metadata and training resources
must be supplied before execution; CPU fixture tests do not establish site
generalization performance.

All four main replacement comparisons require matched initialization. A checkpoint
must specify its source, hash and whether it is official or trained only on the
current training fold. Example:

```json
{
  "weights": {
    "nnunet": {
      "folds": {
        "abide:1": {
          "path": "weights/abide_round1_nnunet.pt",
          "provenance": "training_fold",
          "origin": "Standalone training on ABIDE round 1 training subjects",
          "sha256": "REPLACE_WITH_ACTUAL_SHA256",
          "dataset": "abide",
          "fold": 1
        }
      }
    }
  }
}
```

For official weights, use `provenance: official` and identify the actual release
in `origin`. A schema example is not an assertion that task specific official
weights exist. In particular, nnU Net needs a declared training protocol and
fold specific weights. A nonofficial reference checkpoint executes as
`independent_pretrained`, not `independent_official`, and its actual setting is
preserved in the training report. The suite's legacy `independent_official` job
slot represents that reference comparison; inspect provenance when writing the
paper table.

Registration comparisons also require a shared trained host checkpoint under
`hosts["abide:1"]`, with `path`, `origin` and `sha256`. All matched variants receive
the same host so that a frozen random affine stage is not mistaken for a
pretrained registration pipeline. The trainer checks source hashes and loading.

The suite's `original` job uses the matched full-model trainer with the original
architecture and objective. It is not the unchanged upstream entry-point run.
Keep the existing original baseline reports separately and identify both
protocols in the paper. The composition executor is described in
[Full two-stage composition](full-composition.md). CPU construction has been
checked; its joint GPU training still requires a bounded runtime validation.

The legacy `stagewise` job slot executes `original_frozen_stages`: fixed
pretrained UniBrain stages followed by a trainable classifier. It requires a
checkpoint from the same dataset and training fold. Label this exact control
in the paper; it is not an ANTs or FSL pipeline, nor an independent retraining
of every stage.

## Integration configurations

`integration_configs["synthstrip:full"]` points to an integration benchmark JSON
with exactly one matching case ID and one provider. The suite pins its variant,
seed and repetition to the planned job. Candidate source paths are resolved
relative to that benchmark configuration. See the integration benchmark guide
for evaluator contracts. An unavailable external agent needs an actual command
executor rather than relabeling Codex as that system.

Generate concrete configurations for all 14 candidates, then link them to
explicit suite variants:

```sh
python scripts/generate_integration_config.py --source PINNED_SOURCE --data FOLD_DATA --output NEW_CONFIG_DIRECTORY --provider codex
python scripts/experiment_suite.py import-integration-index --config PRIVATE_SUITE.json --index NEW_CONFIG_DIRECTORY/index.json --variants full codex codex_checks no_gradient
python scripts/experiment_suite.py import-integration-index --config PRIVATE_SUITE.json --index NEW_CONFIG_DIRECTORY/index.json --variants full codex codex_checks no_gradient --apply
```

The first import previews mappings. The explicit `--apply` updates that private
suite configuration, preserving relative paths where possible. An existing
mapping to a different file is rejected. Named Codex and Claude baselines must
use their matching providers. MLE STAR needs its actual executor and cannot be
created by relabeling a Codex configuration. Generate separate indexes for
other providers or model identities before selecting alternate LLM variants.
Configuration import performs no provider calls or model training. Each job
still passes its own capability and dependency checks; nnU Net requires its
dataset-specific plans and provenance. Changing a suite configuration changes
its digest, so use a new output root for a revised experimental protocol.

## Explicit execution

```sh
python scripts/experiment_suite.py run --config runs/paper/config.json --job JOB_ID
python scripts/experiment_suite.py run --config runs/paper/config.json --job JOB_ID --smoke-steps 2 --execute
```

The first command reports prerequisites only. The second explicitly executes
one selected job. Omitting `--smoke-steps` requests the configured full training
budget. No all jobs command is implicit. Repeated execution of an existing job
directory fails rather than overwriting evidence. A failed attempt can be
replanned under a new output root; retain the previous logs.

Compute jobs must use an allocation owned by the experiment. Training workers
check GPU availability and respect the visible allocation. A preflight does
not reserve a device or authorize terminating another process. The suite does
not modify or restart the current baseline service.

## External command protocol

An executor key can be a job ID, `group:method:variant` or `group:variant`.
Its record needs `description`, an argv list and optional `requires` file paths.
Commands run without a shell. Placeholders are `{python}`, `{repo}`, `{job}`,
`{result}`, `{output}`, `{source}`, `{data}`, `{seed}` and `{gpu}`. Bounded smoke
runs require a separate `smoke_argv` with `{max_steps}` and a result containing
`optimizer_updates` within that limit.

The command must write JSON to `{result}` with `status: complete`, the exact
`job_hash` from `{job}`, and its measured metrics. Attach predictions, source
revisions, input manifests, budget and cost evidence under `{output}`. Adapters
for outside systems must implement their actual task and data interfaces; this
protocol does not provide those implementations or verify clinical claims.

## Accounting and statistics

```sh
python scripts/experiment_suite.py summarize --config runs/paper/config.json --output runs/paper/summary_001
```

Each execution stores its immutable job specification, configuration digest,
argv, logs, elapsed time and original report hash. The summary lists every
planned run, including not run, blocked, failed, changed and smoke statuses.
Changed configurations and modified source reports cannot silently contribute
to metrics. Smoke results are never used as scientific results.

Training metrics are first averaged across seeds within each fold. The reported
descriptive mean and sample standard deviation then use fold means, not an
independence assumption over fold by seed combinations. Site experiments first
average seeds within each named site and then report the unweighted mean and
sample standard deviation across sites, with `unit=site_mean`. No significance test
is performed automatically. Integration success proportions use actual
attempts, retaining failed attempts and listing blocked attempts separately.
The plan and CSV make incomplete denominators visible; final paper comparisons
still require matched completed jobs and a prespecified statistical analysis.

For a separate paired subject analysis, provide complete trainer reports in a
JSON manifest with `pairs`, each containing `a` and `b` report paths. Paths are
resolved relative to the manifest on Windows and Linux:

```sh
python scripts/compare_predictions.py MATCHED_REPORTS.json --output NEW_COMPARISON.json
```

This program matches subjects, labels, test hashes, dataset, fold and seed.
It averages each subject's scores over matched seeds and repeated held out
appearances, then computes pooled AUC, accuracy, balanced accuracy and macro
F1. Its estimand is performance of these averaged predictions across the
pooled subjects. It differs from the suite's mean of fold metrics and from the
average performance of an individual trained model. Do not label it as mean
cross-validation performance.

The 95% intervals use a class-stratified paired subject bootstrap conditional
on the fitted models. They do not estimate retraining uncertainty, site-level
sampling uncertainty or correction for multiple comparisons. Each dataset is
analyzed separately. Submit the prespecified complete matched run set and
keep each A and B method definition consistent across folds. The checker rejects
mixed methods, budgets or interventions within a branch, different test hashes
across seeds of one fold, and unequal matched seed sets across included folds.
The output lists included folds and seeds, source report hashes and per-subject
repetition coverage; a supplied subset
does not establish that the full planned experiment matrix is complete.

## Retained failure and repair evidence

Set `case_study_run` to an existing `full_replacement` build directory, then
execute the planned case study job. This reads its report, each failed or
successful attempt, adapters, validation records and checkpoint hashes. It
does not call an Agent or train a model. It exports only allowlisted source
commits, hashes, gradient measurements, update counts and code change line
counts. Subject IDs, paths, prompt text and provider replies stay in the
private run directory. Root and per-attempt validation records must agree.

```sh
python scripts/summarize_case_study.py --run PRIVATE_REPAIR_RUN --output NEW_AUDIT
```

Controlled injected failures are labeled explicitly. A successful repair from
such a case does not estimate naturally occurring failure rates or cohort
accuracy. Missing evidence and modified adapters or checkpoints fail the audit.
