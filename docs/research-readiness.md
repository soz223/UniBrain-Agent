# Research program coverage

Implementation, acceptance checks and scientific results are different states.
This document records program scope. The portable execution contract is the
[experiment suite](experiment-suite.md); each selected run stores its exact
configuration, input and source hashes, observed metrics and checkpoints.
The existing original baseline service is not modified by these programs.
Measured software checks are recorded in
[the public verification record](../catalog/paper_program_verification.json).
The [extended acceptance record](../catalog/full_program_acceptance_20261111.json)
contains the subsequent ABIDE and ADHD checks. Historical checks below retain
their original scope; short runs do not establish full cohort performance.

## Executable components

| Experiment | Program | Measured acceptance and remaining boundary |
| :--- | :--- | :--- |
| Original baseline | `scripts/run_upstream.py` | Existing upstream reproduction; source pinned and unchanged |
| Matched full model training | `scripts/train_replacement.py` | All 14 candidate integrations completed two joint updates in the full host on each dataset subset; checkpoints, intended gradient paths and parameter changes were checked |
| Full registration replacements | `full_candidates.py` and the comparison trainer | Full VoxelMorph and three-level LapIRN passed two full host GPU updates on both dataset subsets, including required diagnosis gradients |
| Planned tissue replacement | `scripts/prepare_nnunet.py` and the comparison trainer | Actual train-only plans and full host GPU checks passed on both dataset subsets; the auxiliary tissue head receives tissue supervision, not diagnosis gradients |
| Agent generation and repair | `scripts/run_full_replacement.py` and the integration benchmark | Actual earlier SynthStrip generation and repair trace retained; controlled injected fault is labeled explicitly |
| Additional full candidates | `full_preprocessing.py`, `full_anatomy_candidates.py`, `full_graph_candidates.py` | Ten added candidates passed native-size CPU checks and full host GPU checks on both dataset subsets; earlier failures remain in the execution history |
| Selection to training cycle | `scripts/run_research_cycle.py` | Actual Codex selection, generated BrainNetCNN adapter, independent acceptance and two joint updates completed on both dataset subsets; the accepted code hash was preserved |
| Training coordination | `taskforce.py` and the comparison trainer | Eight full host updates on each dataset; all four actors and four critics changed parameters with four optimizer steps each; overall validation loss did not decrease |
| Gradient path controls | `full_connectivity.py` and trainer options | Soft label transfer, straight through and Gumbel strategies, selected cuts, mask/graph/sampling undo controls; CPU gradient tests |
| Independent versus joint training | Comparison trainer settings | Explicit weight provenance, frozen parameters and buffers, standalone training followed by host training, joint training; completed cohort comparison remains pending |
| Two module composition | `scripts/run_composition.py` | Simultaneous and sequential SynthStrip plus VoxelMorph replacement passed on both dataset subsets; both modules updated twice, with prior training and state transfer recorded separately |
| Saved pipeline inference | `scripts/evaluate_replacement.py` | ABIDE and ADHD BrainNetCNN checkpoints were reconstructed and evaluated on their two-subject validation subsets with zero optimizer updates and strict state loading |
| Site holdout | `scripts/prepare_site_split.py` | Subject-exact CSV site metadata, isolated export, train and validation stratification, template provenance; actual site metadata must be supplied |
| Repetitions and long budgets | Experiment suite and comparison trainer | Seeds, folds, epoch caps, early stopping, atomic checkpoint and RNG resume; no full batch launches implicitly |
| Intermediate quality | Trainer plus `scripts/evaluate_masks.py` | Dice, registration alignment, HD95 and ASSD; automatic reference labels and explicit physical or voxel units |
| Diagnosis comparisons | Trainer, suite summary and `scripts/compare_predictions.py` | Per subject test outputs; fold summaries and separate paired subject confidence intervals; smoke results excluded |
| Case study | `scripts/summarize_case_study.py` | Reads actual attempts and verifies adapter/checkpoint hashes; no new generation or training |

## Coverage and remaining validation

The full paper plan includes 14 candidate methods. Full candidate constructors,
generated stage adapters, independent evaluator paths and configuration
generation now cover all 14. nnU-Net requires a real compatible dataset plan.
This code coverage is separate from execution evidence. Native-size CPU
component checks, full host GPU checks and actual provider trials have distinct
records. The full host matrix uses two training and two validation subjects per
dataset from the preserved first fold. It does not use reduced architectures.
Its subjects and automatic reference labels are not a full cohort experiment.
Six earlier failures remain recorded. Corrected executions used new output
directories. A source-bound initialization check distinguishes the first zero
extraction gradient from a detached path and requires measured recovery after
one real update. The model initialization and loss were not changed for that
check. Repeated checks do not increase the 28-cell method coverage.

The new preprocessing candidates are RED-WGAN, DeepN4, deepNormalize and the
HarmonizingFlows harmonizer component. Anatomy candidates add complete ERNet
extraction/registration cores and the FastSurfer and Nobrainer backbones with
explicit new 117-channel AAL heads. Graph candidates add the FBNetGen pairwise
MLP, the declared third-party BrainNetCNN implementation and BNT with its
original 100-node pool. Their sMRI adaptations and omitted upstream training
recipes are recorded; they are not relabeled as complete official tools.

Named external systems such as NEURA, NEXUS, NeuroPilot and MLE STAR need their
actual task and data integrations. The external executor protocol is a command
contract, not those implementations. The classical stagewise comparator also
needs a declared executable pipeline. These jobs remain blocked until the real
programs and method-specific interfaces are supplied and checked.

The [external baseline audit](external-baseline-readiness.md) identifies official
source snapshots, missing adapters and incomplete public implementations. Its
catalog records readiness only and does not register or validate executors.

Codex/Claude provider calls under the shared general, checking and workflow
prompts are controlled prompting comparisons. They do not reproduce all tools
or behavior of a general external Agent system. The experiment suite and direct
14-candidate evaluators remain CLI programs. The dashboard's
[full research panel](full-research-workspace.md) now launches the real bounded
research cycle and shows checks, accepted code and joint-training progress.
Its older custom stage editor remains explicitly labeled as a compact prototype.

Consequently, the repository must not yet be described as a complete execution
of every paper baseline. Neither a planned row nor a configured command proves
trainability, method fidelity or a performance advantage.
The core acceptance checks are complete. Full cohort comparisons, site results
and external system integrations remain separate work. The existing baseline
service was not restarted or upgraded during these local checks.

## Protocol boundaries

Original source reproduction and the matched comparison trainer are separate.
The latter restores training mode after validation, selects a checkpoint on
validation, and optionally evaluates test. Its result is not relabeled as an
unchanged upstream `main.py` run.

Dense replacements retain original affine prealignment. nnU-Net imports a
dataset-specific architecture and uses the declared UniBrain objective; it does
not reproduce the entire official training pipeline. Full path interventions
are explicit changes to label sampling and graph weights. The original unused
parcellation head remains unused in the original protocol. The new learned
parcellation variants explicitly combine predicted and transferred atlas
probabilities and add their declared training objective. Do not extend that
connection claim to the original model or every instantiated parameter.

Tissue and AAL masks are automatic reference labels. Native physical geometry
is absent from the structured NPY inputs. Unit grid coordinates support tensor
checks and voxel distances, but do not certify native orientation or millimetre
alignment. Site metadata and template provenance must be independently checked.

Keep source data, private subject reports, checkpoints and AI transcripts in
ignored runtime directories. Paths are supplied as arguments or relative to a
configuration file. Credentials stay in environment variables or local provider
settings. The CPU test workflow runs on Windows and Linux; full CUDA acceptance
is specific to its recorded environment and must be repeated after migration.

Use one or two updates for acceptance before scheduling a scientific job. A
short check establishes execution and measured paths, not accuracy or the
effectiveness of Agent decisions. Retain all failures and account for previous
standalone or sequential training when comparing compute budgets.

New matched comparison runs select the earliest minimum validation diagnosis
cross entropy checkpoint by default. The declared selection metric is included
in the report and checkpoint identity. The optional historical reference loss
includes extraction BCE on the last thresholded mask. Brain Dice uses
the product of all extraction masks. The original five stage extraction and
single stage SynthStrip therefore contribute different mask objects to that
selection loss. Hard mask BCE can dominate the other validation terms. Existing
reference loss runs retain that scope; their results are not relabeled under
the new selection rule. Full research cycles verify the selected checkpoint's
method, adapter, cohort hashes, fold and seed against its internal metadata and
validation history. Test aggregates are recomputed from the recorded subjects.
The nnU-Net tissue head receives its segmentation objective, while diagnosis
does not consume its logits. Its report records the missing diagnosis gradient
path explicitly instead of claiming that diagnosis trains every stage.
