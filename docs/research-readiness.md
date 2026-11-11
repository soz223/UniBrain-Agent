# Research program coverage

Implementation, acceptance checks and scientific results are different states.
This document records program scope. The portable execution contract is the
[experiment suite](experiment-suite.md); each selected run stores its exact
configuration, input and source hashes, observed metrics and checkpoints.
The existing original baseline service is not modified by these programs.
Measured software checks are recorded in
[the public verification record](../catalog/paper_program_verification.json).

## Executable components

| Experiment | Program | Measured acceptance and remaining boundary |
| :--- | :--- | :--- |
| Original baseline | `scripts/run_upstream.py` | Existing upstream reproduction; source pinned and unchanged |
| Matched full model training | `scripts/train_replacement.py` | Full SynthStrip host completed one update, validation, test and checkpoint saving on a distinct tiny sample split |
| Full registration replacements | `full_candidates.py` and the comparison trainer | Full VoxelMorph host passed one CUDA joint update, core and spatial agreement, all 24 candidate parameter gradients and diagnosis gradients into registration; full three-level LapIRN passed CPU core forward, gradients and an update |
| Planned tissue replacement | `scripts/prepare_nnunet.py` and the comparison trainer | Official planner ran on a training sample; its unchanged 64 by 96 by 96 patch plan produced a 16,548,560 parameter candidate that passed CPU forward, gradients and an update at full 96 cubed execution; full host GPU acceptance remains pending |
| Agent generation and repair | `scripts/run_full_replacement.py` and the integration benchmark | Actual earlier SynthStrip generation and repair trace retained; controlled injected fault is labeled explicitly |
| Gradient path controls | `full_connectivity.py` and trainer options | Soft label transfer, straight through and Gumbel strategies, selected cuts, mask/graph/sampling undo controls; CPU gradient tests |
| Independent versus joint training | Comparison trainer settings | Explicit weight provenance, frozen parameters and buffers, standalone training followed by host training, joint training; completed cohort comparison remains pending |
| Two module composition | `scripts/run_composition.py` | Actual full trained extraction checkpoint reconstructed on CPU; 256 state tensors retained and 26 new VoxelMorph tensors added; GPU composition remains pending |
| Site holdout | `scripts/prepare_site_split.py` | Subject-exact CSV site metadata, isolated export, train and validation stratification, template provenance; actual site metadata must be supplied |
| Repetitions and long budgets | Experiment suite and comparison trainer | Seeds, folds, epoch caps, early stopping, atomic checkpoint and RNG resume; no full batch launches implicitly |
| Intermediate quality | Trainer plus `scripts/evaluate_masks.py` | Dice, registration alignment, HD95 and ASSD; automatic reference labels and explicit physical or voxel units |
| Diagnosis comparisons | Trainer, suite summary and `scripts/compare_predictions.py` | Per subject test outputs; fold summaries and separate paired subject confidence intervals; smoke results excluded |
| Case study | `scripts/summarize_case_study.py` | Reads actual attempts and verifies adapter/checkpoint hashes; no new generation or training |

## Scope that is not implemented yet

The full paper plan includes 14 candidate methods. Full candidate construction
and generated adapter evaluators currently cover SynthStrip, VoxelMorph,
LapIRN and the supported nnU-Net plan.
The other ten candidates still have the earlier compact implementations and do
not yet have full architecture integration evaluators. A compact check cannot
complete their full pipeline experiment slots.

Named external systems such as NEURA, NEXUS, NeuroPilot and MLE STAR need their
actual task and data integrations. The external executor protocol is a command
contract, not those implementations. The classical stagewise comparator also
needs a declared executable pipeline. These jobs remain blocked until the real
programs and method-specific interfaces are supplied and checked.

Consequently, the repository must not yet be described as a complete execution
of every paper baseline. Neither a planned row nor a configured command proves
trainability, method fidelity or a performance advantage.

## Protocol boundaries

Original source reproduction and the matched comparison trainer are separate.
The latter restores training mode after validation, selects a checkpoint on
validation, and optionally evaluates test. Its result is not relabeled as an
unchanged upstream `main.py` run.

Dense replacements retain original affine prealignment. nnU-Net imports a
dataset-specific architecture and uses the declared UniBrain objective; it does
not reproduce the entire official training pipeline. Full path interventions
are explicit changes to label sampling and graph weights. The original unused
parcellation head remains unused unless a future experiment explicitly changes
that design. Do not claim gradients through every instantiated parameter.

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
