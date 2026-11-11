# Full model comparison runner

`scripts/train_replacement.py` implements a common training and evaluation loop
for the full UniBrain architecture and its full replacement candidates. The
unchanged upstream reproduction remains `scripts/run_upstream.py`. Selecting
`original` here gives a matched training control, not an assertion that this
loop executes the unmodified upstream training entry point. In particular,
this runner explicitly restores training mode after each validation pass.

No experiment launches from configuration creation or `--preflight`. The latter
checks resources, hashes and split identities, but does not certify CUDA execution.
Training checks GPU allocation before constructing its CUDA model. Use the
cluster scheduler for exclusive allocations; the process check is advisory.

## Settings

| Setting | Candidate optimization | Host optimization |
| :--- | :--- | :--- |
| `original` | Original module set | Joint original objective |
| `original_completed` | Original modules with explicit connection repairs | Joint objective |
| `joint` | Trained together with the host | Joint objective |
| `independent_official` | Official checkpoint, frozen and detached | Host only |
| `independent_pretrained` | Checkpoint trained on the current training fold, frozen and detached | Host only |
| `independent_finetuned` | Standalone task loss first, then frozen and detached | Host training after standalone phase |

Random initialization is an explicitly labeled alternative; it cannot be called
`independent_official`. Candidate weights require a source description and a
verified SHA256. Training fold weights must declare the same dataset and fold.
The program verifies these declarations and hashes; it cannot establish the
truth of external provenance statements. Keep training records with the weights.

SynthStrip uses the accepted generated extraction adapter. VoxelMorph and LapIRN
use five original affine stages followed by their full dense core. The entire
registration block is treated as the replaced module for training and freezing.
Their frozen official comparison therefore requires a shared trained original
host checkpoint, with matching training fold provenance. Freezing randomly
initialized affine stages is rejected for that comparison. All compared runs
must use the same host initialization and candidate initialization.

nnU-Net needs a real dataset specific `3d_fullres` plan supported by the candidate
builder. Its task specific pretrained weights should be labeled `training_fold`,
not official weights. Auxiliary tissue outputs receive losses weighted by
normalized inverse powers of two with nearest resampled targets. This is an
explicit research adaptation, not the complete official nnU-Net trainer.
Use [the training fold preparation program](nnunet-preparation.md) to export
only training subjects and call the official planner with explicit coordinate
units. An incompatible native plan is retained and reported, never rewritten.
The unchanged planned network executes on UniBrain's 96 cubed volume when its
cumulative strides permit that size. The native patch and execution shape are
recorded separately; this is an explicit full volume execution adaptation.

## Execution

```sh
python scripts/train_replacement.py \
  --source /path/to/reg_unibrain --data /path/to/exported/fold \
  --output /path/to/new/run --method synthstrip --adapter /path/to/adapter.py \
  --setting joint --initialization random --epochs 20 --seed 17 --gpu 0 \
  --dataset abide --fold 1 --evaluate-test
```

Use `--max-steps 1` or `2` for a bounded acceptance check. This is a total update
budget across standalone and pipeline phases, not a per phase allowance. For a
one epoch smoke test, use an explicitly exported tiny dataset with distinct
subjects in train, validation and test. Existing split arrays are never edited.
An incomplete epoch does not produce a validation result or a test result.
`--eval-limit` explicitly marks partial validation and test evaluation by
recording both evaluated and full split counts. Do not use it for final results.

Standalone epochs are additional work. `pretrain_steps` and total `steps` are
recorded separately. `train_seconds` includes training, validation and checkpoint
writing, and accumulates across resumptions; it is not pure CUDA kernel time.
Compare total computation or an explicit joint budget rather than calling equal
pipeline epoch counts equal cost.
`--patience N` stops pipeline training after N validation epochs without a
strictly lower reference loss. Zero disables early stopping. Use this for the
longer budget sensitivity experiment with a declared maximum epoch limit.

## Checkpoints and evaluation

`last.pt` stores model, optimizer, Python/NumPy/Torch RNG states and progress,
including position within the deterministic epoch order. Resume in the same run
directory using `--resume /path/to/run/last.pt` and a larger total step budget.
Architecture, fold hashes, weights, adapter, seed and optimization settings must
match. Checkpoints are written through a temporary file and renamed atomically.
Only the latest state and best validation state are retained.

`best.pt` is selected only by the fixed validation reference loss. Test evaluation
requires `--evaluate-test`, runs after training, and loads the validation selected
checkpoint. Per subject test probabilities and metrics are stored in `test.json`.
The test split never selects weights, repairs, stopping points or hyperparameters.
`candidate_core.pt` and its metadata record standalone pretraining for reuse.
Dense candidate exports contain the dense core only; affine initialization is
still supplied through the shared full host checkpoint.

Diagnosis metrics include accuracy, balanced accuracy, macro F1, AUC, sensitivity
and specificity. Undefined single class AUC or rates are null. Intermediate
metrics use the existing reference masks. Dense registration evaluation samples
with the actual dense grid, not its affine precursor. The logs also record one
sampled parameter update per active module; this does not replace complete
gradient and spatial verification from the integration experiment.

## Mechanism controls

`--save-predictions` saves test brain, tissue and atlas label arrays and their
reference arrays for the separate surface metric evaluator. Case numbers name
the private output folders; the test report retains their subject mapping.
Physical spacing is not invented from these arrays. Supply verified spacing
and units to the offline evaluator before reporting millimetre distances.

`--secondary-extraction` combines an accepted SynthStrip adapter with a dense
registration candidate. `--composition-mode simultaneous` initializes both
replacements together; `sequential` requires a previous trained extraction
checkpoint, report and independently supplied SHA256. This first composition
entry requires joint training of all modules. Previous dataset and fold
provenance must match the current experiment. Explicit extraction weights need
their own origin and SHA256.

`--trainable new` or `host` controls the optimized parameter groups.
Completely frozen modules run in evaluation mode during training, so their
BatchNorm buffers remain fixed. Input gradients through frozen modules remain
available unless an explicit detached boundary is selected.
`--cut-extraction` stops gradients through the extracted image but leaves the
brain mask objective intact. `--loss-scale` rescales only the replaced stage's
task loss. `--regularization` explicitly sets dense field smoothness weight.
`--connect-paths` enables the separately defined connection repair forward pass;
`--cut-boundary`, `--repair-strategy` and `--undo-repair` select its ablations.
Every choice is included in the report and resume identity.

Paths and device allocations are external arguments. The code is designed for
Windows and Linux and respects `CUDA_VISIBLE_DEVICES`. Publication claims still
require successful full pipeline acceptance runs and completed scientific
comparisons; unit tests alone establish neither.
