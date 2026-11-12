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
| `original_frozen_stages` | Fixed trained original stages | Original classifier only |
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

`original_frozen_stages` requires a trained original host from the same dataset
and fold. Trainer checkpoint metadata must show positive training steps. Raw
states require a separate `--host-report` with the checkpoint hash, successful
status, update count, dataset and fold. Every stage except `gnn_pred` is frozen,
its outputs are detached, and its BatchNorm buffers stay fixed. This comparator
represents fixed pretrained UniBrain stages and a trainable classifier. It does
not represent an ANTs or FSL workflow.

The ten extended candidates currently support joint training with explicitly
random initialization. Their wrapper checkpoint import requires additional
round trip validation before standalone or pretrained controls are enabled.

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
`last.pt` also retains the complete optional coordinator state. `best.pt` contains
only model weights, identity and progress and cannot resume training. Before
each atomic checkpoint write, the runner estimates the temporary file size and
requires a further 1 GiB of free disk space. Existing results are retained when
the check fails.

`--checkpoint-mode resume` is the default described above. Short acceptance runs
can select `--checkpoint-mode weights`: both `best.pt` and `last.pt` contain only
model weights, progress and identity, marked `evaluation_weights_only`. These
checkpoints support evaluation and declared same fold initialization, but cannot
resume an optimizer or coordinator. When a new best and last snapshot have
identical progress and identity, the runner atomically links them to the same
stored file. Filesystems without hard link support use normal serialization.
Later updates replace the last path atomically and preserve the previous best.
Existing runs are never scanned or rewritten to deduplicate their checkpoints.

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

`--coordination taskforce` enables the existing task actor and centralized critic
implementation during pipeline training. Each task observes its gradient Gram
row and loss and proposes a positive normalized loss weight. The coordinator
learns from recent loss improvement and a gradient norm penalty. This is a
TaskForce inspired research implementation, not an exact reproduction of that
paper. Validation and test do not query or update the coordinator.

The four controlled losses are extraction, registration, tissue segmentation and
diagnosis. A learned AAL head's additional supervised loss and optional dense
regularization retain their declared fixed coefficients. Gradient probes cover
trainable parameters from all model children by default, subject to a per
module element budget. `--probe-modules` selects a declared subset. The report
retains actual tensor names and element count. Frozen parameters are excluded.

The replay batch defaults to four transitions. The first two training updates
therefore exercise decisions but do not establish policy learning. Every update
records weights, Gram matrix, reward, replay size and actual actor and critic
optimizer step counts. Resume restores actors, critics, target networks, their
optimizers, replay, the preceding transition and the coordinator RNG. A CPU
regression verifies identical next decisions and optimizer updates after saving
and restoring that state.

`--verify-gradients` measures diagnosis gradients to every selected candidate,
total objective gradients to candidate parameters, and the input image gradient
on each training step. Connected joint configurations require finite nonzero
paths when the candidate is trainable and no gradient ablation is selected.
Protocol version 2 also measures diagnosis gradients to the original input in
the same reverse pass as candidate parameter gradients. This detects a graph
or classifier adapter that detaches its input while auxiliary losses still
train earlier stages. Reports from the preceding protocol retain their legacy
boundary status and do not certify this additional path.
An explicit exception is the nnU-Net tissue head: diagnosis does not consume
its tissue logits, so that path is recorded as expected absent. Its tissue loss
must still produce parameter gradients and updates. This configuration cannot
be described as diagnosis gradients reaching every stage.

Learned parcellation heads use 117 classes including background. Subject AAL
targets must be integers drawn exactly from the template vocabulary; unknown
labels are rejected instead of being silently mapped to a neighboring class.

The current validation reference loss retains the upstream style extraction
BCE on the last thresholded mask, whereas brain Dice evaluates the product of
all masks. Different extraction stage counts can therefore affect checkpoint
selection. Final scientific comparisons should prespecify a common selection
protocol; acceptance checks establish executable behavior, not comparable
clinical performance.

Paths and device allocations are external arguments. The code is designed for
Windows and Linux and respects `CUDA_VISIBLE_DEVICES`. Publication claims still
require successful full pipeline acceptance runs and completed scientific
comparisons; unit tests alone establish neither.
