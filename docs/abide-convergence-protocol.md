# ABIDE convergence investigation

This experiment compares the complete UniBrain paper configuration with the same model controlled by TaskForce gradient agents. It investigates learning failure; it is not a completed performance result.

## Shared configuration

* Retain the existing ABIDE fold 1: 651 training, 217 validation and 217 test subjects, with 96 cubed images.
* Use the full 271,483,811 parameter paper model and the pinned upstream components. This is the paper implementation, not a claim of bitwise reproduction of upstream `main.py`.
* Start both models from seed 17. Use the same subject ordering, augmentation, Adam learning rate of 0.00001 and weight decay of 0.000001.
* Set 1,000 epochs and patience 0. A 43,200 second execution budget pauses training at an update boundary, retains optimizer and policy state, and permits continuation. Epoch evaluation and checkpoint writing may finish after the deadline.
* Select `best.pt` using validation classification cross entropy. Retain `last.pt` for continuation. Checkpoint selection never stops training.
* Evaluate the complete validation split each epoch. Also evaluate a fixed, class-balanced 64-subject training subset in evaluation mode. This subset diagnoses fitting and is not a full training-set estimate or a selection criterion. Keep the test split unused for tuning.

## Input scale correction

The prepared subject images are in [0, 1], while the prepared intensity template has a maximum of approximately 95.61. The upstream sample intensity template is in [0, 1]; its SHA256 is `9f7cbef2016a83418ed3438f28f166e0fe77c62b0bba32f043cdc41e00889b28`.

Both runs use `--template-normalization minmax`. Scaling occurs after loading and affects only the intensity template. It preserves geometry and categorical labels, leaves source files unchanged, and records the source and resulting ranges. Checkpoint inference restores the same transform. This correction addresses an observed mismatch; its impact on convergence remains to be measured.

## Gradient coordination

The control retains the paper loss coefficients: extraction 1, registration 0.1, segmentation 1 and classification 1. The treatment uses the TaskForce core: full parameter gradient Gram observations, four local actors, centralized critics, shared training reward, replay and target networks. The replay batch size is 64. Validation never supplies policy rewards.

The treatment explicitly enables `--taskforce-action-bound 1`. Its actor output is `tanh(a)` and exploration is clipped to the same interval before softmax. Target actors use the same bound. This is a documented stability choice for this experiment, not a verified setting from the TaskForce authors. Each task's softmax share is at least `1 / (1 + 3 exp(2))`, approximately 0.043. Applied loss coefficients still divide each share by its observed gradient norm. The bound addresses near-zero shares observed in an earlier, different coordinator; it does not establish that the new coordinator would otherwise collapse.

## Running and interpreting

Run `scripts/train_replacement.py` in separate processes on idle GPUs. Common options are:

```text
--source SOURCE --data ABIDE_FOLD --output RUN
--method original --setting original --paper-original
--dataset abide --fold 1 --epochs 1000 --patience 0 --seed 17
--wall-seconds 43200 --train-probe-size 64 --template-normalization minmax
--verify-gradients --checkpoint-mode resume --selection-metric classification_loss
```

The control adds `--coordination none`; the treatment adds `--coordination taskforce --taskforce-profile paper --taskforce-action-bound 1 --coordination-batch-size 64`. Select each physical GPU with `--gpu`. To continue, retain the same code snapshot and arguments and add `--resume RUN/last.pt`; a new wall budget does not alter checkpoint identity.

Inspect `report.json` for complete epoch metrics, training-probe AUC, prediction spread and diagnosis proportions. Inspect `updates.jsonl` for losses, gradient norms, applied weights and actor/critic updates. Registered backend training runs expose aggregate telemetry to the Training page. Neither successful parameter updates nor a few favorable minibatch losses establish convergence. Compare validation trajectories at matched epochs or optimizer steps. Different GPU models and policy overhead make wall-time-only accuracy comparisons confounded.
