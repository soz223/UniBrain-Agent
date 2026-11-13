# UniBrain gradient agent

This experiment adds a TaskForce training coordinator to the paper-aligned original UniBrain. Extraction, registration, template transfer, segmentation, ROI features and the graph classifier retain their original functions and configuration. No replacement module or validation-driven loss search is used. New experiments default to the [core profile](taskforce-core-audit.md); historical experiments use the explicitly named legacy profile.

Four task actors control extraction, registration, segmentation and classification. Each observes its training loss and its gradient Gram row. Softmax shares combine normalized gradients in the actual model update. Centralized critics learn from training loss improvement and a mixed-gradient norm penalty. Replay sampling, target networks, optimizer states, OU noise and RNG states are checkpointed. Inference uses the trained UniBrain checkpoint.

The core profile observes all trainable parameters. The legacy profile samples up to 50000 parameter elements per module and retains base coefficients 1, 0.1, 1 and 1. Records distinguish these profiles, including gradient scope, task shares, effective loss coefficients, rewards and optimizer counts. A conflict matrix is diagnostic evidence, not a convergence guarantee. The audit lists the remaining task adaptations and implementation choices.

```sh
python scripts/train_replacement.py \
  --source "$UNIBRAIN_SOURCE" --data "$FOLD_DIRECTORY" --output "$RUN_DIRECTORY" \
  --method original --setting original --paper-original --coordination taskforce \
  --dataset abide --fold 1 --epochs 20 --patience 0 --gpu 0
```

Use the corresponding ADHD fold with `--dataset adhd`. Dataset splits are preserved. Validation measures outcomes only; test data are not evaluated during these development runs. The fixed-coefficient original experiment remains a separate reference.

On 2026-11-12, the legacy ABIDE and ADHD acceptance subsets completed eight full-model updates. Each coordinator performed four policy updates. Independent CPU inspection verified that all four actors and all four critics changed from seeded initialization. Model parameter updates and classification gradient paths were verified on the full 96-cubed architecture. This evidence belongs to the legacy profile and must not be attributed to the new core profile.

Training records are written to `updates.jsonl`; epoch validation and policy summaries are in `report.json`. `last.pt` includes the model, model optimizer, all coordinator networks and optimizers, replay, progress and RNG states. `scripts/check_policy_checkpoint.py` verifies actual coordinator parameter changes rather than trusting counters alone.

The legacy ABIDE acceptance checkpoint was subsequently resumed from update 8 to update 10. Independent inspection verified six policy updates after restoration. CPU regression tests now check exact next-decision and update equivalence for both profiles. Full-cohort training remains a separate convergence test.

The core profile subsequently passed two full-model updates on each dataset, followed by independent parameter and functional action checks. All four actors and critics learned, and the learned policies changed task shares with exploration disabled. Its [separate acceptance record](../catalog/taskforce_core_acceptance_20261112.json) prevents attribution of legacy results to the new controller.
