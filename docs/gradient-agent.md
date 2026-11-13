# UniBrain gradient agent

This experiment adds only a TaskForce-inspired training coordinator to the paper-aligned original UniBrain. Extraction, registration, template transfer, segmentation, ROI features and the graph classifier retain their original functions and configuration. The base coefficients remain 1, 0.1, 1 and 1. No replacement module or validation-driven loss search is used.

Four task actors control extraction, registration, segmentation and classification. Each observes its training loss and its gradient Gram row. The jointly normalized actions scale the base coefficients in the actual backward pass. Centralized critics learn from training-loss improvement and a mixed-gradient norm penalty. Replay sampling, target networks, optimizer states and RNG states are checkpointed. Inference uses the trained UniBrain; it does not require training losses or query the coordinator.

The implementation follows the TaskForce mechanism, not an exact reproduction of its benchmark system. Gradient observations use explicitly recorded parameter probes, bounded to 50,000 elements per module by default. They are not full-model gradient measurements. The records include sampled tensor names, gradient norms, cosine similarities, conflicting task pairs, actions, effective coefficients, rewards and actor/critic update counts. The conflict matrix is diagnostic evidence, not a guarantee of conflict removal or convergence.

```sh
python scripts/train_replacement.py \
  --source "$UNIBRAIN_SOURCE" --data "$FOLD_DIRECTORY" --output "$RUN_DIRECTORY" \
  --method original --setting original --paper-original --coordination taskforce \
  --dataset abide --fold 1 --epochs 20 --patience 0 --gpu 0
```

Use the corresponding ADHD fold with `--dataset adhd`. Dataset splits are preserved. Validation measures outcomes only; test data are not evaluated during these development runs. The fixed-coefficient original experiment remains a separate reference.

On 2026-11-12, both ABIDE and ADHD acceptance subsets completed eight full-model updates. Each coordinator performed four policy updates. Independent CPU inspection verified that all four actors and all four critics changed from seeded initialization. Model parameter updates and classification gradient paths were verified on the full 96-cubed architecture. These checks establish trainability and policy execution, not full-cohort convergence.

Training records are written to `updates.jsonl`; epoch validation and policy summaries are in `report.json`. `last.pt` includes the model, model optimizer, all coordinator networks and optimizers, replay, progress and RNG states. `scripts/check_policy_checkpoint.py` verifies actual coordinator parameter changes rather than trusting counters alone.

The ABIDE acceptance checkpoint was subsequently resumed from update 8 to update 10. Independent inspection verified six policy updates after restoration. CPU regression tests reproduce the next coordinator decision and parameter update exactly after restoring its state. Full-cohort training remains a separate convergence test.
