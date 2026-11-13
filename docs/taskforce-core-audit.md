# TaskForce core mechanisms in UniBrain

The reference is TaskForce, CVPR 2026, equations 5 to 13, Algorithm 1, and supplementary Appendix A and B. `taskforce_paper.py` implements the core profile. `taskforce.py` retains the legacy controller for historical experiments. Prompt based research and diagnostic assistants are separate from these trainable RL agents.

## Component audit

| Core component from Appendix A | Implementation | Executable evidence |
| :--- | :--- | :--- |
| Gradient Gram observation | Autograd on every trainable parameter; a 4 by 4 Gram matrix plus four log transformed losses | Analytic Gram test includes unused parameters and preserves existing gradients |
| Multiple task agents | Four separate actors, one for each original task loss | Parameter checks establish updates in every actor |
| Centralized training | Four critics each receive all 20 observation values and all four raw action logits | Critic input checks and independent checkpoint parameter comparisons |
| Decentralized execution | Each actor reads its own five-value observation only | Changing another actor's observation leaves its output unchanged |
| Shared loss and gradient reward | Sum of log loss improvements plus minus 0.001 times the previous mixed gradient squared norm | Independent reward calculation and transition association test |

The four tasks are extraction, registration, tissue segmentation and classification. Atlas transfer and graph construction participate through their connected task losses. They do not acquire fabricated independent supervision.

## Actual model update

For raw task gradients `g_j`, use scales `s_j = ||g_j||_2`, with scale one at numerical zero. The local observation is one row of the normalized Gram matrix plus `log(1 + L_j)`. Actors produce logits `a`; `w = softmax(a)`. The model loss is `sum_j (w_j / s_j) * L_j`, with detached coefficients. Its backward pass therefore gives exactly `G = sum_j w_j * (g_j / s_j)`. Tests compare this gradient and the resulting SGD parameter update with an independent analytic calculation. The UniBrain optimizer remains Adam with its original configuration.

The next training sample completes the previous transition. Its shared reward uses the preceding action and Gram matrix. Replay stores `(O_previous, a_previous, R, O_current)`. The current model update finishes before the policy replay update. Each actor has its own centralized critic and target actor/critic. This follows the main paper's single-episode interpretation and requires no second forward pass on the same subject. Validation and test data never produce policy rewards.

## Policy configuration

Appendix B specifies three-layer MLPs with width 64, Adam at 0.0005 for both network types, replay capacity 100000, minibatch 64, target coefficient 0.01, discount 0.95 and OU exploration whose amplitude decays from 0.3 to 0.05 over 10000 updates. These are the defaults. Reward weights are 1 and 0.001. Loss, gradient and reward normalization follow the stated implementation requirements.

Some details are not specified in the paper. This implementation uses ReLU, zero actor output initialization, L2 gradient normalization, minibatch reward standardization when variance is nonzero, and OU parameters theta 0.15 and sigma 0.2. `--taskforce-gradient-normalization none` exposes an explicit unnormalized control. These choices are recorded and should not be represented as verified official source behavior.

Algorithm 1 computes the current transition reward but prints `R_previous` in the insertion line. We associate the newly measured improvement with the preceding action, as defined by the transition and reward equations. This resolves the indexing ambiguity explicitly.

TaskForce describes shared model parameters. UniBrain has a branched serial graph rather than a common backbone for all four tasks. We use the complete trainable parameter space, with unused task coordinates contributing zero. Consequently, some low cosines reflect disjoint parameter support. The original convex-gradient argument does not establish convergence of this nonconvex UniBrain model trained with Adam and stochastic policies.

## Verification and reproduction

Run the controller tests with `python -m unittest tests.test_taskforce_paper tests.test_policy_checkpoint`. They check the model gradient, actual actor/critic updates, policy ordering, decentralized inputs, shared reward and exact resume behavior. `scripts/validate_taskforce_core.py` runs two updates of the complete paper-aligned UniBrain on a prepared ABIDE or ADHD fold, then independently verifies each actor and critic against seeded initialization. It uses replay batch one only to exercise policy learning within two updates; normal training defaults to batch 64.

```sh
python scripts/validate_taskforce_core.py \
  --source "$UNIBRAIN_SOURCE" --data "$FOLD_DIRECTORY" \
  --output "$NEW_RUN_DIRECTORY" --dataset abide --fold 1 --gpu 0
```

Use a new output directory. The script checks GPU occupancy before loading the model. It preserves source and data files. `report.json`, `updates.jsonl`, `last.pt` and `policy_check.json` bind configuration, measurements, model and policy state. A short acceptance run demonstrates executable mechanisms, not convergence or a benefit over the original UniBrain. Those claims require matched full-cohort experiments.
