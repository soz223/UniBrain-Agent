# Generalization diagnosis

The fixed balanced 64-subject training probe is a fitting diagnostic, not a
full-cohort training estimate. Evaluate the complete training and validation splits
from one immutable checkpoint before attributing the gap to probe selection.

Run `python -m unibrain_agent.generalization_audit` with `--data`, `--manifest`,
`--checkpoint`, `--source`, `--output` and an idle `--gpu`. The manifest maps each
subject ID to its original site through `raw_img`. This audit currently accepts
the full original paper architecture. It verifies source architecture and data
hashes, uses identical evaluation paths, writes case-level predictions and reports
the probe complement, site results, calibration and stratified bootstrap AUC
intervals. These intervals condition on this checkpoint and split; they do not
measure five-fold or training-seed uncertainty. Test data are not evaluated.

`scripts/review_training_diagnosis.py` accepts the resulting report. The rule
backend identifies measured gaps and confident errors; the configured language
model may review the same evidence. Advisory review does not change a running
experiment. Site-only scores use training labels exclusively. Site imbalance and
a fitting gap are observations, not proof of a particular causal mechanism.

## Cross-subject coordination experiment

An anchored coordinator can use `--coordination-reward cross_subject`. Each update
also measures another training subject before and after the model optimizer step.
The reference excludes the updated subject and alternates diagnosis classes. Its
index is reproducible from the seed and step. Evaluation mode, model buffers and
random state are restored after each reference forward. No reference loss is
directly backpropagated into the model.

The shared policy reward is the original-coefficient-weighted mean improvement
in `log1p` task losses on that reference. The self-sample improvement remains in
telemetry but does not supply this reward. Actors, centralized critics and replay
learn from the reference reward. The original gradient safeguard, learning rate,
module definitions and failure checks remain active. A different training subject
is not an unseen validation subject; held-out benefit must still be measured.

Use `--resume-reward-upgrade` only with an allowlisted anchored checkpoint to
start a separate experiment. It retains model and Adam state and resets policy
learning because the reward semantics changed. Recovery retains the selected
reward mode. Record the warm-start step and compare against the same checkpoint
and training budget. Never describe inherited validation history as new treatment
evidence, or a few successful updates as improved generalization.
