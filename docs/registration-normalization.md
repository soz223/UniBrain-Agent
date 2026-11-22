# Registration normalization

The five registration stages share one encoder. Its BatchNorm history mixes the
successive image distributions. The `per_image` policy uses each current image's
spatial statistics in both training and evaluation. It retains the original
affine parameters and checkpoint keys. Stored running buffers remain available
for legacy checkpoints but do not affect this policy. A forward accepts one
subject, matching the full UniBrain training interface.

Select `--registration-normalization per_image` for both comparison arms.
The policy and implementation hashes are bound to the checkpoint. The evaluator
restores that recorded policy and rejects inconsistent report metadata. Historical
checkpoints retain their `stored` interpretation unless explicitly migrated.

`--resume-normalization-upgrade` permits only audited source versions. Model
weights, optimizer state and RNG state survive. The Agent policy is initialized
again because the reward transition distribution changes. Counters, validation
history and best-model selection start a new continuation phase; the archived
source identity, epoch and step remain in `normalization_migration`. Original
logs and checkpoints must remain in their original directories.

For a matched continuation, copy the same original checkpoint to both new arms.
The Agent uses the audited `classification_advantage` reward. These are warm-start
experiments, not a claim of exact original-paper reproduction or an isolated
BatchNorm-only Agent ablation. The continuation epoch budget is additional to the
archived training. Validation selects checkpoints; test labels are not used for
repair. Confirm full-model gradients, updates, save/resume and complete validation
before reporting improvement.

Run `scripts/verify_code_release.py` against the same release manifest on each
server before launch. Its file and dependency checks supplement the checkpoint's
data and configuration identity checks. Hardware differences may still affect
floating-point execution.
