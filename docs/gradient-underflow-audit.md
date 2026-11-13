# Classification gradient underflow

An ABIDE TaskForce run stopped before optimizer update 34,693. Replaying the saved optimizer and RNG state reproduced the failure. Classification cross entropy rounded to zero, its full parameter gradient norm was approximately 3.33e-38, and the extraction parameter gradients underflowed to zero. Extraction still had finite nonzero total gradients from the other objectives. The mask remained unsaturated over many image voxels and all five registration images were nonempty.

The audit now distinguishes a disconnected path from numerical underflow. For a required path whose gradients are all present, finite and zero, it may recompute diagnostic gradients using a scale of 1e20. This probe requires cross entropy at most 1e-7 and all original classification gradients at most 1e-12 in absolute magnitude. Every parameter on the affected path must remain connected, and the scaled probe must be finite and nonzero. Detached paths, partially missing paths, genuinely constant branches and nonfinite gradients still fail.

The original gradient measurements remain in the report. Scaled measurements are marked diagnostic only. The probe does not accumulate parameter gradients or change the model objective, TaskForce decisions or optimizer update. Updates log affected modules, and the report counts underflow events.

`--resume-gradient-audit-upgrade` permits an explicit migration from the recorded pre-fix trainer hash. Every other checkpoint identity field must match, including model code, data hashes, optimizer protocol and coordinator. Preserve the old checkpoint and failed attempt before continuation. The migration is reported separately and does not change the original control run.

A later attempt stopped after update 34,867: the classification-only feature
module had native classification gradient L1 approximately 2.85e-38, but its
weighted total gradient was zero. The audit now also probes the weighted total
objective for classification-only modules and covers the input classification
path. Persistent anomalies remain bounded. See [training recovery](training-recovery.md)
for monitoring, checkpoint recovery, and the distinction between diagnostic
gradients and actual optimizer gradients.
