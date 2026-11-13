# Training recovery

The trainer distinguishes a disconnected path from an isolated FP32 underflow.
It keeps native measurements and optimizer gradients unchanged. When classification
loss is at most 1e-7 and its gradients are tiny, a diagnostic backward pass scales
the same graph by 1e20. A zero path passes only if every expected tensor is present
and the scaled gradients are finite and nonzero. Total-gradient probes apply only
to the classification-only ROI feature and graph classifier modules. Missing,
nonfinite, structurally zero, and persistently underflowing paths still fail.
The ninth consecutive recovered event in a path stops the worker for diagnosis.

For a saturated single-subject cross entropy, multiplying the loss cannot recover
a softmax probability that has already rounded to zero. A second diagnostic
verifies the direct NLL/log-softmax graph and the correct target, then evaluates
the exact CE gradient direction normalized by the error probability. Its seed is
minus one for the target and the conditional softmax over all other classes.
This is a positive rescaling of the analytic CE derivative, not an injected
optimizer gradient. The error probability must be at most 1e-12. Total-objective
acceptance additionally verifies a finite positive derivative with respect to the
classification loss and is restricted to the original paper model's
classification-only modules. A zero objective coefficient or cut loss still fails.

`gradient_recoveries.jsonl` preserves numerical recovery events, including replayed
attempts. `health.json` reports the current phase, committed step and counters.
Counters and streaks are stored in the resumable checkpoint's progress state.
Scaled diagnostic gradients are never reported as optimizer gradients or parameter
updates. The original optimizer, loss, modules, split and policy stay unchanged.

Run the portable supervisor with:

```sh
python -m unibrain_agent.training_supervisor supervisor.json
```

The manifest contains `command` (argument array), `cwd`, `output`, `environment`,
`wall_seconds`, optional `max_restarts` (default 2), and `stall_seconds` (default
900). The command must enable `--verify-gradients` and use the matching output
directory. Use an existing full resume checkpoint and `--resume` to continue.
Choose paths and Python executable on the deployment host; no host is hardcoded.
When CUDA_VISIBLE_DEVICES is a GPU UUID, occupancy is checked before every launch.
Use one supervisor per output directory. A lock prevents duplicate supervisors;
after an ungraceful supervisor crash, verify no owned worker is alive before
manually removing the stale lock. Never remove another running worker's lock.

The supervisor watches update, health, report and checkpoint files. It terminates
only its own stalled child, allows checkpointing, and resumes at most the configured
number of times. An unexpectedly SIGKILLed worker may resume its previous committed
checkpoint. Gradient failures, OOM, unknown Python errors, corrupt checkpoints and
occupied GPUs require attention. They do not cause blind restarts or configuration
changes. All attempts share one wall-time budget; explicit supervisor termination
stops its child and does not trigger recovery. `supervisor.json`,
`supervisor_events.jsonl` and `supervised_worker.log` retain operational evidence.

The Training view exposes supervision and numerical recovery through Training
health. The specialist prompts explain the same decision boundaries. Automatic
execution uses deterministic, tested rules; an LLM recommendation cannot bypass
them. It does not autonomously rewrite arbitrary training code or tune to test
labels. Recovery is operational evidence, not proof of convergence or higher AUC.

For acceptance, reproduce the failure, verify later native gradients and optimizer
updates, then complete full validation cycles on the original split. Retain the
pre-repair checkpoints and failed attempt reports. Report training probe AUC and
full validation AUC separately.

## Acceptance on ABIDE round 1

The full 271,483,811-parameter paper model resumed its model, optimizer, policy and
random states. Replaying step 35,419 reproduced a CE error log-probability of
-106.2756. Native classification gradients were zero, but the exact direction
probe verified every expected tensor and the positive classification coefficient.
At step 35,420, native classification and total gradients were nonzero again and
all five modules had sampled parameter updates. No diagnostic gradient was applied
to the optimizer.

Epoch 55 completed on all 217 validation subjects, followed by epoch 56. At step
35,860 the worker was running, the policy had 35,796 updates, and seven numerical
recoveries had been recorded with all consecutive counters back at zero. The
original control process retained its original PID and remained active. These
observations establish bounded execution recovery, not indefinite reliability.

| Evaluation epoch | Training probe AUC, 64 subjects | Validation AUC, 217 subjects |
| :--- | :--- | :--- |
| 54 | 0.8066 | 0.5419 |
| 55 | 0.7783 | 0.5197 |

Classification generalization did not improve in these two evaluations. Recovery
does not tune model weights against validation outcomes. Controlled full-model
fault injections at logit margins 90 and 160 passed while preserving the native
optimizer gradients. A double-precision stable-CE reference verified the normalized
direction. Detached paths, zero coefficients, persistent underflow and nonfinite
gradients remain rejected. Linux subprocess tests verified successful checkpoint
restart and termination at the retry limit. Code CI passed on Linux Python 3.10,
Linux Python 3.12 and Windows Python 3.12 for implementation commit `c93dbc5`.
