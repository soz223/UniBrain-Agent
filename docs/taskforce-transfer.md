# TaskForce transfer and training display

TaskForce is a cooperative reinforcement learning controller for multitask optimization. Its task agents are trainable actors with centralized critics. Prompt based literature and diagnostic roles remain a separate optional feature.

## Source audit

The CVPR 2026 paper, Figure 1, Sections 4.1 and 4.2, equations 5 through 13 and Algorithm 1 define the mechanism used here. The paper has been read from the supplied PDF. On November 12, 2026, checks of the CVF page, first author's publication page, lab publication page and public GitHub repository search did not identify a verified official TaskForce code repository. No unrelated repository was substituted. The current controller is our implementation of the mechanism.

Sources: [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/html/Choi_TaskForce_Cooperative_Multi-agent_Reinforcement_Learning_for_Multi-task_Optimization_CVPR_2026_paper.html), [first author](https://wonhyeok-choi.github.io/), [author GitHub](https://github.com/wonhyeok-choi), [lab publications](https://sunghoonim.github.io/publications.html).

## Mechanism

| Step | TaskForce | UniBrain implementation |
| :--- | :--- | :--- |
| Observe | Each actor receives its Gram matrix row and own task loss | Four tasks: extraction, registration, segmentation and classification |
| Act | One scalar logit per actor; joint softmax gives task weights | Four local actors; exploration noise before softmax |
| Update model | Weighted sum of task gradients | Weighted differentiable losses followed by the full UniBrain backward pass |
| Reward | Training loss improvement plus a penalty on the mixed gradient norm | Next training sample supplies the subsequent loss observation; validation is excluded |
| Learn policies | Off policy replay; one centralized critic per actor; target networks | Four critics see all observations and actions; replay, optimizers, targets and RNG state are checkpointed |
| Inference | Use the optimized model | The trained UniBrain checkpoint operates without querying the gradient policy |

The current controller uses parameter probes rather than every model gradient. It normalizes each Gram row by the mean diagonal, applies log1p to losses, clips observations and normalizes the gradient reward term. Effective coefficients are `T * softmax(logits) * base_weight`, retaining UniBrain's base coefficients. The critics receive normalized shares `w` rather than raw logits `a`. Replay learning runs when the next observation completes the preceding transition, before selecting the next action; Algorithm 1 places policy learning after the model update. These adaptation choices differ from the paper's stated formulation. The interface labels this implementation as TaskForce inspired.

## Training workspace

The view distinguishes forward computation, observed autograd paths and policy learning. Autograd paths indicate connectivity; the probe Gram matrix indicates gradient relationships on sampled parameters. Neither proves full model conflict removal or convergence. Arrow motion illustrates message or computation flow and does not encode the direction of a gradient in anatomical space.

The controller records task losses, Gram and cosine matrices, policy shares, applied coefficients, shared reward components, replay size and optimizer update counters. New runs also record normalized actor inputs, exploratory logits and critic TD losses. Existing records retain N/A for fields they never logged. A replay slider selects actual recorded updates. Policy shares and effective coefficients have separate chart modes.

For the paper configuration, the display correctly routes raw MRI to the tissue segmentation network and labels inverse affine atlas transfer as trilinear. Earlier logs used a generic nearest transfer label; the read only monitor corrects that display metadata using the explicitly recorded paper profile. The underlying logs and checkpoints remain untouched.

## Register training runs

In the backend runtime JSON, register directories under `training_runs`:

```json
{
  "training_runs": {
    "abide_round1": {"path": "runs/abide_round1", "label": "ABIDE fold 1"},
    "adhd_round1": {"path": "runs/adhd_round1", "label": "ADHD fold 1"}
  }
}
```

Paths are resolved relative to the runtime configuration. Restart only the web backend to apply registration changes. The monitor reads the last 128 updates, bounded to 4 MiB, and epoch validation summaries. It loads no checkpoints or volumes, launches no workers and exposes only registered run IDs through authenticated API routes. Partial trailing JSON lines are ignored until complete. The reported run status and the age of the last file update remain separate because file timestamps cannot prove a process is alive.

The browser polls a selected registered run every ten seconds while Training is open. Native workspace runs and full research cycle updates use their existing telemetry events. GitHub Pages publishes the interface only; each browser must explicitly connect its own backend to obtain live training records.

For a separate monitoring workstation, `scripts/sync_training_monitor.py` can mirror aggregate records over an existing SSH login. Supply a private JSON mapping of safe run IDs to server directories with `--runs`, a destination with `--output`, and an optional polling interval with `--interval`. Register the resulting local directories in `training_runs`. Original file timestamps survive mirroring, so a disconnected mirror becomes visibly stale. The script performs only remote file reads. Server addresses, directories and credentials are deployment configuration and are not embedded in application code.
