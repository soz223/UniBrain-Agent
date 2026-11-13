# TaskForce transfer and training display

TaskForce is a cooperative reinforcement learning controller for multitask optimization. Its task agents are trainable actors with centralized critics. Prompt based literature and diagnostic roles remain a separate optional feature.

## Source audit

The CVPR 2026 paper, Figure 1, Sections 4.1 and 4.2, equations 5 through 13 and Algorithm 1 define the mechanism used here. The supplied paper and official supplement have been read. Appendix A identifies five core components; Appendix B provides the policy configuration. On November 12, 2026, checks of the CVF page, first author's publication page, lab publication page and public GitHub repository search did not identify a verified official TaskForce code repository. The controller is our implementation of the published method.

Sources: [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/html/Choi_TaskForce_Cooperative_Multi-agent_Reinforcement_Learning_for_Multi-task_Optimization_CVPR_2026_paper.html), [first author](https://wonhyeok-choi.github.io/), [author GitHub](https://github.com/wonhyeok-choi), [lab publications](https://sunghoonim.github.io/publications.html).

Official [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Choi_TaskForce_Cooperative_Multi-agent_CVPR_2026_supplemental.pdf). The complete component audit and implementation choices are in [TaskForce core audit](taskforce-core-audit.md).

## Mechanism

| Step | TaskForce | UniBrain implementation |
| :--- | :--- | :--- |
| Observe | Each actor receives its Gram matrix row and own task loss | Four tasks: extraction, registration, segmentation and classification |
| Act | One scalar logit per actor; joint softmax gives task weights | Four local actors; exploration noise before softmax |
| Update model | Weighted sum of task gradients | Weighted differentiable losses followed by the full UniBrain backward pass |
| Reward | Training loss improvement plus a penalty on the mixed gradient norm | Next training sample supplies the subsequent loss observation; validation is excluded |
| Learn policies | Off policy replay; one centralized critic per actor; target networks | Four critics see all observations and actions; replay, optimizers, targets and RNG state are checkpointed |
| Inference | Use the optimized model | The trained UniBrain checkpoint operates without querying the gradient policy |

New runs default to `--taskforce-profile paper`. The controller observes gradients over every trainable model parameter, produces softmax task shares, stores raw action logits in replay, and learns after the model optimizer step. L2 normalization gives an applied loss coefficient `w / gradient_norm`, so backward equals the convex combination of normalized task gradients. Original module definitions and task losses remain unchanged. UniBrain's original coefficients are recorded separately and are not multiplied into the policy shares.

The earlier `legacy` profile remains available for explicit historical comparisons. It samples parameter tensors, scales shares by `T * base_weight`, stores normalized shares in replay, and learns before the next action. Active legacy runs retain their immutable code and checkpoints. Loss, gradient and reward normalization are also present in the original TaskForce supplement; normalization itself is not a distinction introduced by UniBrain Agent.

## Training workspace

The view distinguishes forward computation, observed autograd paths and policy learning. Each record determines its profile label, gradient scope, replay action and coefficient formula. Legacy records retain their sampled-gradient interpretation. Neither a Gram matrix nor an autograd path proves conflict removal or convergence. Arrow motion illustrates computation flow and does not encode the direction of a gradient in anatomical space.

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
