# Research cycle

The cycle connects a real provider's candidate selection to generated code,
independent validation, repair and subsequent full-model training. It selects
from the audited 14-candidate library. It does not claim a live literature
search or reproduction of an unrestricted external coding agent.

```sh
python scripts/run_research_cycle.py --source sources/reg_unibrain --data data/fold1 --output private_runs/new_cycle --provider codex --task "Learn a task-specific structural brain graph" --stage network_generation --acceptance-steps 1 --train-steps 2
```

Add `--preflight` to audit local inputs, sources and provider availability with
no provider or GPU calls. Add `--selection-only` to make one real provider
selection and stop before code generation, acceptance or training. Each mode
requires a new output directory. Static readiness does not prove login or
endpoint access; a successful real selection supplies that separate evidence.

Use a prepared UniBrain fold directory and the configured local provider
account. Source paths, data paths and GPU index are explicit and portable.
Original source and data directories are preserved. Source availability, hashes
and split separation are audited before selection. Images and subject records
are not sent to the provider. The selection prompt receives the task, schema,
declared method interfaces, adaptations and source evidence hashes. Select an
explicit model with `--model` for reproducible comparisons.
Dataset-specific nnU-Net plans contribute only selected architecture fields.
Subject lists, dataset names and arbitrary plan metadata are excluded from
provider context. Imaging arrays and subject-table files are rejected as source
evidence. The actual plan remains a local evaluator dependency.

The selected method enters the existing `workflow` integration procedure. The
provider screens applicability, generates the constrained adapter, receives
measured checks and may revise it. Final acceptance always uses the complete
independent evaluator, including hidden checks in ablation experiments. Failed
or blocked validation never starts training. The cycle allows abstention.

After acceptance, the exact checked code is copied to `accepted_adapter.py`.
Its hash must match a passed independent evaluation. The cycle passes this
file through `--adapter` to the real joint trainer. Completion requires the
training report to confirm that this exact adapter was active and that the
declared optimizer budget was respected. Generated code is not replaced with
a built-in implementation after validation.
Training also verifies the configured source/data paths and requires the
accepted training-array hash to match its own audit. Missing acceptance hashes
block training, including for SynthStrip. A zero exit status
alone does not satisfy the training readiness check.

Joint training always enables `--connect-paths` and `--verify-gradients`.
It records the continuous atlas/graph connection as an explicit integration
change and verifies diagnosis gradients into the selected candidates, with
the original auxiliary nnU-Net tissue head retaining its documented exception.
This is a modified training protocol. SynthStrip acceptance still checks its
declared registration-to-extraction path in the original host; the subsequent
connected training phase adds the stricter diagnosis-path requirement. The
two phases are not represented as having identical host connections.

`--coordination` accepts `none` or `taskforce`, defaulting to `none`.
The cycle checks the reported mode and requires an observed policy decision
when TaskForce coordination is requested. `--checkpoint-mode` defaults to
`weights` for short acceptance runs; select `resume` when optimizer and RNG
state must be retained. The actual checkpoint mode, verified update count and
unchanged accepted code hash must match the requested protocol.

Acceptance and training have separate update budgets. Defaults allow one
acceptance update and two training updates. The selection call is separate
from the integration procedure's `--max-calls` budget. Any observed acceptance
update closes further training-based repair attempts. Training initializes a
fresh model using the accepted code and recorded seed; it does not resume the
acceptance optimizer or silently reuse its checkpoint. Increase `--train-steps`,
`--epochs` and `--eval-limit` explicitly for larger runs. Default evaluation is
limited to two cases and is diagnostic, not a cohort accuracy estimate.

`report.json` records selection rationale, actual provider usage, source audit,
validation and repair trace, accepted code identity, training command, status
and separate observed update counts. `source_audit.json`, `selection/`,
`integration/`, the accepted code, training logs and `training/report.json`
retain the underlying evidence. Dollar cost remains unknown unless priced
externally. Provider failures, unavailable dependencies, inconsistent reports
and occupied GPUs remain visible blocked outcomes.

The orchestration tests mock providers and workers to verify control flow,
budget enforcement and exact adapter path/hash transmission. They do not
constitute an executed AI trial or a full GPU acceptance experiment.
The current dashboard does not yet launch this new CLI cycle. Its provider
calls use the shared constrained prompting harness, not a complete standalone
Codex, Claude Code or MLE-STAR system reproduction.
