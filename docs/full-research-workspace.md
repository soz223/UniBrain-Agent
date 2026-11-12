# Full research workspace

The main dashboard's **Full model research** panel launches the actual
`scripts/run_research_cycle.py` program. It selects from the audited 14-method
library, generates and checks an adapter, retains failed attempts, and reuses
accepted code for full UniBrain joint training. The existing original baseline
API remains separate. The older custom stage editor is explicitly labeled a
legacy compact prototype.

Connect a backend and your AI account in AI settings. Enter a task and optionally
restrict its stage. The expandable settings accept backend source and prepared
fold paths, dataset and fold identity, GPU index, and nnU-Net plans. Source,
data and Python defaults come from the backend's portable runtime configuration.
The default data path follows the attached Data workspace dataset when present.
Verify this path before a run, particularly when switching from a smoke subset.

Codex and Claude Code must be installed and visible on the backend service
process's `PATH`, under the operating system account that runs the backend.
An installed desktop application alone does not establish this. Configure that
service environment and restart it after changing `PATH`; no platform-specific
application path is searched or embedded in the code.

**Check prerequisites** invokes the static cycle preflight. It performs no model
inference or training. **Run full research cycle** defaults to one acceptance
update and two joint-training updates. These are separate budgets. This panel
launches bounded execution checks, not full cohort performance experiments.
TaskForce coordination can be selected for the full host. The research cycle
uses the default replay batch of four, so the first policy update occurs at
training update five. Shorter runs collect experience and display a warming-up
state. Selecting coordination never increases the requested training budget.
The research cycle
always enables connected paths and training gradient verification; acceptance
and subsequent training retain their documented distinct connection contracts.

The panel polls the cycle report, independent checker output and training
`updates.jsonl`. It shows the selected method, active phase, attempts, observed
checks, update counts and loss trajectory. The latest event supplies current
TaskForce weights and policy counts even before the trainer flushes its epoch
report. Accepted code can be reviewed only after its hash is rechecked. Export
contains the displayed summary, not subject predictions, images or raw AI
transcripts. Token usage is reported when provided; no price is assumed.

The new authenticated routes are `GET /api/research-cycles`,
`POST /api/research-cycles`, `GET /api/research-cycles/{id}` and
`GET /api/research-cycles/{id}/code`. Launching requires an explicitly connected
AI browser session. A running cycle and a baseline or data build cannot launch
concurrently through this backend. Worker GPU checks still apply; this mutual
exclusion does not reserve a GPU or control other processes.

Private artifacts use the configured `research_cycles` directory, defaulting to
`builds/research_cycles`. Use `UNIBRAIN_RESEARCH_CYCLES` or the backend config to
relocate it. API credentials are passed only in the worker environment; they
are excluded from commands, job files and responses. Each launch has a unique
directory. A server restart marks an unfinished UI job interrupted instead of
silently resuming or overwriting it. Inspect retained worker state before retrying.

CPU tests cover the real CLI argument boundary, private environment credentials,
live policy records, adapter mutation, authentication and concurrent launch
rejection. Browser checks exercised the form, stage controls and TaskForce
selection. These checks did not launch an AI request or a GPU experiment.
