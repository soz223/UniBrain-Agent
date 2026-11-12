# External baseline readiness

Audit date: 2026-11-11. This is a source and interface audit, not execution
evidence. No external system was installed, authenticated or run for this audit.
The [catalog](../catalog/external_baselines.json) records candidate source pins;
it does not register an experiment executor.

`executable_available` means the official program can be obtained. It does not
mean its dependencies or its UniBrain task adapter have been validated here.
`adapter_missing` means this repository still needs a real system runner and
result adapter. `upstream_incomplete` means the inspected public resources do
not provide the complete named system.

## Official programs

| System and suite labels | Source snapshot | Task scope and prerequisites | Remaining work |
| :--- | :--- | :--- | :--- |
| Codex: `codex`, `codex_unibrain` | [Official source, c40729e](https://github.com/openai/codex/tree/c40729ea5cf8e947481b130c59807273eda2f8e9), tag `rust-v0.167.0-alpha.3.2`; local CLI reported the matching version | General code editing and execution. Needs the CLI, valid provider access, an isolated task workspace and the UniBrain environment. | `adapter_missing`: run the official tool loop with workspace edits enabled, collect JSONL events and evaluate its actual artifacts. |
| Claude Code: `claude`, `claude_unibrain` | [Official release v2.1.322](https://github.com/anthropics/claude-code/releases/tag/v2.1.322), [release repository commit 6e1e44f](https://github.com/anthropics/claude-code/tree/6e1e44f6433bfa17d198e932bff53373aa229eea) | General code editing and execution. Needs the official CLI, valid provider access and the same isolated task environment. The release repository is not evidence that the full runtime source is available. | `adapter_missing`: enable the real read, edit and execution tools; collect tool events, budget outcomes and produced artifacts. |
| OpenClaw: `openclaw_unibrain` | [Official source, 1cbfc6e](https://github.com/openclaw/openclaw/tree/1cbfc6eeba6df6de89312040be442388692d977c), package `2026.11.3` | Configurable coding agent runtime. Source requires Node `>=24.16.0 <25` or `>=26.1.0`. Needs a pinned tool/runtime configuration, provider access and UniBrain dependencies. | `adapter_missing`: invoke its own `agent exec` loop in the task workspace and preserve transcripts and results. Do not silently select a different harness or fallback model. |
| MLE STAR: `mle_star` | [Official sample, 9bf35ee](https://github.com/google/adk-recipes/tree/9bf35ee20cc6ee487be27fa9f0017c558559742f/python/agents/machine-learning-engineering), package `0.1.0` | Searches for models, executes code, refines influential components and ensembles solutions. Needs Python 3.12+, ADK, Gemini API or Vertex access, search and the model execution environment. | `adapter_missing`: supply `task_description.txt`, isolated data, the objective and submission contract; retain the official initialization, refinement, ensemble and submission agents. |
| NEXUS: `nexus` | [Author source, c2e15d4](https://github.com/LearningKeqi/Virtual-Neuroscientist-NEXUS/tree/c2e15d4983489fd89d604fdbbefbbf1726958068); no tags found | Autonomous inspection, preprocessing, QC and downstream predictive modeling. Needs LangGraph/LangChain, an OpenAI-compatible configuration and the selected FSL, AFNI, ANTs, FreeSurfer, fMRIPrep or MRIQC tools. | `adapter_missing`: adapt task entry, data and result formats. Its Python execution tool hardcodes SLURM, partition `h200day` and a workspace path; infrastructure changes must be explicit. |
| NEURA: `neura`, `neura_unibrain` | [Author source, c349dde](https://github.com/NeuroScienceLab/NEURA/tree/c349dde3e9d713181ecb7b734c8bb42fe9646c37); no tags found | Research planning, registered imaging tools, QC and reports. Needs Python 3.9+, LangGraph and its configured SiliconFlow endpoint. Selected tools additionally need MATLAB/SPM/DPABI, DSI Studio or FSL/FreeSurfer; documented deployment uses Windows and WSL2. | `adapter_missing`: invoke the real research graph and export results. A UniBrain training tool can use `BaseTool` and `ToolRegistry`; that extension must be reported as our adaptation. |
| NeuroPilot: `neuropilot` | [Paper v2](https://arxiv.org/html/2608.07541v2); [author DICOM skill, 7e98805](https://github.com/acmlab/dcm2bids-skill/tree/7e9880588e203404d08652aee576a2c68ee71f87), declared `1.0.0`; [CyberNeuro, 0709f13](https://github.com/acmlab/CyberNeuro/tree/0709f137b5097aea8ea2ca9877c783926bf2404b) | Paper describes DICOM standardization, preprocessing and QC skills. The public conversion skill needs an agent runtime, Python 3.8+, dcm2bids, dcm2niix and bids-validator. | `upstream_incomplete`: the inspected CyberNeuro tree contains a PDF manual; the two other paper skills were not found in the author's public repositories or exact-name repository searches. The conversion skill alone is not the full baseline. |

Google renamed `adk-samples` to `adk-recipes` and removed the old samples in
[63bfb33](https://github.com/google/adk-recipes/commit/63bfb33c4df5d00b6ef9e2e0e864fbcb01e37645).
MLE STAR remains obtainable at the pinned historical commit. Its inspected
configuration disables the optional data leakage and data usage checkers by
default; record their actual settings. The public sample and its defaults do
not establish reproduction of the paper's reported results.

Official automation interfaces are documented for
[Codex](https://learn.chatgpt.com/docs/non-interactive-mode),
[Claude Code](https://code.claude.com/docs/en/headless) and
[OpenClaw](https://docs.openclaw.ai/cli/agent). NEXUS exposes script generation
and execution in its [downstream tools](https://github.com/LearningKeqi/Virtual-Neuroscientist-NEXUS/blob/c2e15d4983489fd89d604fdbbefbbf1726958068/source/tool_lib/downstream_analysis.py).
NEURA exposes extension interfaces in its [registry](https://github.com/NeuroScienceLab/NEURA/blob/c349dde3e9d713181ecb7b734c8bb42fe9646c37/src/tools/registry.py).

## Comparison contract

The suite currently lists both `codex` and `codex_unibrain`, both `claude` and
`claude_unibrain`, and both `neura` and `neura_unibrain`. Their distinct starting
artifacts and permitted changes are not yet specified. A possible distinction
is building a diagnostic pipeline from scratch versus modifying a supplied
UniBrain pipeline. That remains a protocol decision, not an implemented split.

Current `providers.invoke` calls are constrained code-generation calls:
Codex uses a read-only sandbox, and Claude's tools are disabled. Shared prompts
and our repair loop measure controlled prompting variants. A named full-system
comparison must run that system's own tool loop and save its generated code,
commands, feedback, failures and final model. A catalog entry or a successful
provider call cannot establish this.

Every runner needs a fixed task package, source and model versions, explicit
wall-time and training budgets, process cleanup, and an independent result
adapter. The final checker must assess actual saved artifacts. Record any
feedback exposed during development and any added UniBrain tool; do not supply
our entire selection and repair policy and call it an external system.

Only train and validation data should be accessible to an editing agent.
The current master array includes test labels, so giving the agent that file
would violate the intended held-out protocol. Create an isolated task export
with the existing splits, and let a separate evaluator access test labels.
Preserve the original arrays and record the export mapping privately.

The structured 96 cubed arrays do not preserve native orientation, voxel size
or NIfTI affine metadata. A unit-affine export does not reconstruct that
geometry. Native preprocessing comparisons require the original geometry and
modality metadata. Array-based UniBrain adaptation can use the existing tensor
contract, with that scope stated explicitly.

## Minimal executable subset

Implement the official Codex plus UniBrain runner first, then the pinned MLE
STAR sample and its task adapter. Add Claude Code or OpenClaw through the same
isolated task and evaluation contract once their runtime access is available.
NEXUS and NEURA need separate imaging/runtime integration. Keep full NeuroPilot
blocked until the remaining implementation can be obtained and identified.
These are implementation priorities, not completed experiments or exclusions
from the paper plan.
