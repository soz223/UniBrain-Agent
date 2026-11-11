# UniBrain Agent

An agent controlled training workspace built on the original [UniBrain](https://github.com/Anonymous7852/UniBrain) implementation. Select modules, inspect generated pipeline code, verify gradients and watch the selected pipeline train.

## Start

The default pipeline is the full original UniBrain CUDA workflow. Use Python 3.10 with a CUDA GPU. Run these commands from the repository root on Windows:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-upstream.txt -r requirements.txt
.venv/Scripts/python.exe scripts/bootstrap.py --source reg_unibrain --samples
.venv/Scripts/python.exe -m unibrain_agent serve --open
```

The dashboard opens at http://127.0.0.1:8777. Subsequent Windows launches can use `Launch UniBrain Agent.cmd`. Linux uses `.venv/bin/python` or `bash launch.sh`. See [deployment instructions](docs/deployment.md) for Linux setup, external cohorts, GPU selection, remote access and CPU development.

The bootstrap script downloads pinned source files and the original public UniBrain examples. It does not download pretrained weights. Upstream licenses remain with their source files. A partial source fetch can be requested with `--source reg_unibrain`.

The original workflow does not require an agent login. Custom compositions use the provider selected in **AI settings**, initially Codex CLI with its existing login. `--backend rules` is available through the CLI for an explicit offline test. A failed model invocation never silently falls back to rules. Prompts contain the catalog and aggregate diagnostics; MRI arrays and subject records stay local.

**AI settings** uses provider cards for Codex CLI, Claude Code, OpenAI API, Anthropic API and a compatible endpoint. It shows the connected CLI account, email and plan when supplied by the official CLI, with refresh, switch and sign out controls. API keys entered in the page stay in backend memory; environment variables support persistent deployments.

The [GitHub Pages frontend](https://soz223.github.io/UniBrain-Agent/) runs independently of the Python backend. In **AI settings**, connect your workstation or HTTPS GPU server with its address and access token. All research, training, account and volume requests go to that server. The public frontend includes the application and method catalog; experiment data and credentials are loaded only after authentication. See [AI connections and deployment](docs/deployment.md) for setup.

Public pages start disconnected on every load, even if an older backend address or token exists in browser storage. AI access starts disconnected in every page. After signing in through your own backend CLI or saving your API credentials, click **Connect my AI account**. The backend issues a separate temporary AI session; it rejects model calls and account identity queries without that session. **Disconnect AI** closes page access while keeping the backend CLI login intact.

## Data build agent

The **Data** workspace is separate from the pipeline page. It shows the attached dataset, its split counts, one important warning and the input requirements. Source checks, conversion and AI discussion remain available on demand. The full source inventory and execution events remain in backend records. See the [data agent guide](docs/data-agent.md) for the input contract and conversion rules.

## Choose, validate, run

1. A fresh workspace starts with UniBrain extraction, registration, segmentation, atlas transfer, cosine graph and GCN. Optional additions start skipped. Previous experiments do not become the default selection. **New UniBrain run** restores this configuration.
2. **Prepare UniBrain** checks the pinned original source, CUDA environment and data paths. **Confirm & run** executes original `main.py` and `train.py` with the requested epoch budget. Full widths, five registration stages, original losses, augmentation and Adam are preserved. Original nearest label sampling, discrete edges and unused parcellation computations are also preserved.
3. To modify the pipeline, select alternative tools and click **Apply changes**. This explicitly enters the compact adapter profile, enables the displayed soft atlas and continuous graph connections, generates `pipeline.py`, and tests two optimizer updates on one sample. Alternatives such as FBNetGen are optional. Run `python scripts/bootstrap.py` to fetch their pinned code.
4. **Review Generated Code** is always visible and activates when code is generated. It shows the entry point, composition diff, actual registered implementations and source hashes, with a file download control. Original runs also expose the unchanged upstream model and training code. Original prerequisite checks do not claim complete gradient connectivity.
5. **Confirm & run** immediately opens the live monitor. It shows optimizer progress, recorded losses, actual module calls, agent activity when applicable, CPU usage and memory. The original worker also reports allocated GPU memory. Resource metrics exclude separate agent processes and cloud inference.
6. Inspect all 96 slices in **Model outputs**. The run panel contains resource use, model and agent calls, and training and validation metrics. The **Training** workspace shows measured learning signals and coordination. Completion does not imply improved performance.

The original and custom adapter profiles are explicitly separate. The CLI retains paired adapter comparisons for explicit comparison work; the dashboard runs only the selected pipeline. The sidebar contains pipeline navigation, a reset to UniBrain and runtime information.

The numeric training budget supports 2 through 1000. It means full epochs for the original trainer and optimizer steps for the compact adapter runner; the label changes accordingly. Defaults remain 2 original epochs or 5 adapter steps. Pending changes appear in amber, active work in blue with a spinner, validated builds in green and failures in red.

## Research conversations and training roles

Open **Research** in the sidebar to choose a stage, try an example question or continue a conversation. The researcher reads the project catalog and can retrieve live publication metadata and available abstracts from Crossref. Answers include source links, mechanisms, applicability and recommendations. **Use in pipeline** selects a registered adapter for review and validation; unregistered publications remain literature leads. Retrieval failures are displayed. A metadata result is not a new code audit.

Open **Training** for learning signals and the optional prompt role review. The review has extraction, registration, segmentation and classification specialists plus a coordinator. Saved-run reviews are advisory and never change training weights. Original runs without per-task gradients produce a request for measurements rather than invented conflict scores.

For an original UniBrain pipeline, enable **Learned gradient policy** before **Confirm & run**. Four task actors observe the individual losses and a task gradient Gram matrix measured on bounded parameter probes across five trainable modules. A centralized critic for each task learns from the next training update, and the resulting weights directly change the full model loss used for backpropagation. The run monitor displays live weights and policy update counts; the saved run contains every decision and a separate policy checkpoint. This is a TaskForce-inspired actor-critic prototype, not an exact reproduction of its published algorithm. The original model checkpoint incorporates the policy-controlled training and is used for validation and inference. The policy itself requires labeled losses, so it does not act on an unlabeled inference sample. Leave the option off to run the original loss coefficients. For a custom pipeline, **Use training roles** retains the prompt-based coordinator and its one-time bounded gradient-norm decision.

Prompts live in [`prompts`](prompts), and prepared builds pin their hashes. Conversations and execution transcripts stay in the ignored `assistant_sessions` directory, configurable through `UNIBRAIN_ASSISTANT`. The server uses the configured AI provider and does not send MRI arrays or subject records to the role prompts. Public metadata queries contain stage terms and question keywords. No agent silently falls back to the rules backend.

The [assistant verification record](catalog/assistant_verification.json) documents a live literature search, follow-up conversation, advisory role review and CUDA sample run with five actual model calls. Integration follows [Codex structured execution](https://learn.chatgpt.com/docs/non-interactive-mode) and [Crossref metadata retrieval](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/).

## Run the original UniBrain

The default dashboard and the following headless command use the full original CUDA workflow. A separate worker interpreter can be configured with `UNIBRAIN_ORIGINAL_PYTHON`.

```powershell
python -m pip install -r requirements-upstream.txt -r requirements.txt
python scripts/bootstrap.py --source reg_unibrain --samples
python -m unibrain_agent original --epochs 2
```

The verifier executes original `main.py`, `train.py`, `model.py` and `utils.py` without editing them. The wrapper sets the explicit epoch budget, routes external data and output paths, memory maps inputs, and records optimizer observations. Network widths, five registration stages, losses, augmentation, Adam settings and the requested sample splits remain as specified upstream. The original trainer loads the test split but evaluates the validation split. Each execution writes an isolated run directory.

Two epochs cover ten training updates, two validation passes and the upstream checkpoint saving branch. This verifies execution of the original sample workflow; it does not reproduce the paper's accuracy or its full 1000 epoch schedule. `requirements-upstream.txt` describes the tested environment, which uses PyTorch 2.5.1 rather than the repository's historical 2.1.2 environment. The original model requires substantially more memory than the compact profile.

The measured run in [catalog/upstream_verification.json](catalog/upstream_verification.json) completed all ten updates and both validation passes in 658.9 seconds on Windows with an RTX 3060 Ti. The model contained 305,812,248 parameters. Its 1,223,405,454 byte checkpoint had finite tensors and loaded strictly into the full original architecture. All four upstream code files remained unchanged. Source behavior findings, including the unused parcellation head and the trainer's mode handling after validation, are recorded separately in that audit.

The dashboard execution and Windows CUDA / Ubuntu CPU adapter checks are recorded in [catalog/portability_verification.json](catalog/portability_verification.json). The [Linux original-model audit](catalog/linux_original_verification.json) verifies smoke subsets from all five ABIDE and ADHD folds, plus two authenticated backend API runs, on an RTX 6000 Ada. All 48 optimizer updates passed; the full original architecture and four upstream source files were preserved. Full-cohort throughput, distributed training and paper-result reproduction remain unverified.

The agent composes registered, source pinned adapters. It does not freely rewrite arbitrary upstream code or implement unseen methods. Every generated build is retained separately, so selecting a new method cannot overwrite the original UniBrain source.

## Full architecture replacement experiment

`scripts/run_full_replacement.py` provides an isolated AI code generation,
gradient checking and repair loop for a full default SynthStrip extraction
network inside the full original UniBrain host. It retains five registration
stages and the original loss and optimizer settings. It does not use the
compact adapters described below. Validation is bounded to one or two updates.
See [the portable execution guide](docs/full-replacement.md) for paths, GPU
allocation, fault injection and the precise gradient contract. This command is
separate from the dashboard's existing compact composition action.

The [full replacement verification](catalog/full_replacement_verification.json)
records two actual Codex calls: generation and repair of an explicitly injected
hard mask fault. The repaired model completed two joint updates, inference and
strict checkpoint reload on Windows CUDA. Linux CUDA execution of this new
loop remains unverified; CPU portability and allocation checks are tested.

## Paper experiment programs

The [experiment suite](docs/experiment-suite.md) plans the paper matrix, checks
prerequisites, executes an explicitly selected job and summarizes retained
evidence. [configs/paper.example.json](configs/paper.example.json) contains
portable path placeholders and no machine or account credentials. Creating a
plan does not start training. Missing implementations and resources remain
visible as blocked jobs.

The [full comparison trainer](docs/replacement-training.md) supports original
controls, full SynthStrip, VoxelMorph, LapIRN and planned nnU-Net candidates,
independent and joint training, connection interventions, checkpoint resume,
validation selection and held-out test predictions. These are separate CLI
programs; the dashboard's custom composition action still uses its declared
compact profile. [Full composition](docs/full-composition.md) adds simultaneous
or sequential extraction and registration replacement with audited state
transfer.

The [integration benchmark](docs/integration-benchmark.md) runs controlled
Agent prompts and independent checks. It is not a reproduction of every named
external Agent system. `scripts/summarize_case_study.py` audits actual retained
repair attempts. `scripts/prepare_site_split.py`,
`scripts/prepare_nnunet.py`, `scripts/evaluate_masks.py` and
`scripts/compare_predictions.py` provide the corresponding data and analysis
programs. See [research readiness](docs/research-readiness.md) for measured
coverage, remaining implementation gaps and reproducibility boundaries.

## Available compact stages

| Stage | Research choices | Current adaptation |
| :--- | :--- | :--- |
| Denoising, optional | RED WGAN (2019, Medical Image Analysis) | Original generator at 16 cubed with tensor resizing |
| Bias correction, optional | DeepN4 (2024, Neuroinformatics) | Original core, bounded log bias and tensor correction |
| Brain extraction | SynthStrip (2022, NeuroImage), ERNet (2022, KDD) | Soft mask; ERNet occupies extraction and registration together |
| Intensity normalization, optional | deepNormalize (2021, Medical Image Analysis), Harmonizing Flows (2023, IPMI; 2025, Medical Image Analysis) | Single T1 normalization or tensor slice harmonization |
| Registration | VoxelMorph (2018, CVPR; 2019, IEEE TMI), LapIRN (2020, MICCAI), ERNet | Affine plus SVF or first scale LapIRN, with inverse grid composition |
| Tissue segmentation | nnU Net (2021, Nature Methods), deepNormalize | Four tissue classes; native auxiliary heads included in the loss |
| Brain parcellation | FastSurferCNN (2020, NeuroImage), HighRes3DNet (2017, IPMI) | Newly initialized 117 class AAL heads and supervised retraining |
| Network generation | FBNetGen (2022, MIDL) | Pair MLP with symmetric continuous adjacency |
| Classification | BrainNetCNN (2017, NeuroImage), Brain Network Transformer (2022, NeurIPS) | 116 node graph input; declared compatibility adapters |
| Cortical surface, optional branch | CorticalFlow (2021, NeurIPS) | Cataloged but blocked for these samples: mesh, physical affine and surface loss are missing |

Original UniBrain modules and differentiable connecting operators remain available. The table describes adapted model cores, not complete clinical tools. In particular, a newly trained AAL head does not map FastSurfer's pretrained 95 labels to the 116 AAL regions. The original atlas label vocabulary determines the new head's channel order.

The compact CPU profile preserves the 96 cubed external image interface. Individual heavy cores use declared lower resolutions or smaller widths. All models use random weights. Earlier isolated checks from the workbook are preserved separately from current composition checks.

## Training checks

Custom adapter preparation requires finite outputs, finite gradients for every active parameter tensor, a nonzero image input gradient, observable updates in every active module, and usable classification paths through extraction, registration, ROI features and graph weights. Zero derivatives and missing paths are reported separately. Frozen parameters are excluded from trainable coverage. Original preparation checks prerequisites; its actual training and finite checkpoint are audited after execution, preserving original gradient boundaries.

Spatial checks cover array shapes, affine determinant, probability mass, out of bounds sampling, grid folding and inverse cycle error. NPY samples do not provide physical spacing or orientation, so these checks do not establish physical coordinate alignment.

The tissue branch has its own supervised objective. Learned parcellation adds AAL cross entropy to that branch's loss. nnU Net auxiliary heads receive supervision. Dense registration adds velocity smoothness. Custom adapters use a prompt coordinator once before the second optimizer update. Original UniBrain can instead use the learned gradient policy at every training step. **Training → Training signals → Inspect policy** shows the observed gradient matrix, changing weights, rewards and actor-critic update count.

For paired CLI experiments, the post training gate requires successful spatial checks, classification gradients to registration and graph weights, updates in all active modules, validation reference loss at most 1.10 times baseline, and training time at most four times baseline. Passing trainability does not guarantee passing the quality or compute gate. This comparison gate is not applied to single pipeline dashboard runs.

The validation reference objective is fixed: extraction BCE, 0.1 times GCC registration loss, ground truth tissue cross entropy, and diagnosis cross entropy. The public training examples contain only one diagnosis class. These short runs demonstrate execution and trainability, not generalization or diagnostic accuracy.

## Data

The default data directory is `sources/reg_unibrain/dataset`. It contains `adhd_train_sample.npy`, `adhd_val_sample.npy`, and the three `template_img*_96.npy` files. Train and validation arrays use the original structured fields: `id`, `label`, `raw_img_np`, `raw_img_brain_mask_np`, `raw_img_gm_mask_np`, `raw_img_aal_mask_np`. Volume shape is 96 by 96 by 96, batch size is one. Test examples are not used.

Set `UNIBRAIN_CONFIG` to an external JSON file, or use the `UNIBRAIN_SOURCES`, `UNIBRAIN_DATA`, `UNIBRAIN_RUNS`, `UNIBRAIN_BUILDS` and device environment overrides. Relative configuration paths resolve against that file's directory. `config.example.json` contains portable defaults. An ignored `local.json` can provide machine specific interpreters. Reprepare builds after migration; stored builds bind to the verified code and runtime configuration.

## Reproduce and maintain

```powershell
python -m unittest discover -s tests -v
python scripts/check_adapters.py
python -m unibrain_agent prepare catalog/example_selection.json --backend rules
python -m unibrain_agent run --build-id BUILD_ID --backend rules --steps 2
```

The recorded sample check in `catalog/integration_checks.json` covers 16 configurations and 14 research methods. `catalog/example_run.json` records a Codex driven five method composition: it trained successfully, then restored the baseline because validation loss and compute exceeded the configured limits.

`check_adapters.py` executes each supported adapter in a complete sample pipeline. It requires downloaded sources and sample data and takes several minutes. GitHub Actions runs the smaller unit and API contract checks. A successful isolated or single adapter check does not certify arbitrary combinations; each selected combination is tested again.

`catalog/research.json` preserves the English workbook entries, source workbook hash and excluded candidates. `scripts/import_workbook.py` updates it from an authorized workbook without editing that workbook. `catalog/sources.json` pins the upstream commits and normalized file hashes.

Core implementation lives in `unibrain_agent`. The dashboard lives in `web`. Generated code, agent transcripts and preflight diagnostics stay under `builds`. Training reports and checkpoints stay under `runs`. These runtime artifacts, sample data, upstream sources and local paths are excluded from Git.

For a new method, add its pinned sources and catalog record, implement the tensor adapter and stage contract, test it in a complete sample pipeline, and document its adaptations. Keep a failed check visible; never substitute a different method while retaining its name.
