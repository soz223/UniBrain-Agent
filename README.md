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

The dashboard opens at http://127.0.0.1:8776. Subsequent Windows launches can use `Launch UniBrain Agent.cmd`. Linux uses `.venv/bin/python` or `bash launch.sh`. See [deployment instructions](docs/deployment.md) for Linux setup, external cohorts, GPU selection, remote access and CPU development.

The bootstrap script downloads pinned source files and the original public UniBrain examples. It does not download pretrained weights. Upstream licenses remain with their source files. A partial source fetch can be requested with `--source reg_unibrain`.

The original workflow does not require an agent login. Custom compositions use the provider selected in **AI settings**, initially Codex CLI with its existing login. `--backend rules` is available through the CLI for an explicit offline test. A failed model invocation never silently falls back to rules. Prompts contain the catalog and aggregate diagnostics; MRI arrays and subject records stay local.

**AI settings** uses provider cards for Codex CLI, Claude Code, OpenAI API, Anthropic API and a compatible endpoint. It shows the connected CLI account, email and plan when supplied by the official CLI, with refresh, switch and sign out controls. API keys entered in the page stay in backend memory; environment variables support persistent deployments.

The [GitHub Pages frontend](https://soz223.github.io/UniBrain-Agent/) runs independently of the Python backend. In **AI settings**, connect your workstation or HTTPS GPU server with its address and access token. All research, training, account and volume requests go to that server. The public frontend includes the application and method catalog; experiment data and credentials are loaded only after authentication. See [AI connections and deployment](docs/deployment.md) for setup.

## Choose, validate, run

1. A fresh workspace starts with UniBrain extraction, registration, segmentation, atlas transfer, cosine graph and GCN. Optional additions start skipped. Previous experiments do not become the default selection. **New UniBrain run** restores this configuration.
2. **Prepare UniBrain** checks the pinned original source, CUDA environment and data paths. **Confirm & run** executes original `main.py` and `train.py` with the requested epoch budget. Full widths, five registration stages, original losses, augmentation and Adam are preserved. Original nearest label sampling, discrete edges and unused parcellation computations are also preserved.
3. To modify the pipeline, select alternative tools and click **Apply changes**. This explicitly enters the compact adapter profile, enables the displayed soft atlas and continuous graph connections, generates `pipeline.py`, and tests two optimizer updates on one sample. Alternatives such as FBNetGen are optional. Run `python scripts/bootstrap.py` to fetch their pinned code.
4. **Review code** shows the generated entry point, adapter changes and measured checks. Original prerequisite checks do not claim complete gradient connectivity.
5. **Confirm & run** immediately opens the live monitor. It shows optimizer progress, recorded losses, actual module calls, agent activity when applicable, CPU usage and memory. The original worker also reports allocated GPU memory. Resource metrics exclude separate agent processes and cloud inference.
6. Inspect all 96 slices. **What is learning?** shows measured loss connectivity for adapters, or observed original parameter updates for native runs. Expand calls, the notebook, validation or source checks as needed. Completion does not imply improved performance.

The original and custom adapter profiles are explicitly separate. The CLI retains paired adapter comparisons for explicit comparison work; the dashboard runs only the selected pipeline. The sidebar contains pipeline navigation, a reset to UniBrain and runtime information.

## Research conversations and training roles

Open **Research** in the sidebar to choose a stage, try an example question or continue a conversation. The researcher reads the project catalog and can retrieve live publication metadata and available abstracts from Crossref. Answers include source links, mechanisms, applicability and recommendations. **Use in pipeline** selects a registered adapter for review and validation; unregistered publications remain literature leads. Retrieval failures are displayed. A metadata result is not a new code audit.

Open **Training roles** for extraction, registration, segmentation and classification specialists plus a coordinator. Each role has **View prompt** and **Run role** buttons; **Run all roles** performs a sequential review. The coordinator receives the specialists' outputs and the measured context. These saved-run reviews are advisory and never change training weights. Original runs without per-task gradients produce a request for measurements rather than invented conflict scores.

For a custom pipeline, enable **Use training roles** before **Confirm & run**. The executor measures task gradients before optimizer update 2, calls the four specialists and coordinator, then either retains the weights or applies bounded gradient-norm balancing. The applied decision, before/after weights and role calls appear in the run record. Original reproduction keeps this switch disabled. This is prompt based coordination; a TaskForce reinforcement learning policy, gradient projection and performance improvement are not claimed.

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

The new dashboard execution and Windows CUDA / Ubuntu CPU adapter checks are recorded in [catalog/portability_verification.json](catalog/portability_verification.json). Linux full original CUDA training, distributed training and large cohort throughput have not been verified.

The agent composes registered, source pinned adapters. It does not freely rewrite arbitrary upstream code or implement unseen methods. Every generated build is retained separately, so selecting a new method cannot overwrite the original UniBrain source.

## Available stages

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

The tissue branch has its own supervised objective. Learned parcellation adds AAL cross entropy to that branch's loss. nnU Net auxiliary heads receive supervision. Dense registration adds velocity smoothness. The coordinator makes one decision before the second optimizer update. It sees gradient norms and shared parameter cosines, then keeps the four loss weights or applies bounded gradient norm balancing. **What is learning? → Agent training decision → Inspect decision** shows the rationale and weights before and after. This is not TaskForce reinforcement learning or a learned multiagent policy.

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
