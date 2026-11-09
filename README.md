# UniBrain Agent

An agent controlled training workspace built on the original [UniBrain](https://github.com/Anonymous7852/UniBrain) implementation. Select modules, inspect generated pipeline code, verify gradients and watch the selected pipeline train.

## Start

Python 3.12 is the tested runtime. Run these commands from the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/bootstrap.py --samples
.venv/Scripts/python.exe -m unibrain_agent serve --open
```

The dashboard opens at http://127.0.0.1:8770. On Windows, subsequent launches can use `Launch UniBrain Agent.cmd`. On Linux or macOS, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

The bootstrap script downloads pinned source files and the original public UniBrain examples. It does not download pretrained weights. Upstream licenses remain with their source files. A partial source fetch can be requested with `--source reg_unibrain`.

Codex CLI with an existing login provides the composition, coordination and review agents. The dashboard uses Codex; `--backend rules` is available through the CLI for an explicit offline test. A failed Codex invocation never silently falls back to rules. Prompts contain the catalog and aggregate diagnostics; MRI arrays and subject records stay local.

## Choose, validate, run

1. Click an ordered pipeline step and search for a tool. Selected tools are marked; optional steps offer **Skip**. Tissue segmentation and cortical surface appear separately as branches. The catalog retains all 15 works from the research workbook; tools requiring missing inputs are disabled.
2. Select multiple tools, then click **Apply changes**. The agent reviews the composition, generates `pipeline.py`, and checks two optimizer steps on one real public sample. Build activity shows the actual phase, elapsed time, forward passes, backpropagation and audit events.
3. **Review code** shows the generated file, the change from baseline, adapter details and measured parameter updates.
4. **Confirm & run** immediately opens the live run monitor and executes only the selected pipeline. The executor checks that its code and implementation have not changed since validation. The monitor shows completed optimizer steps, a loss curve, actual module calls, agent activity, worker CPU usage and resident memory. CPU usage is normalized to total machine capacity; resources exclude the separate Codex process and cloud inference.
5. Inspect all 96 slices with the slider, arrow keys or adjacent slice buttons. **What is learning?** explains which modules receive gradients from each loss. Expand **Model & agent calls**, **Experiment notebook**, **Validation results** or **Model, data & checks** for additional measurements. The notebook preserves decisions and execution events. A completed run saves its checkpoint; it does not imply acceptance or improved performance.

The dashboard uses compact adapted model cores. Running this profile does not establish reproduction of the unchanged upstream training entry point or its reported results. The CLI retains the paired baseline and candidate experiment for explicit comparison work; the dashboard does not run or display that comparison.

## Run the original UniBrain

Use a separate Python 3.10 CUDA environment for the full upstream model. The CPU adapters in the dashboard are a separate profile.

```powershell
python -m pip install -r requirements-upstream.txt
git clone https://github.com/Anonymous7852/UniBrain.git builds/original/source
git -C builds/original/source checkout f9f8d906b1725046d532f862a28876482aca2cfa
python scripts/run_upstream.py builds/original/source --epochs 2
```

The verifier executes the original `main.py`, `train.py`, `model.py` and `utils.py` without editing them. It changes only the epoch budget passed to the original trainer, records optimizer observations, and saves file hashes and diagnostics next to the checkout. Network widths, five registration stages, losses, augmentation, Adam settings and all sample splits remain as specified upstream. The original trainer loads the test split but evaluates the validation split.

Two epochs cover ten training updates, two validation passes and the upstream checkpoint saving branch. This verifies execution of the original sample workflow; it does not reproduce the paper's accuracy or its full 1000 epoch schedule. `requirements-upstream.txt` describes the tested environment, which uses PyTorch 2.5.1 rather than the repository's historical 2.1.2 environment. The original model requires substantially more memory than the compact profile.

The measured run in [catalog/upstream_verification.json](catalog/upstream_verification.json) completed all ten updates and both validation passes in 658.9 seconds on Windows with an RTX 3060 Ti. The model contained 305,812,248 parameters. Its 1,223,405,454 byte checkpoint had finite tensors and loaded strictly into the full original architecture. All four upstream code files remained unchanged. Source behavior findings, including the unused parcellation head and the trainer's mode handling after validation, are recorded separately in that audit.

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

Preparation requires finite outputs, finite gradients for every active parameter tensor, a nonzero image input gradient, observable updates in every active module, and usable classification paths through extraction, registration, ROI features and graph weights. Zero derivatives and missing paths are reported separately. Frozen parameters are excluded from trainable coverage.

Spatial checks cover array shapes, affine determinant, probability mass, out of bounds sampling, grid folding and inverse cycle error. NPY samples do not provide physical spacing or orientation, so these checks do not establish physical coordinate alignment.

The tissue branch has its own supervised objective. Learned parcellation adds AAL cross entropy to that branch's loss. nnU Net auxiliary heads receive supervision. Dense registration adds velocity smoothness. The coordinator makes one decision before the second optimizer update. It sees gradient norms and shared parameter cosines, then keeps the four loss weights or applies bounded gradient norm balancing. **What is learning? → Agent training decision → Inspect decision** shows the rationale and weights before and after. This is not TaskForce reinforcement learning or a learned multiagent policy.

For paired CLI experiments, the post training gate requires successful spatial checks, classification gradients to registration and graph weights, updates in all active modules, validation reference loss at most 1.10 times baseline, and training time at most four times baseline. Passing trainability does not guarantee passing the quality or compute gate. This comparison gate is not applied to single pipeline dashboard runs.

The validation reference objective is fixed: extraction BCE, 0.1 times GCC registration loss, ground truth tissue cross entropy, and diagnosis cross entropy. The public training examples contain only one diagnosis class. These short runs demonstrate execution and trainability, not generalization or diagnostic accuracy.

## Data

The default data directory is `sources/reg_unibrain/dataset`. It contains `adhd_train_sample.npy`, `adhd_val_sample.npy`, and the three `template_img*_96.npy` files. Train and validation arrays use the original structured fields: `id`, `label`, `raw_img_np`, `raw_img_brain_mask_np`, `raw_img_gm_mask_np`, `raw_img_aal_mask_np`. Volume shape is 96 by 96 by 96, batch size is one. Test examples are not used.

Set `UNIBRAIN_SOURCES` and `UNIBRAIN_RUNS` to reuse local source and run directories. An ignored `local.json` can specify `sources`, `runs` and the Windows launcher `python` path. None of these local settings are required by a fresh checkout.

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
