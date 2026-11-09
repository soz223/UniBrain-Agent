# UniBrain Agent

An agent controlled training workspace built on the original [UniBrain](https://github.com/Anonymous7852/UniBrain) implementation. Select modules, inspect generated pipeline code, verify gradients and train a measured baseline and candidate comparison.

## Start

Python 3.12 is the tested runtime. Run these commands from the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/bootstrap.py --samples
.venv/Scripts/python.exe -m unibrain_agent serve --open
```

The dashboard opens at http://127.0.0.1:8768. On Windows, subsequent launches can use `Launch UniBrain Agent.cmd`. On Linux or macOS, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

The bootstrap script downloads pinned source files and the original public UniBrain examples. It does not download pretrained weights. Upstream licenses remain with their source files. A partial source fetch can be requested with `--source reg_unibrain`.

Codex CLI with an existing login provides the composition, coordination and review agents. Select **Rules demo** for an explicit offline workflow. A failed Codex invocation never silently falls back to rules. Prompts contain the catalog and aggregate diagnostics; MRI arrays and subject records stay local.

## Choose, validate, run

1. Open **Pipeline builder** and choose a tool for a stage. The selector includes all 15 works from the supplied research workbook, with publication, source, prior evidence and transfer limits.
2. **Use tool and validate** fixes that choice, asks the agent to review the composition, generates `pipeline.py`, and checks two optimizer steps on one real public sample.
3. **Review code** shows the generated file, the change from baseline, adapter details and measured parameter updates.
4. **Build and run** runs the validated file. The executor checks that its code and implementation have not changed since validation.
5. Compare the resulting images, gradient matrix, training curves, graph, validation results, decisions and execution log. A failed acceptance gate restores the baseline checkpoint.

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

The tissue branch has its own supervised objective. Learned parcellation adds AAL cross entropy to that branch's loss. nnU Net auxiliary heads receive supervision. Dense registration adds velocity smoothness. The coordinator may adjust the four task weights with bounded gradient norm balancing; this is not TaskForce reinforcement learning.

The post training gate requires successful spatial checks, classification gradients to registration and graph weights, updates in all active modules, validation reference loss at most 1.10 times baseline, and training time at most four times baseline. Passing trainability does not guarantee passing the quality or compute gate.

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

`check_adapters.py` executes each supported adapter in a complete sample pipeline. It requires downloaded sources and sample data and takes several minutes. GitHub Actions runs the smaller unit and API contract checks. A successful isolated or single adapter check does not certify arbitrary combinations; each selected combination is tested again.

`catalog/research.json` preserves the English workbook entries, source workbook hash and excluded candidates. `scripts/import_workbook.py` updates it from an authorized workbook without editing that workbook. `catalog/sources.json` pins the upstream commits and normalized file hashes.

Core implementation lives in `unibrain_agent`. The dashboard lives in `web`. Generated code, agent transcripts and preflight diagnostics stay under `builds`. Training reports and checkpoints stay under `runs`. These runtime artifacts, sample data, upstream sources and local paths are excluded from Git.

For a new method, add its pinned sources and catalog record, implement the tensor adapter and stage contract, test it in a complete sample pipeline, and document its adaptations. Keep a failed check visible; never substitute a different method while retaining its name.
