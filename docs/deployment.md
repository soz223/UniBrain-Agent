# Deployment

The repository contains the application, pipeline compiler, execution workers, audits, UI, dependency files and source manifest. Upstream code and example data are fetched by `scripts/bootstrap.py` at pinned revisions. Data, checkpoints, credentials and local configuration are excluded from Git.

## Linux with a CUDA GPU

Use Python 3.10 for the tested original UniBrain environment. Install the NVIDIA driver on the host before creating the environment. The pinned PyTorch wheel supplies CUDA 12.1 libraries.

```bash
git clone https://github.com/soz223/UniBrain-Agent.git
cd UniBrain-Agent
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-upstream.txt -r requirements.txt
python scripts/bootstrap.py --source reg_unibrain --samples
python -m unibrain_agent serve --port 8773
```

Open `http://127.0.0.1:8773`. The initial pipeline uses the full original UniBrain: original extraction, five registration stages, segmentation, nearest atlas transfer, cosine graph with original edge perturbation, and GCN. Optional stages start skipped. The original architecture, loss equations, augmentation and optimizer are preserved. The default two epoch budget exercises training, validation and checkpoint saving; the upstream paper schedule is 1000 epochs.

The original CUDA workflow runs in a separate process. It never silently uses the compact adapters when CUDA or dependencies are missing. The original trainer's existing behavior, including its mode handling after validation, is retained.

Fetch all alternative sources before using the module chooser:

```bash
python scripts/bootstrap.py
```

Selecting a replacement enters the compact adapter profile. Soft atlas and continuous graph connections are explicitly enabled and shown in the editor. The generated changes and gradient checks can be reviewed before training. This custom profile is distinct from the full original model.

## Data and storage

Copy `config.example.json` to a machine specific configuration outside the checkout, edit the paths, and point `UNIBRAIN_CONFIG` to it. Relative paths resolve against that configuration file's directory. Absolute paths work on both Windows and Linux.

```json
{
  "sources": "/scratch/unibrain/sources",
  "original_source": "/scratch/unibrain/sources/reg_unibrain",
  "data": "/datasets/my_cohort",
  "runs": "/scratch/unibrain/runs",
  "builds": "/scratch/unibrain/builds",
  "device": "auto",
  "threads": 8
}
```

```bash
export UNIBRAIN_CONFIG=/scratch/unibrain/runtime.json
python scripts/bootstrap.py --source reg_unibrain
python -m unibrain_agent original --epochs 1000
```

The original data interface requires `adhd_train_sample.npy`, `adhd_val_sample.npy`, `adhd_test_sample.npy`, `template_img_orig_96.npy`, `template_img_gm_mask_orig_96.npy` and `template_img_aal_mask_orig_96.npy`. Dataset arrays contain the original fields `id`, `label`, `raw_img_np`, `raw_img_brain_mask_np`, `raw_img_gm_mask_np`, `raw_img_aal_mask_np`. Images remain 96 cubed and batch size remains one. Diagnosis labels are binary. File names are interface names; the files can contain larger cohorts.

The loader memory maps the structured arrays and materializes one sample at a time. Custom profile validation scans data in bounded chunks. This bounds loading memory, but does not establish large cohort throughput or change the original model's image size, batch size, validation cost or checkpoint schedule.

Environment overrides: `UNIBRAIN_SOURCES`, `UNIBRAIN_ORIGINAL_SOURCE`, `UNIBRAIN_DATA`, `UNIBRAIN_RUNS`, `UNIBRAIN_BUILDS`, `UNIBRAIN_DEVICE`, `UNIBRAIN_THREADS`, `UNIBRAIN_ORIGINAL_PYTHON`. The original Python executable defaults to the current interpreter. Set the last variable when the UI and original CUDA worker use separate environments.

## Remote server access

The server binds only to loopback. Run it on the remote host, then forward its port from your computer:

```bash
ssh -L 8773:127.0.0.1:8773 user@server
```

Open the local forwarded URL. No publicly exposed service or web login is required. On clusters, use an allocated compute node and the institution's SSH routing.

## GPU selection and job scheduling

`device: auto` uses the first visible GPU for original training. `cuda:N` selects an index within `CUDA_VISIBLE_DEVICES`, respecting an existing scheduler allocation. Original source uses CUDA device zero inside the isolated worker, so the selected GPU is remapped to that index.

```bash
CUDA_VISIBLE_DEVICES=2 UNIBRAIN_RUNS=/scratch/job2/runs python -m unibrain_agent original --epochs 1000
```

Run separate experiments on different GPU allocations with separate output directories. This is single GPU training per experiment. Distributed data parallel training, automatic multi GPU scaling, checkpoint resume and large cohort performance have not been implemented or validated. The full original model has 305,812,248 parameters and can exceed small GPU memory capacity; provision resources for the original architecture rather than expecting adapter memory use.

## Windows

Create a Python 3.10 environment, install the same requirements, and run the same Python commands. Use `.venv/Scripts/python.exe` in place of `.venv/bin/python`. `Launch UniBrain Agent.cmd` launches the local service. An ignored `local.json` can supply the UI `python` executable and a separate `original_python` executable. No user specific Windows path is committed.

## CPU only development

Install `requirements.txt` with a CPU PyTorch wheel and fetch sources and samples. The UI starts normally; the original workflow reports that CUDA is required. Explicitly select an adapter configuration to run the compact CPU profile, or use the headless commands:

```bash
python -m unibrain_agent prepare catalog/adapter_selection.json --backend rules
python -m unibrain_agent run --build-id BUILD_ID --single --backend rules --steps 2 --device cpu
```

The rules backend is an explicit deterministic test mode. The interactive custom pipeline uses Codex CLI with an existing login; a failed Codex invocation never silently falls back to rules. The untouched original workflow does not require an agent login.

## Verified environments

The [execution record](../catalog/portability_verification.json) contains the November 9, 2026 checks. The Windows dashboard completed full original CUDA training for two epochs, ten updates and two validation passes with unchanged upstream sources and a finite checkpoint. The compact adapter profile completed two updates and five validation samples on both Windows CUDA and Ubuntu WSL CPU. Twenty five unit tests passed on Windows and Ubuntu. Linux full original CUDA training and large cohort throughput remain unverified.
