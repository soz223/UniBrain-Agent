# Deployment

The repository contains the application, pipeline compiler, execution workers, audits, UI, dependency files and source manifest. Upstream code and example data are fetched by `scripts/bootstrap.py` at pinned revisions. Data, checkpoints, credentials and local configuration are excluded from Git.

## Separate frontend and backend

The [public frontend](https://soz223.github.io/UniBrain-Agent/) is hosted on GitHub Pages. Python, the CLI agents, datasets and training run on your own backend. Open **AI settings**, enter the backend URL and access token, and click **Connect backend**. Changing the address moves the workspace to that backend. Disconnecting the browser does not cancel a running job.

The backend address is remembered in browser local storage. Public pages require a manual backend connection on every load; saved credentials do not reconnect automatically. The backend access token stays in tab session storage. AI API keys are sent only to the configured backend and are not stored in browser storage.

AI access starts disconnected on every page load. After manually connecting the backend, save your own provider configuration and click **Connect my AI account**. This explicit action reads that backend's selected account and grants the page a temporary AI session. The session lives only in page memory and expires after eight hours at most. Reloading disconnects the page. Account identity queries and model calls require this session in addition to backend access. **Disconnect AI** revokes it without signing the backend CLI out.

The public page includes the method catalog and role prompts. Before connecting, you can browse these choices; execution controls require a backend. Saved results, slices, conversations, account identity and job progress are retrieved through authenticated API calls. GitHub receives no backend token or AI key through this application.

This is a private workspace per backend. People sharing its access token share its jobs, data and operating system user's CLI accounts. Separate backend instances are needed for isolated users.

## Connect an HTTPS GPU server

Install the backend using the Linux instructions below. Generate a private token on that server and configure the exact frontend origin:

```bash
export UNIBRAIN_ACCESS_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(36))')"
export UNIBRAIN_ALLOWED_ORIGINS=https://soz223.github.io
export UNIBRAIN_PUBLIC_URL=https://gpu.example.edu
python -m unibrain_agent serve --port 8777
```

Keep the token in your private service environment or secret manager. Enter the same token in **Backend access token**. An origin has no repository path: use `https://soz223.github.io`, not the full Pages project URL. Multiple permitted origins can be separated by commas. Wildcards are rejected. API calls require the bearer token whenever remote access is configured.

Put an HTTPS reverse proxy in front of the loopback Python service. For example, a Caddy site configured for your own domain:

```text
gpu.example.edu {
    reverse_proxy 127.0.0.1:8777
}
```

Use `https://gpu.example.edu` as the frontend's backend URL. Configure DNS and the proxy certificate on your server. `UNIBRAIN_PUBLIC_URL` permits the expected HTTP Host header; it is an origin with no path. The proxy must preserve Authorization and Origin headers. If you use a URL prefix, strip that prefix before forwarding to Python. The frontend supports a prefix such as `https://gpu.example.edu/unibrain`.

The default Python bind address is loopback. `--host 0.0.0.0` is available for a private container network behind a proxy; it requires an access token and should not replace HTTPS at the public boundary. HTTP remote addresses are rejected by the frontend. Localhost HTTP is supported for development or an SSH tunnel. A browser may ask for local network permission when Pages connects to localhost; keep the browser's normal security controls enabled.

For a local Windows backend connected from Pages:

```powershell
$env:UNIBRAIN_ACCESS_TOKEN = python -c 'import secrets; print(secrets.token_urlsafe(36))'
$env:UNIBRAIN_ALLOWED_ORIGINS = 'https://soz223.github.io'
python -m unibrain_agent serve --port 8777
```

Copy the generated value into the page's token field and use `http://127.0.0.1:8777`. Treat the token as a password. A strictly local combined UI and backend still works without these environment variables.

## Publish the frontend

The `Publish frontend` workflow builds and deploys `dist` on pushes to `main`. In a fork, enable **Settings → Pages → Source → GitHub Actions**, then run that workflow. The static build needs only the Python standard library:

```bash
python scripts/build_frontend.py
python -m http.server 8786 --bind 127.0.0.1 --directory dist
```

For this local preview, also allow `http://127.0.0.1:8786` in the backend's `UNIBRAIN_ALLOWED_ORIGINS`. The build never embeds a saved experiment or AI configuration. See [GitHub's custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Your AI connection

Open **AI settings** in the sidebar. Select a provider card, optionally set a CLI model or enter an API model ID, save, explicitly connect your account, then test. The account card reads the official CLI's login status, email and plan only after that connection. Saving different settings invalidates existing browser AI sessions. API providers show credential source instead of guessing an account identity. Research, data discussions, role reviews and newly prepared custom builds use the selected connection. CLI `--backend configured` does the same; `--backend codex` explicitly selects Codex, and `--backend rules` is deterministic test mode.

* **Codex CLI:** install the official CLI on the backend machine, make `codex` visible on the backend service process's `PATH`, run `codex login` under that service account, and complete its official authentication flow. Restart the backend after changing its environment. The page provides sign-in instructions and login checks. A headless host can use `codex login --device-auth` where available. The application uses that machine's login, not a bundled account. See [Codex authentication](https://learn.chatgpt.com/docs/auth?surface=cli).
* **Claude Code:** install a current official CLI on the backend machine, run `claude auth login`, and complete sign-in. The adapter requires `--safe-mode`, `--json-schema` and `--no-session-persistence`; it disables built-in and MCP tools for evidence-only decisions. See the [CLI reference](https://code.claude.com/docs/en/cli-reference). The page reuses this login; it does not implement a separate OAuth application.
* **OpenAI API:** enter your model ID and key, or set `OPENAI_API_KEY` on the backend. Uses the Responses API with structured JSON and `store: false`.
* **Anthropic API:** enter your model ID and key, or set `ANTHROPIC_API_KEY`. Uses the Messages API with structured output.
* **OpenAI compatible API:** supply the base URL, model ID and key, or set `UNIBRAIN_API_KEY`. The endpoint must support Chat Completions and JSON schema output. HTTPS is required except for localhost. Redirects are rejected so credentials are not forwarded to another endpoint.

UI-entered keys stay only in backend process memory and are forgotten on restart. Blank key fields preserve the existing session key. Use **Forget the saved session key** to clear it; an environment key, if configured, remains available. Changing a compatible endpoint clears its previous session key. Model and endpoint preferences persist in ignored `ai.local.json`; `UNIBRAIN_AI_SETTINGS` can place that preferences file outside the checkout. Keys and CLI credentials are never written to it or returned by the settings API. Each agent snapshots its provider at creation, and calls record the actual provider and requested model.

The page explains where to run the official login commands. Passwords and OAuth approvals are completed by the user in the official flow. **Switch account** provides those instructions; **Sign out** invokes the official CLI logout after confirmation. It affects that backend operating system user's other CLI sessions too. Account changes are blocked while a workspace job or connection test is running. API access and CLI subscription access have their provider's respective quotas and billing. A connection test makes one small model request.

The provider transport and credential boundaries have automated tests. Codex is tested with the existing local login. Claude Code and API live connectivity must be checked with your installed CLI, credentials and model access; mocked protocol tests do not establish those accounts' availability.

## Linux with a CUDA GPU

Use Python 3.10 for the tested original UniBrain environment. Install the NVIDIA driver on the host before creating the environment. The pinned PyTorch wheel supplies CUDA 12.1 libraries.

```bash
git clone https://github.com/soz223/UniBrain-Agent.git
cd UniBrain-Agent
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-upstream.txt -r requirements.txt
python scripts/bootstrap.py --source reg_unibrain --samples
python -m unibrain_agent serve --port 8777
```

Open `http://127.0.0.1:8777`. The initial pipeline uses the full original UniBrain: original extraction, five registration stages, segmentation, nearest atlas transfer, cosine graph with original edge perturbation, and GCN. Optional stages start skipped. The original architecture, loss equations, augmentation and optimizer are preserved. The default two epoch budget exercises training, validation and checkpoint saving; the upstream paper schedule is 1000 epochs.

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

The server binds to loopback by default. You can access the combined UI through an SSH tunnel instead of configuring a public HTTPS endpoint:

```bash
ssh -L 8777:127.0.0.1:8777 user@server
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

The rules backend is an explicit deterministic test mode. The interactive custom pipeline uses AI settings; a failed model invocation never silently falls back to rules. The untouched original workflow does not require an agent login.

## Verified environments

The [execution record](../catalog/portability_verification.json) contains the November 9, 2026 checks. The Windows dashboard completed full original CUDA training for two epochs, ten updates and two validation passes with unchanged upstream sources and a finite checkpoint. The compact adapter profile completed two updates and five validation samples on both Windows CUDA and Ubuntu WSL CPU. Twenty five unit tests passed on Windows and Ubuntu.

The [November 11 Linux audit](../catalog/linux_original_verification.json) covers smoke subsets from all five folds of ABIDE and ADHD. Each run used two training subjects, two validation subjects, two epochs and four optimizer updates. The test subset was loaded by the original trainer but not evaluated. All ten runs passed source, gradient, module-update, validation and checkpoint checks. Two additional runs through the deployed API passed, for 48 updates in total. The model retained all 305,812,248 parameters. Sixty-eight backend unit tests passed. These checks establish execution on this GPU, not full-cohort throughput or paper accuracy.

## Prepared cohort folds

Export an interface directory from a prepared cohort's master array and original fold indices:

```bash
python scripts/export_cohort_fold.py /datasets/abide /scratch/smoke/abide/round1 --fold 1
```

The default exports one subject per class in each split and labels the result `smoke_subset` in `cohort.json`. It preserves each selected record exactly and converts the supplied original 96 cubed NIfTI templates to NPY. Add `--full` to export every record in that fold. The original master array and fold indices remain unchanged. Output interface filenames retain the upstream `adhd_` prefix for either cohort.

For the ten-fold smoke runner, export directories named `data/abide/round1` through `round5` and `data/adhd/round1` through `round5` under one workspace. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_cohort_folds.py /scratch/smoke --source sources/reg_unibrain
```

The runner executes folds sequentially and stops on the first failed audit. It retains logs, checkpoints and `summary.json`. Existing run folders are not overwritten. In the frontend's Data workspace, inspect a specific exported fold directory and attach it before preparing a new pipeline.

## Persistent Linux backend

After verifying the runtime, install a user service with a private token and an explicit dataset:

```bash
python scripts/install_user_backend.py --data /scratch/smoke/abide/round1 --state /scratch/unibrain-service --gpu 0 --port 18777
loginctl enable-linger "$USER"
```

The installer creates `unibrain-backend.service`, an external runtime configuration and a mode-600 `backend.env`. Existing service configurations are not overwritten. Linger keeps the user service available after logout where host policy permits. Check it with `systemctl --user status unibrain-backend` and view native service logs with `journalctl --user -u unibrain-backend`.

Forward the same port from your workstation:

```bash
ssh -N -L 18777:127.0.0.1:18777 user@server
```

In the public frontend's AI settings, use `http://127.0.0.1:18777` and the token from the server's private `backend.env`. AI accounts remain disconnected until explicitly connected. The service defaults to the dataset supplied at installation; a smoke directory remains a small test subset until a full exported fold is inspected and attached. Only one experiment runs per backend instance. This does not reserve the GPU against unrelated processes on the host.

## Full cohort baseline queue

Run `scripts/run_baseline_queue.py` with a workspace, cohort root, backend, state directory, epoch budget and physical GPU index. It exports each original fold, attaches it through the backend, checks GPU occupancy immediately before each launch, and runs the unchanged UniBrain architecture and losses. The deployed frontend displays the current fold through the backend status stream. The queue records fold mapping and GPU checks in `baseline_queue.json`, and stores complete test metrics beside each run. It stops on a failed audit or run. A restarted queue requires review of the existing state before another launch.

The original entry point requests 1000 epochs and saves a model after every epoch except the first. A smaller epoch budget is a budgeted baseline, not a full reproduction of that schedule. The original trainer loads the test split but does not evaluate it. `scripts/evaluate_upstream.py` evaluates the saved checkpoint on the held out test split after training, without changing the original training source. Tissue and atlas masks in prepared cohorts may be automatic reference labels; interpret their overlap metrics accordingly.
