# Classroom demonstration

Retired on November 23, 2026. The current teaching interface is the
[original workspace with embedded simulated data](https://soz223.github.io/UniBrain-Agent/?demo=1).
See [demo-mode.md](demo-mode.md). The standalone classroom page was removed
and its dedicated service was stopped. The instructions below are retained
as historical documentation of the isolated inference implementation.

Updated 2026-11-23. This document describes the portable deployment procedure.
Deployment-specific verification belongs in the dated classroom handoff.

The public page is `classroom.html`. Its controller uses a real LLM to select
`inspect_pipeline`, then `run_inference`, then explain the measured result.
The harness validates each decision and executes a fixed Python worker.
The worker loads the complete PaperUniBrain architecture and a trained checkpoint.
This is inference, not a training demonstration or a cohort performance test.

## Isolated assets

Use a separate checkout, environment, asset directory and writable state directory.
Copy a completed checkpoint into the asset directory; do not point at an active
`best.pt` or `last.pt`. Record its SHA256 and training provenance. Select an
unmodified upstream public sample, preferably extracted from the pinned Git
object rather than an old local dataset. Export only its `raw_img_np` as a
96 by 96 by 96 numeric NPY. Keep subject identifiers and reference labels out
of the demonstration asset. Public availability does not remove source terms;
retain the source attribution and verify that the intended display is permitted.

Copy these templates beside it:

1. `template_img_orig_96.npy`
2. `template_img_gm_mask_orig_96.npy`
3. `template_img_aal_mask_orig_96.npy`

Use the pinned original source from `catalog/sources.json`. Install the normal
upstream dependencies in a dedicated environment. Keep checkpoint, sample and
source copies read-only. No production directory is a demo output directory.
Keep `state` outside the checkout, assets and configuration directories.

`scripts/classroom_inference.py` loads weights strictly. It uses CPU tensors,
`eval()` and `torch.inference_mode()`, with no optimizer or gradient updates.
Two process-local affine identity helpers allocate on CPU instead of CUDA.
The checkpoint's registration and template normalization settings are preserved.
The source files and affine equations remain unchanged.

## Configuration and credentials

Copy `configs/classroom.example.json` outside the repository and replace its
example absolute paths. Set a model ID actually available to the chosen account.
API providers use their server-side credential environment variable. For Codex,
set the provider to `codex` and configure the desired explicit model ID; prepare
the isolated `state/codex` login before launching the service. Do not copy an
entire personal Codex configuration or enable unrelated tools for the demo.
The CLI disables native tools; only the harness exposes the two allowed actions.

Create a private environment file with mode `600`. It must already exist before
installation and contain `UNIBRAIN_DEMO_TOKEN` with at least 32 random characters.
Use a fresh token generated locally, for example with Python `secrets.token_urlsafe`.
API mode also needs `OPENAI_API_KEY`, or the configured `api_key_env` variable.
Never commit this file, print its contents, or put its values in a URL.

Students can read health and public demonstration results. Only the presenter
can create a run. The token is entered in the page's presenter controls and is
kept in browser session memory. The server permits one run at a time, limits
request length and hourly runs, and rejects unknown tools or invalid ordering.
Image visualizations stay in the UI; the LLM receives numeric summaries only.
Use demonstration prompts suitable for public display: requests and responses
are visible in the public trace.

## Linux service

The installer takes explicit paths and never changes the production backend.
Run it on the destination host, using paths appropriate to that host:

```sh
python scripts/install_classroom_service.py \
  --project /srv/unibrain-demo/app \
  --python /srv/unibrain-demo/venv/bin/python \
  --config /srv/unibrain-demo/config/classroom.json \
  --env-file /srv/unibrain-demo/config/classroom.env \
  --assets /srv/unibrain-demo/assets
```

The default writes and verifies a new unit without starting it. Add `--start`
after checking the configuration. Existing units are not overwritten.
The unit hides CUDA, limits CPU to two cores and memory to 12 GB, gives the
process lower scheduling priority, and makes the filesystem read-only except
for demo state and private temporary storage. The cgroup also contains workers.
These controls depend on the host's user-service namespace and cgroup support;
verify them on the destination rather than weakening them silently.

```sh
systemctl --user status unibrain-classroom.service
journalctl --user -u unibrain-classroom.service -f
systemctl --user show unibrain-classroom.service -p CPUQuotaPerSecUSec -p MemoryMax
```

For availability after logout, check the site's user-service lingering policy.
An SSH forwarding session alone is not a public classroom deployment.

## HTTPS and GitHub Pages

Keep the API on `127.0.0.1:18878` and expose it through an HTTPS reverse proxy
or an approved tunnel. Only the demo service belongs behind this endpoint.
Do not proxy the production backend or its training APIs.
Set `backend_url` in `web/classroom-config.json` to the HTTPS base URL, then run
`python scripts/build_frontend.py` and deploy through the existing Pages workflow.
The frontend also supports entering an endpoint manually.

A temporary tunnel URL can change when its process restarts. Recheck health,
update the public configuration, and republish when it changes. Prefer a named
tunnel or stable reverse-proxy hostname for recurring lectures. Pages serves
the interface; live calls require the backend, network and LLM credentials.
Keep the presenter token out of public frontend configuration.

Before class, complete one end-to-end request through the public page. Verify
the provider and model ID, three real LLM decisions, tool start and completion
events, returned probabilities and actual MRI images. Check that production
training processes and GPU usage continue normally. An HTTP health response
alone does not establish that the LLM account or model worker can execute.

## Ten-minute classroom sequence

1. Minute 0 to 2: identify the domain model, LLM and harness in the architecture.
2. Minute 2 to 3: show the Python wrapper and the example request below.
3. Minute 3 to 6: submit once. Follow the real tool calls while the CPU model runs.
4. Minute 6 to 8: inspect predicted probabilities, stage outputs and MRI slices.
5. Minute 8 to 10: distinguish tool execution from model reasoning and discuss
   how a different Python model could implement the same tool interface.

Example prompt:

> Inspect the current UniBrain pipeline, run inference on the demo image, and
> explain the observed prediction and intermediate outputs in plain English.

Use generic class labels until the checkpoint's class mapping is verified.
Never present a single-sample probability as accuracy, AUC or a clinical diagnosis.
Keep one completed trace available as a clearly labeled previous execution if
the live service is unavailable. Do not portray a previous run as a new LLM call.
The full CPU worker may take minutes; rehearse on the actual host and leave
enough time for its measured latency. Do not start repeated requests while busy.

Stop only the demonstration service when finished:

```sh
systemctl --user stop unibrain-classroom.service
```

Keep demonstration logs separate from publication experiments. The service has
no training, code modification or production-data upload endpoint.
# Verified classroom deployment

Verified on November 23, 2026 (America/New_York). The public entry is
https://soz223.github.io/UniBrain-Agent/classroom.html. Add `?recording=1`
for an accelerated replay of a completed real execution. Replay makes no
LLM or inference requests and remains explicitly labeled as recorded.

The live deployment uses Codex CLI with `gpt-6-astra`, two allowlisted tools,
and the complete paper-form UniBrain model with 271,483,811 parameters.
The first accepted run made three LLM calls, executed one CPU forward in
28.972 seconds, and recorded zero training updates. Its total elapsed time
was approximately 65 seconds. This verifies invocation, not model accuracy.
The checkpoint is a copied ADHD experiment artifact, and the input MRI
comes from the pinned upstream public example. No cohort performance or
independent test result is claimed for this demonstration.

The deployment has a 2-CPU quota, 12 GiB memory limit, hidden CUDA devices,
and a read-only filesystem except its isolated state directory. The normal
training backend runs as a separate, unchanged service. Public viewers can
read results; starting a run requires the separate instructor credential.

The current HTTPS address uses a temporary Cloudflare tunnel. Its hostname
can change after a tunnel restart; update `web/classroom-config.json` and
publish Pages again if that happens. A permanent deployment should use a
named tunnel or an existing HTTPS reverse proxy. The static replay remains
available through Pages independently of the inference backend.

