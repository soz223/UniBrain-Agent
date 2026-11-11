# Full UniBrain replacement loop

This experimental command replaces extraction with the full default SynthStrip
network inside the pinned original UniBrain model. It does not use the compact
composition runner. Other host modules, all five registration stages, the four
original loss terms and Adam settings are retained. Initialization is random;
this is a trainability experiment, not pretrained SynthStrip reproduction.

The configured AI provider writes a Python adapter function. A separate worker
checks it in the full model and returns measured diagnostics to the provider
for revision. Each proposal, executed adapter, prompt, provider usage, failure
and checkpoint remains in a new output directory. No source file, baseline
configuration, service or baseline checkpoint is modified.

## Portable setup

Use Python 3.10 and the dependencies in `requirements-upstream.txt` and
`requirements.txt`, on Windows or Linux with CUDA. Configure the pinned source
root with `UNIBRAIN_SOURCES` or an external configuration file as documented in
[deployment.md](deployment.md). Fetch `reg_unibrain` and `freesurfer_synthstrip`
with `scripts/bootstrap.py`. Configure your own supported AI provider.

From the repository root, using that environment's Python:

```sh
python scripts/run_full_replacement.py \
  --source /path/to/pinned/reg_unibrain \
  --data /path/to/UniBrain/data \
  --output /path/to/new/experiment \
  --provider codex --gpu 0 --steps 2
```

On PowerShell put the command on one line, or use PowerShell line continuation.
`--python` can select a separate CUDA worker interpreter. The data directory
uses the existing structured training NPY and the three original template NPY
files. Only the first one or two training subjects are read. No test subjects
are used. The output directory must not exist. Paths are arguments, not machine
specific constants. Generated functions contain no machine specific paths.

The worker checks the selected GPU before creating the CUDA model. It refuses
occupied devices rather than waiting inside or changing the baseline queue.
On Windows, WDDM lists desktop clients as compute clients; the guard excludes
desktop rendering but checks for training process names and available memory.
This check is advisory and not a cluster scheduler reservation. Use your
cluster's allocation mechanism on shared machines.
`--gpu` indexes the devices allowed by `CUDA_VISIBLE_DEVICES` when that
variable is present; the worker never expands the scheduler's allocation.

## Checks and evidence

The original host must contain 305,812,248 parameters before replacement.
Untouched module identities and parameter counts are recorded. SynthStrip uses
its default 16 base channels, seven levels and maximum 64 channels. Images are
padded from 96 to 128 cubed and cropped back; they are not downsampled.

The required path is registration loss to extraction parameters. The worker
also checks extraction input gradients, output shape and range, finite joint
gradients and observed optimizer updates in extraction, registration, tissue
segmentation, region features and classification. Original discrete atlas and
graph boundaries remain; this does not certify classification gradients to
every earlier stage. The original unused parcellation head remains present.

Successful validation takes at most two optimizer steps, checks inference,
and saves an isolated full model state dictionary. A failure after any update
ends the loop instead of retrying training and exceeding that budget.
Infrastructure failures such as an occupied GPU or exhausted memory do not
trigger code repair. No provider failure silently falls back to rules.

Use `--inject-hard-mask` to test the repair loop with a deliberately introduced
hard training mask. This is labeled fault injection in the report; it is not a
claim that the provider made that error naturally. The checker must reject it
before an optimizer step, and the next proposal must come from the provider.

The function validator restricts generated code to a small tensor adapter
interface. It is not a general purpose hostile code sandbox. General repository
editing, unseen methods and arbitrary code execution are outside this first
implementation. Official inference fidelity, anatomical correctness and
diagnostic accuracy require separate evaluation.

## Read the result

`report.json` records overall status and the complete attempt sequence.
`attempt_N/validation.json` records full model checks and measured updates.
`attempt_N/adapter.py` is the actual generated or repaired code.
`attempt_N/checkpoint.pt` is saved only for a successful attempt.
Keep prompts, data and checkpoints in private ignored directories.

To reconstruct an accepted model on another machine, configure the same pinned
sources, copy the accepted adapter and checkpoint, then use:

```python
from unibrain_agent.full_model import build_full_model

model, config, architecture = build_full_model(source, adapter, checkpoint)
model.cuda().eval()
```

Loading checks all checkpoint tensors for finite values and uses strict state
dictionary loading. The original host still requires CUDA for its forward
pass. A one or two step checkpoint contains model weights, not Adam state for
an exact interrupted training resume.
