# Integration benchmark

The driver separates the agent's visible feedback from an independent final
evaluation. Disabling gradient feedback never disables the final gradient
criterion. Results distinguish failed integration, rejected applicability and
blocked infrastructure. A rejected candidate is not a successful integration.

## Commands

```sh
python scripts/benchmark_integration.py preflight --config experiment.json
python scripts/benchmark_integration.py run --config experiment.json --output private_runs/new_trial
python scripts/benchmark_integration.py faults --output private_runs/new_fault_test
python scripts/generate_integration_config.py --source sources/reg_unibrain --data data/fold1 --output private_runs/integration_configs --plans data/nnUNetPlans.json
```

The output directory must not exist. The `faults` command runs a CPU synthetic
checker test with a clean control and 14 injected defects. These results test
checker plumbing, not the 14 paper methods or full MRI processing pipelines.
The defects cover detachment, a NumPy round trip, hard masks, nearest sampling,
frozen or omitted parameters, orientation, grid shifts and declared unit errors.

`run` calls the configured real AI provider. It never substitutes a rules
backend or simulated agent. All source evidence is supplied explicitly; images
and subject rows are not included in prompts. Provider errors remain blocked.

## Configuration

Paths to source files and evaluator working directories resolve relative to
the configuration file. The evaluator is an argument list, never a shell
command. Configure its executable and own AI account on the target machine.

```json
{
  "version": 1,
  "cases": [{
    "id": "synthstrip_extraction",
    "method": "SynthStrip",
    "stage": "extraction",
    "expected_fit": true,
    "source_files": ["sources/host_excerpt.py", "sources/candidate_excerpt.py"],
    "interface": "Define adapt(core, image, if_train). Return two single element lists: extracted image and mask. Input is 96 cubed; core is the full default SynthStrip. Available names: torch and F. Only core, F.pad, torch.sigmoid and tensor.to calls are accepted.",
    "knowledge": "Keep the registration loss connected to extraction parameters. The core requires padding to 128 cubed. Preserve the tensor grid when cropping. A hard decision belongs to inference only.",
    "max_optimizer_updates": 2,
    "evaluator": {
      "cwd": ".",
      "command": ["{python}", "scripts/evaluate_integration.py", "--source", "sources/reg_unibrain", "--data", "data/sample", "--adapter", "{adapter}", "--output", "{output}", "--seed", "{seed}", "--gpu", "0", "--steps", "2"],
      "timeout": 1800
    }
  }],
  "providers": [{"provider": "codex", "model": ""}],
  "variants": ["general", "checks", "workflow"],
  "seeds": [17],
  "repetitions": 3,
  "max_calls": 4
}
```

The example paths are placeholders. `preflight` checks their availability and
the evaluator executable; it does not certify provider authentication, model
accuracy, GPU availability or future evaluator completion.

The configuration generator writes one concrete case file for each of
SynthStrip, VoxelMorph, LapIRN and nnU-Net. Select a subset with `--methods`.
Each file contains one provider and can be used by the experiment suite's
`integration_configs` mapping. Configure the provider account locally;
credentials are never embedded in the files. Regenerate these machine-specific
configurations after migration. Source and runtime paths come from CLI arguments
and the portable local configuration, not usernames in the program.

nnU-Net remains blocked until a real dataset-specific plans file is supplied.
The supported plan is native 96 cubed PlainConvUNet with four tissue classes;
the evaluator does not invent a generic plan. `index.json` explicitly lists
the other ten paper candidates as not implemented in the full integration
benchmark. They are not routed to compact adapters.

## Compared procedures

`general` receives the same source and interface and submits one proposal.
`checks` may request the checker through the structured `action=check` tool
protocol and then submit. `workflow` screens applicability and follows the
integration, checking and revision procedure with the supplied knowledge.
This is a controlled prompting comparison on one common execution harness,
not a reproduction of all features of the standalone Codex or Claude tools.

The available ablations are `without_screening`, `without_knowledge`,
`without_spatial`, `without_gradient`, `without_fidelity`, and
`without_revision`. Applicability accuracy requires human adjudicated positive
and negative cases. The source files, interface, provider, model, call budget
and evaluator must remain fixed across a matched comparison. Screening counts
against the call budget. CLI LLM sampling is not guaranteed deterministic by
the checker seed; repetitions measure this variation.

All checks always contribute to final validity. A hidden check is removed from
both structured feedback and diagnostics; raw error strings are not forwarded
because they can disclose the ablated signal. A code hash reuses an identical
evaluation within the trial. Any observed optimizer update closes the training
budget, including an update followed by a failed check. The trial cannot ask
for another evaluated revision after that point. The evaluator must itself
obey its per-call step bound; an overrun is blocked, never accepted as valid.

## Independent evaluator contract

The executable receives `{adapter}`, `{output}` and `{seed}` and writes:

```json
{
  "status": "passed",
  "runs": true,
  "checks": {"contract": true, "gradient": true, "spatial": true, "fidelity": true, "updates": true},
  "optimizer_updates": 2,
  "diagnostic": {"gradient": {"norm": 0.01}}
}
```

The numbers above describe the schema, not a measured result. Success requires
all five Boolean checks plus observed optimizer updates. Missing checks block
success. Any measured failure is sufficient to reject an integration even if
later checks are unmeasured. Infrastructure failure is reported separately.

`scripts/evaluate_integration.py` selects a concrete evaluator with `--method`.
The default is SynthStrip. It executes the existing full UniBrain worker in
the evaluator process, then independently compares the
adapter with the pinned SynthStrip core at the same weights and input. Fidelity
means agreement under the specified padding, cropping and soft-mask contract.
It does not establish equivalence to the complete official inference tool's
preprocessing, pretrained predictions, physical geometry or clinical accuracy.
Its spatial check compares aligned tensor-grid foreground support. A richer
physical-space benchmark needs metadata and its own evaluator.

VoxelMorph and LapIRN adapters replace the callable dense core after the five
original affine stages. The generated `adapt(core, moving, fixed)` function
actually executes during every candidate forward. The fixed evaluator compares
the complete native output structure with the same pinned core at the same
weights, independently reconstructs forward and inverse sampling grids, checks
registration and classification gradients into candidate parameters and moving
inputs, and verifies joint optimizer updates in the candidate and active host
modules. Velocity units, component order, affine composition and sampling corner
conventions remain part of the declared contract. Soft atlas sampling for ROI
features is an explicitly recorded host adaptation.

The nnU-Net adapter replaces `seg_net` through `adapt(core, image)`. It retains
the native list of deep supervision logits. Checks compare their values, grids
and voxel labels, measure tissue-loss gradients into both parameters and MRI,
and verify joint updates. The original auxiliary segmentation head does not
feed the classifier, so this evaluator makes no classification-to-segmentation
gradient claim.

All four evaluators use full host architectures and preserve source hashes,
generated code hashes, actual checks, update counts and checkpoints. They execute
at most one or two training updates. Missing packages, GPU occupancy and invalid
plans remain infrastructure failures. An interrupted evaluator preserves any
already written update count; an uncertain or blocked trial is not retried.

The ten remaining candidates and external agent systems need additional
implementations. A catalog entry or compact adapter alone is not sufficient.
MLE-STAR, NEXUS, NEURA and NeuroPilot are not silently mapped to another provider.

## Evidence

`report.json` contains trial results and separate outcome counts.
`trial_N/result.json` records first-attempt validity, final validity, silent
gradient failure, applicability, real call usage, elapsed time and update count.
The exact prompts, generated code, evaluator outputs and logs are preserved.
LLM cost remains null unless externally priced; token counts are never treated
as dollar cost. This implementation makes no automatic human-audit claim.

CPU tests exercise actual adapter binding, detached outputs with unchanged
values, orientation faults, interface failures and independent grid composition.
These tests establish checker behavior on fixtures. A GPU execution of a
handwritten reference adapter establishes software execution for that candidate;
it is not a measured Agent integration success. Full benchmark trials and model
accuracy experiments require their own reports.

Keep all outputs in an ignored private directory. Use the suite scheduler for
dataset, fold and candidate matrices; this driver does not train full cohorts.
