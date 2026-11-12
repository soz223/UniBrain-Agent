# Executed training objectives

The frontend renders backend terms and coefficients. It contains no method to
loss mapping. `/api/objective/preview` returns a declared configuration, clearly
separate from measured execution. Disconnected custom selections show an
unavailable state instead of retaining the previous formula.

The full replacement trainer collects `objective` immediately before backward.
Each term records its scalar value, effective coefficient, weighted value and
autograd operations. Walking the actual autograd graph identifies reachable
parameter modules. A missing path is not silently replaced with an expected
connection. Structural reachability is distinct from a finite, nonzero gradient;
the existing full host gradient checks supply the latter evidence.

The sum of recorded weighted terms must reconstruct the executed total within
floating point tolerance. An unreported penalty fails this check. TaskForce
coefficients come from the current policy decision. Standalone and frozen-stage
settings record their effective coefficients, including zero coefficients.
Regularization is recorded as its own term using the tensor used in the total.

Research cycle responses return the latest recorded objective and gradient
checks. Older runs lacking this record display an unavailable state. The UI
does not infer their losses from a method name. Existing immutable checkpoints
and upstream source files are unchanged.

The original upstream entry point and the compact runner also emit executed
objective records through the live monitor. The original entry's coefficients
are read from its bound training arguments. Compact loss components share the
same tensors as the grouped training losses, so smoothness and AAL terms can be
expanded using the effective task weights without recomputing a forward pass.

## Bounded validation

`catalog/objective_acceptance_20261112.json` contains two full host checks on the
original ABIDE round 1 acceptance subset, with native 96 cubed input. VoxelMorph
registration with an explicit 0.01 regularization coefficient and RED WGAN
denoising each completed two optimizer updates. Both passed the existing input
and candidate gradient checks. RED WGAN uses downstream task supervision, with
no separate denoising or discriminator objective. These checks establish
execution and display consistency, not cohort performance or convergence.

An additional original-entry control used one training subject for two epochs
to meet the upstream checkpoint requirement while keeping two optimizer updates.
The original finite-gradient and active-module update checks passed, and both
recorded objectives reconstructed the original total. Its validation and test
arrays were retained from the same acceptance subset.

The verification used an isolated checkout and output directories on an idle
GPU. Existing training processes and their source directories were not changed.


### Stage alignment

The backend supplies an ordered `stages` list with `loss_ids` for each active stage.
Both pipeline cards and objective cards use this order. Regularization belongs to
registration; stages with no independent objective retain an N/A card and do not
connect to the loss summation bus. N/A does not establish gradient connectivity.
The measured autograd connections remain separate evidence.

The selection view stays selected when a saved research run is loaded. Choosing
Full research run switches both rows together to the recorded configuration and
disables draft preparation until the selection view is restored.
