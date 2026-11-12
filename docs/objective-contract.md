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

## Bounded validation

`catalog/objective_acceptance_20261112.json` contains two full host checks on the
original ABIDE round 1 acceptance subset, with native 96 cubed input. VoxelMorph
registration with an explicit 0.01 regularization coefficient and RED WGAN
denoising each completed two optimizer updates. Both passed the existing input
and candidate gradient checks. RED WGAN uses downstream task supervision, with
no separate denoising or discriminator objective. These checks establish
execution and display consistency, not cohort performance or convergence.

The verification used an isolated checkout and output directories on an idle
GPU. Existing training processes and their source directories were not changed.
