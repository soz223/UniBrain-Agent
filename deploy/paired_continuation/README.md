# Paired continuation

`launch_pair.py` accepts a paired experiment manifest. Each job specifies its
command, working directory, output directory and environment. The manifest must
also specify `release_manifest`. The launcher verifies the code, upstream source
and dependencies before alternating the two arms on their allocated GPU.

Keep the model release immutable after launch. Place copied resume checkpoints
in new output directories and retain their source files. Record the shared source
checkpoint hash. See `docs/registration-normalization.md` for the supported
normalization migration and the additional continuation epoch budget.

`run_allocation.py` accepts the manifest and its Slurm batch script. Use a 24 hour
allocation. It requests graceful checkpointing after 23 hours 45 minutes and
submits a dependent continuation only for this scheduled allocation boundary.
An external cancellation does not submit another job. Training failures remain
visible through the bounded supervisor and do not create an unbounded retry loop.
The compute node must have Slurm client access for allocation continuation.

Both entry points read paths from the manifest. They contain no server addresses,
credentials, datasets or checkpoints. Keep the two files together when deploying.
