Respond to the user in concise Chinese. Write code, comments and product text in English.

Use this repository as the maintained UniBrain Agent implementation. Keep upstream source files immutable and pinned in catalog/sources.json. Do not commit datasets, checkpoints, credentials, local paths or agent transcripts.

Every executable adapter must preserve its documented tensor interface. Verify finite outputs, input gradients, parameter gradients and optimizer updates before describing a configuration as trainable. Isolated core checks do not establish whole pipeline compatibility. Keep failed builds and their diagnostics; never silently substitute a different method.

Run the relevant unit tests and a sample build when changing pipeline composition. Run browser interaction checks when changing the stage editor. Keep publication evidence separate from current integration results.
