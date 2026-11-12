# Full graph candidates

`unibrain_agent.full_graph_candidates.build_graph_candidate(method)` returns a
PyTorch module and measured architecture metadata. It loads the locked source
definitions and verifies their hashes. The source files remain unchanged.
All three candidates use the complete 116-node AAL graph. These builders do not
use the compact adapters.

| Candidate | Host target | Input | Output | Retained architecture |
| --- | --- | --- | --- | --- |
| FBNetGen pairwise MLP | `graph_gen` | ROI features `[B,116,128]` | Graph `[B,116,116]` | Native `Embed2GraphByLinear`, 256 to 128 to 1 for each pair |
| BrainNetCNN | `gnn_pred` | Graph `[B,116,116]` | Scores `[B,2]` | E2E widths 32 and 64, N2G 256, dense widths 128 and 30 |
| Brain Network Transformer | `gnn_pred` | Graph `[B,116,116]` | Logits `[B,2]` | Two four-head encoders, FFN width 1024, 100 pooled nodes, classifier 800 to 256 to 32 to 2 |

Both classifiers accept the host's second `features` argument for interface
compatibility, but classify adjacency matrices. They do not consume that
additional argument. Their verified input gradient is with respect to the graph.

## Method scope

FBNetGen contributes its pairwise graph generation component. The original
paper uses functional MRI temporal embeddings. This integration supplies the
128-dimensional structural MRI ROI features produced by full UniBrain. The
source MLP width follows that feature dimension. Its directed nonnegative
output, parameter initialization and final ReLU are retained. No embedding
softmax or symmetry projection is added. Relation matrices become registered
buffers so `.to(device)` and checkpoints work across CPU and GPU. This component
reuse does not reproduce the complete FBNetGen temporal extractor, predictor or
training objectives. A dead final ReLU is a possible measured failure and is not
hidden by changing its bias initialization.

BrainNetCNN is the third-party PyTorch implementation in the locked FBNETGEN
repository. It is not the original authors' Caffe implementation, and no Caffe
checkpoint conversion is supplied. The shared source compatibility loader makes
dropout follow training mode; all original layer widths and output activations
are retained. Adjacency matrices now represent the structural MRI pipeline.

BNT uses the original `bnt.yaml` settings, including 100 pooled nodes, frozen
orthogonal centers, projected assignments and no positional encoding. The
upstream constructor changes the first configured node count from 360 to the
provided AAL count, 116. The 100-node pool is retained. Configuration source
hash and commit are recorded in metadata. The shared compatibility loader
adapts the TransformerEncoderLayer call signature to current PyTorch. This
integration reuses its classifier; the original functional MRI preprocessing
and auxiliary training recipe are separate from the UniBrain objective.

## Host connection and validation

Assign the returned module to its declared `host_target`. Preserve continuous
edge weights when running joint training. Original UniBrain's training-time
`edge_perturbation` reconstructs a binary adjacency matrix, disconnecting graph
generation parameters and upstream ROI features from an adjacency-only
classifier. A full integration must explicitly repair that boundary and record
the intervention, for example through the project's connected host.

The CPU tests run the actual pinned cores at the full declared input sizes.
Each checks a finite forward pass, nonzero input and parameter gradients, and
one optimizer update. They also check the native architectural widths and
reject compact feature or node inputs. These are component execution checks.
Full 96 cubed UniBrain execution, end-to-end gradients through every selected
boundary, pretrained weights and diagnostic accuracy require separate evidence.

The extended trainer currently permits random-initialized joint training only.
Its standalone and pretrained-weight modes are explicitly blocked. The lower
level builder accepts a strict complete-wrapper state dictionary: for these
graph modules, keys include the wrapper's `core.` prefix and registered buffers.
A bare source-core checkpoint is rejected rather than silently reinterpreted.
CPU round-trip tests of that interface do not establish compatibility with an
official release, an independent training recipe or every candidate checkpoint.
