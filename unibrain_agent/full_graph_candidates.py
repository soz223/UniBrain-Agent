"""Pinned graph components at the original UniBrain node and feature sizes."""

from types import SimpleNamespace

import torch
from torch import nn

from .sources import load, scope, verify


NODES = 116
FEATURES = 128
METHODS = ("fbnetgen", "brainnetcnn", "bnt")
BNT_CONFIG = {"sizes": [360, 100], "pooling": [False, True], "pos_encoding": "none",
              "orthogonal": True, "freeze_center": True, "project_assignment": True, "pos_embed_dim": 360}
BNT_CONFIG_SHA256 = "3e1751f2ee893522fc6f4ffd95246bce3e22eae6a50731d3cb93b3c401b5cbf7"


def require_tensor(value, shape, name):
    if not torch.is_tensor(value) or value.ndim != 3 or tuple(value.shape[1:]) != shape:
        raise ValueError(name + " has an incompatible node or feature shape")
    if not value.is_floating_point():
        raise ValueError(name + " must be floating point")


class FullPairGraph(nn.Module):
    def __init__(self):
        super().__init__()
        verify(["FBNETGEN"])
        definitions = load("FBNETGEN/model/model.py", {"Embed2GraphByLinear"})
        self.core = definitions["Embed2GraphByLinear"](FEATURES, roi_num=NODES)
        for name in ("rel_rec", "rel_send"):
            value = getattr(self.core, name)
            delattr(self.core, name)
            self.core.register_buffer(name, value)

    def forward(self, features):
        require_tensor(features, (NODES, FEATURES), "ROI features")
        return self.core(features).squeeze(-1)


class FullBrainNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        verify(["FBNETGEN"])
        definitions = load("FBNETGEN/model/model.py", {"BrainNetCNN", "E2EBlock"})
        self.core = definitions["BrainNetCNN"](NODES)

    def forward(self, graph, features=None):
        require_tensor(graph, (NODES, NODES), "Graph")
        return self.core(graph)


class FullBrainNetworkTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        verify(["BrainNetworkTransformer"])
        namespace = scope()
        for filename, names in (
                ("base.py", {"BaseModel"}),
                ("BNT/components/transformer_encoder.py", {"InterpretableTransformerEncoder"}),
                ("BNT/ptdec/cluster.py", {"ClusterAssignment"}),
                ("BNT/ptdec/dec.py", {"DEC"}),
                ("BNT/bnt.py", {"TransPoolingEncoder", "BrainNetworkTransformer"})):
            load("BrainNetworkTransformer/source/models/" + filename, names, namespace)
        config = {key: list(value) if isinstance(value, list) else value for key, value in BNT_CONFIG.items()}
        self.core = namespace["BrainNetworkTransformer"](SimpleNamespace(
            dataset=SimpleNamespace(node_sz=NODES), model=SimpleNamespace(**config)))

    def forward(self, graph, features=None):
        require_tensor(graph, (NODES, NODES), "Graph")
        return self.core(None, graph)


def build_graph_candidate(method):
    constructors = {"fbnetgen": FullPairGraph, "brainnetcnn": FullBrainNetCNN, "bnt": FullBrainNetworkTransformer}
    if method not in constructors:
        raise ValueError("Unsupported full graph candidate: " + method)
    module = constructors[method]()
    repository = "BrainNetworkTransformer" if method == "bnt" else "FBNETGEN"
    metadata = {"method": method, "source": verify([repository]), "nodes": NODES, "compact": False,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "host_target": "graph_gen" if method == "fbnetgen" else "gnn_pred",
                "input_shape": [None, NODES, FEATURES if method == "fbnetgen" else NODES],
                "output_shape": [None, NODES, NODES] if method == "fbnetgen" else [None, 2],
                "initialization": "Unchanged source initialization; random weights",
                "host_requirement": "Preserve continuous edge weights during training; original binary edge perturbation disconnects graph gradients"}
    if method == "fbnetgen":
        metadata.update(component="Embed2GraphByLinear", hidden_width=FEATURES,
                        adaptation="Pairwise MLP from FBNetGen applied to the 128-dimensional sMRI ROI embeddings; directed native output retained",
                        scope="Component reuse; the original fMRI temporal extractor, embedding softmax, predictor and training objectives are not reproduced",
                        runtime_changes="Device-specific relation tensors registered as buffers; no parameter initialization or architecture changes")
    elif method == "brainnetcnn":
        metadata.update(component="BrainNetCNN", widths=[32, 64, 256, 128, 30, 2],
                        adaptation="Classify the learned sMRI adjacency matrix; the separate ROI feature argument is unused",
                        provenance="Third-party PyTorch implementation bundled in FBNETGEN, not the original author Caffe implementation",
                        runtime_changes="Source dropout obeys module training mode through the shared compatibility loader")
    else:
        metadata.update(component="BrainNetworkTransformer", attention_heads=4, transformer_feedforward_width=1024,
                        pooling_sizes=[NODES, 100], classifier_widths=[800, 256, 32, 2],
                        adaptation="AAL116 sMRI adjacency rows replace the original functional-connectivity node features; no temporal input is consumed",
                        defaults={**BNT_CONFIG, "sizes": list(BNT_CONFIG["sizes"])},
                        defaults_reference="source/conf/model/bnt.yaml at source commit", defaults_sha256=BNT_CONFIG_SHA256,
                        runtime_changes="Modern TransformerEncoderLayer signature compatibility; source frozen orthogonal centers retained",
                        scope="Source BNT classifier architecture with AAL116 input; source fMRI preprocessing and auxiliary training recipe are not reproduced")
    return module, metadata
