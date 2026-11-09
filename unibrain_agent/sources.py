import ast
import functools
import hashlib
import inspect
import json
import math
import sys
from abc import abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .paths import CATALOG, SOURCES


def verify(names):
    manifest = json.loads((CATALOG / "sources.json").read_text(encoding="utf-8"))
    evidence = []
    for spec in manifest["repositories"]:
        if spec["directory"] not in names:
            continue
        for item in spec["files"]:
            path = SOURCES / spec["directory"] / item["path"]
            if not path.is_file():
                raise ValueError(f"Missing source: {path.name}. Run scripts/bootstrap.py.")
            digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Pinned source changed: {spec['directory']}/{item['path']}")
        evidence.append({"repository": spec["repository"], "commit": spec["commit"],
                         "verified_files": len(spec["files"]), "verified_clean": True})
    if len(evidence) != len(set(names)):
        raise ValueError("Unknown source requested")
    return evidence


class Compatibility(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "cuda":
                node.func.attr, node.args = "to", [ast.Constant("cpu")]
            elif node.func.attr == "getargspec":
                node.func.attr = "getfullargspec"
                return ast.copy_location(ast.Subscript(node, ast.Slice(upper=ast.Constant(4)), ast.Load()), node)
            elif (node.func.attr == "dropout" and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"
                  and not any(k.arg == "training" for k in node.keywords)):
                node.keywords.append(ast.keyword(arg="training", value=ast.Attribute(ast.Name("self", ast.Load()), "training", ast.Load())))
            elif node.func.attr == "__init__" and len(node.args) == 10 and isinstance(node.args[-2], ast.Name) and node.args[-2].id == "device":
                device, dtype = node.args[-2:]
                node.args = node.args[:-2]
                node.keywords.extend([ast.keyword(arg="device", value=device), ast.keyword(arg="dtype", value=dtype)])
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == "_sa_block" and "is_causal" not in [arg.arg for arg in node.args.args]:
            node.args.kwonlyargs.append(ast.arg(arg="is_causal"))
            node.args.kw_defaults.append(ast.Constant(False))
        return node


def scope():
    return {"torch": torch, "nn": nn, "F": F, "nnf": F, "np": np, "math": math,
            "Tensor": Tensor, "Parameter": nn.Parameter, "Linear": nn.Linear,
            "Normal": torch.distributions.Normal, "Variable": torch.autograd.Variable,
            "inspect": inspect, "functools": functools, "TransformerEncoderLayer": nn.TransformerEncoderLayer,
            "DictConfig": SimpleNamespace, "List": List, "Optional": Optional, "Tuple": Tuple,
            "abstractmethod": abstractmethod, "softmax": F.softmax}


def load(relative, names, namespace=None):
    namespace = namespace if namespace is not None else scope()
    path = SOURCES / relative
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Source definitions missing: {relative}")
    nodes.sort(key=lambda n: n.lineno)
    module = ast.fix_missing_locations(Compatibility().visit(ast.Module(body=nodes, type_ignores=[])))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def package(folder):
    path = str(SOURCES / folder)
    if path not in sys.path:
        sys.path.insert(0, path)


def voxel_core(size=32):
    ns = load("reg_voxelmorph_master/voxelmorph/torch/modelio.py", {"store_config_args", "LoadableModel"})
    layers = load("reg_voxelmorph_master/voxelmorph/torch/layers.py", {"SpatialTransformer", "VecInt", "ResizeTransform"})
    ns["layers"] = SimpleNamespace(**layers)
    load("reg_voxelmorph_master/voxelmorph/py/utils.py", {"default_unet_features"}, ns)
    load("reg_voxelmorph_master/voxelmorph/torch/networks.py", {"Unet", "VxmDense", "ConvBlock"}, ns)
    return ns["VxmDense"]((size,) * 3, nb_unet_features=[[4, 8], [8, 8, 4]], int_steps=7, int_downsize=1, bidir=True)


def bnt_core(nodes=116):
    ns = scope()
    for file, names in [("base.py", {"BaseModel"}), ("BNT/components/transformer_encoder.py", {"InterpretableTransformerEncoder"}),
                        ("BNT/ptdec/cluster.py", {"ClusterAssignment"}), ("BNT/ptdec/dec.py", {"DEC"}),
                        ("BNT/bnt.py", {"TransPoolingEncoder", "BrainNetworkTransformer"})]:
        load("BrainNetworkTransformer/source/models/" + file, names, ns)
    config = SimpleNamespace(dataset=SimpleNamespace(node_sz=nodes), model=SimpleNamespace(
        sizes=[nodes, 8], pooling=[False, True], pos_encoding="none", orthogonal=True,
        freeze_center=True, project_assignment=True, pos_embed_dim=nodes))
    return ns["BrainNetworkTransformer"](config)
