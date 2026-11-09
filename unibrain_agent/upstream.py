import ast
import hashlib
import math
import subprocess
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from .data import digest
from .paths import SOURCES, UPSTREAM

UNIBRAIN_COMMIT = "f9f8d906b1725046d532f862a28876482aca2cfa"
SYNTHSTRIP_COMMIT = "4f8728e5132e8d39a8f1bf3d24d15fed62f4fc9d"


def definitions(path, names, nested=False):
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    nodes = ast.walk(tree) if nested else tree.body
    selected = [n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f"Missing upstream definitions in {path}")
    selected.sort(key=lambda n: n.lineno)
    scope = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
             "Variable": torch.autograd.Variable, "GCNConv": GCNConv}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    evidence = [{"symbol": n.name, "line": n.lineno, "end_line": n.end_lineno,
                 "sha256": hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()}
                for n in selected]
    return scope, evidence


def homogeneous(theta):
    bottom = theta.new_tensor([0, 0, 0, 1]).expand(theta.shape[0], 1, 4)
    return torch.cat((theta, bottom), dim=1)


def compose(batch_size, current, previous):
    affine = homogeneous(current) @ homogeneous(previous)
    return affine[:, :3], torch.linalg.inv(affine)[:, :3]


@lru_cache
def unibrain_core():
    names = {"conv_block", "Unet", "encoder", "multi_stage_ext", "multi_stage_reg",
             "Roi_feature_net", "Graph_gen_net_norm", "GNN_predictor", "GCC",
             "feature_perturbation", "edge_perturbation"}
    scope, evidence = definitions(UPSTREAM / "model.py", names)
    # The original helpers allocate CUDA constants unconditionally.
    scope["theta_dot_with_inv"] = compose
    scope["get_batch_identity_theta_3_4"] = lambda size: torch.eye(3, 4).unsqueeze(0).repeat(size, 1, 1)
    return SimpleNamespace(**{name: scope[name] for name in names}, evidence=evidence)


@lru_cache
def synthstrip_core():
    path = SOURCES / "freesurfer_synthstrip" / "mri_synthstrip" / "mri_synthstrip"
    scope, evidence = definitions(path, {"StripModel", "ConvBlock"}, nested=True)
    return SimpleNamespace(StripModel=scope["StripModel"], evidence=evidence)


def verify_sources():
    from .sources import verify
    return verify(["reg_unibrain", "freesurfer_synthstrip"])
