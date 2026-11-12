"""Generated adapters around pinned full registration and tissue cores."""

import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .full_replacement import AdapterFailure


PREPROCESSORS = ("redwgan", "deepn4", "deepnormalize", "harmonizing_flows")
PARCELLATORS = ("fastsurfer", "nobrainer")
CLASSIFIERS = ("brainnetcnn", "bnt")
METHODS = ("voxelmorph", "lapirn", "nnunet", *PREPROCESSORS, *PARCELLATORS, *CLASSIFIERS, "fbnetgen", "ernet")


def adapter_signatures(method):
    if method == "ernet":
        return {"adapt_extraction": ["core", "image", "if_train"], "adapt_registration": ["core", "fixed", "moving"]}
    if method in CLASSIFIERS:
        return {"adapt": ["core", "graph", "features"]}
    if method == "fbnetgen":
        return {"adapt": ["core", "features"]}
    if method in ("voxelmorph", "lapirn"):
        return {"adapt": ["core", "moving", "fixed"]}
    return {"adapt": ["core", "image"]}


def check_adapter(code, method):
    if method not in METHODS:
        raise ValueError("Unsupported full integration candidate: " + method)
    tree = ast.parse(code)
    signatures = adapter_signatures(method)
    if (len(tree.body) != len(signatures) or any(not isinstance(fn, ast.FunctionDef) for fn in tree.body)
            or {fn.name for fn in tree.body} != set(signatures)):
        raise ValueError("Required adapter functions: " + ", ".join(signatures))
    for fn in tree.body:
        signature = signatures[fn.name]
        if ([arg.arg for arg in fn.args.args] != signature or fn.decorator_list
                or fn.args.defaults or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg
                or fn.returns or any(arg.annotation for arg in fn.args.args)):
            raise ValueError("Required signature: " + fn.name + "(" + ", ".join(signature) + ")")
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.With, ast.AsyncWith,
                 ast.For, ast.While, ast.Try, ast.Global, ast.Nonlocal, ast.Delete,
                 ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.NamedExpr,
                 ast.AugAssign, ast.AnnAssign, ast.Raise, ast.Yield, ast.YieldFrom)
    allowed = {"core", "F.pad", "F.interpolate", "F.grid_sample", "torch.cat", "torch.stack",
               "torch.sigmoid", "torch.softmax", "torch.flip", "torch.zeros_like", "torch.ones_like"}
    tensor_methods = {"to", "float", "reshape", "view", "permute", "transpose", "contiguous",
                      "clone", "detach", "flip", "squeeze", "unsqueeze", "softmax"}
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or isinstance(node, ast.FunctionDef) and node not in tree.body:
            raise ValueError("Unsupported adapter statement")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Private names are not allowed")
        if isinstance(node, ast.Assign) and any(not isinstance(t, ast.Name) or t.id in {"core", "torch", "F"} for t in node.targets):
            raise ValueError("Only local tensor assignments are allowed")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name not in allowed and not (isinstance(node.func, ast.Attribute) and node.func.attr in tensor_methods):
                raise ValueError("Unsupported adapter call: " + name)
    return tree


class GeneratedCore(nn.Module):
    def __init__(self, core, code, method, entrypoint="adapt"):
        super().__init__()
        tree = check_adapter(code, method)
        namespace = {"torch": torch, "F": F, "__builtins__": {}}
        exec(compile(tree, "generated_adapter.py", "exec"), namespace)
        self.core, self.adapt, self.method = core, namespace[entrypoint], method
        self.entrypoint = entrypoint
        self.last_inputs = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "core":
                raise
            return getattr(super().__getattr__("core"), name)

    def forward(self, *inputs):
        self.last_inputs = inputs
        try:
            return self.adapt(self.core, *inputs)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            raise AdapterFailure("Generated adapter failed: " + str(error)) from error

    @property
    def integrate(self):
        return self.core.integrate

    @property
    def fullsize(self):
        return self.core.fullsize

    @property
    def diff_transform(self):
        return self.core.diff_transform

    @property
    def grid_1(self):
        return self.core.grid_1


def bind_adapter(model, method, adapter):
    code = Path(adapter).read_text(encoding="utf-8")
    if method == "nnunet":
        wrapper = GeneratedCore(model.seg_net, code, method)
        model.seg_net = wrapper
    else:
        wrapper = GeneratedCore(model.reg_net.core, code, method)
        model.reg_net.core = wrapper
    return wrapper


def bind_extended_adapters(model, method, adapter):
    code = Path(adapter).read_text(encoding="utf-8")
    check_adapter(code, method)
    targets = tuple(model.candidate_modules)
    if method == "ernet" and set(targets) != {"ext_net", "reg_net"}:
        raise ValueError("ERNet requires both extraction and registration modules")
    if method != "ernet" and len(targets) != 1:
        raise ValueError("Expected exactly one declared candidate module")
    wrappers = {}
    for name in targets:
        entrypoint = {"ext_net": "adapt_extraction", "reg_net": "adapt_registration"}.get(name, "adapt") if method == "ernet" else "adapt"
        wrapper = GeneratedCore(getattr(model, name), code, method, entrypoint)
        setattr(model, name, wrapper)
        wrappers[name] = wrapper
    return wrappers


def tensor_leaves(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in tensor_leaves(child)]
    raise AdapterFailure("Core outputs must contain only tensors")


def compare_tree(actual, reference):
    if torch.is_tensor(reference):
        if not torch.is_tensor(actual) or actual.shape != reference.shape:
            return False, None
        if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
            return False, None
        return True, float((actual - reference).abs().max())
    if not isinstance(actual, (list, tuple)) or not isinstance(reference, (list, tuple)) or len(actual) != len(reference):
        return False, None
    rows = [compare_tree(a, r) for a, r in zip(actual, reference)]
    if not rows or not all(row[0] for row in rows):
        return False, None
    return True, max(row[1] for row in rows)


def core_probe(wrapper, inputs):
    """Compare adapter output with the unchanged core on the same tensor grid."""
    with torch.no_grad():
        actual = wrapper(*inputs)
        reference = wrapper.core(*inputs)
        contract, error = compare_tree(actual, reference)
        fidelity = contract and error <= 1e-5
        if wrapper.method == "nnunet":
            spatial = contract and all(torch.equal(a.argmax(1), r.argmax(1))
                                       for a, r in zip(tensor_leaves(actual), tensor_leaves(reference)))
        elif wrapper.method in ("voxelmorph", "lapirn"):
            index = 2 if wrapper.method == "voxelmorph" else 0
            spatial = contract and torch.allclose(actual[index], reference[index], atol=1e-6, rtol=1e-5)
        elif wrapper.method in PARCELLATORS:
            spatial = contract and torch.equal(actual.argmax(1), reference.argmax(1))
        else:
            spatial = fidelity
    return {"contract": bool(contract), "fidelity": bool(fidelity), "spatial": bool(spatial),
            "maximum_core_error": error, "reference_shapes": [list(t.shape) for t in tensor_leaves(reference)],
            "scope": "Same pinned core, weights, native outputs and voxel coordinates; not pretrained clinical accuracy"}, reference


def gradient_evidence(loss, parameters, inputs):
    parameters = tuple(value for value in parameters if value.requires_grad)
    inputs = tuple(value for value in inputs if torch.is_tensor(value) and value.requires_grad)
    if not loss.requires_grad:
        return {"passed": False, "parameter_norm": 0., "input_norm": 0., "connected_tensors": 0}
    grads = torch.autograd.grad(loss, (*parameters, *inputs), allow_unused=True, retain_graph=True)
    pgrads, igrads = grads[:len(parameters)], grads[len(parameters):]
    finite = all(torch.isfinite(g).all().item() for g in grads if g is not None)
    norm = lambda values: sum(g.detach().double().square().sum().item() for g in values if g is not None) ** .5
    pnorm, inorm = norm(pgrads), norm(igrads)
    return {"passed": bool(inputs) and finite and pnorm > 0 and inorm > 0, "parameter_norm": pnorm, "input_norm": inorm,
            "connected_tensors": sum(g is not None for g in pgrads), "total_tensors": len(parameters), "finite": finite}


def registration_geometry(model, output, reference):
    """Independently reconstruct the dense plus affine sampling grids."""
    registration = model.reg_net
    if registration.method == "voxelmorph":
        velocity = reference[2]
        positive, negative = [registration.core.integrate(v) for v in (velocity, -velocity)]
        if registration.core.fullsize:
            positive, negative = [registration.core.fullsize(v) for v in (positive, negative)]
        positive, negative = [v[:, [2, 1, 0]] * (2 / 95) for v in (positive, negative)]
    else:
        positive, velocity = reference[0], reference[3]
        negative = registration.core.diff_transform(-velocity, registration.core.grid_1)
    shape = output[0][-1].shape
    device, dtype = positive.device, positive.dtype
    affine, inverse = output[3][-1], output[4][-1]
    identity = F.affine_grid(torch.eye(3, 4, device=device, dtype=dtype)[None], shape, align_corners=True)
    displaced = identity + positive.movedim(1, -1)
    forward = torch.matmul(displaced, affine[:, :, :3].transpose(1, 2).view(1, 1, 1, 3, 3))
    forward = forward + affine[:, None, None, None, :, 3]
    backward = F.affine_grid(inverse, shape, align_corners=True)
    backward = backward + F.grid_sample(negative, backward, padding_mode="border", align_corners=True).movedim(1, -1)
    observed = (registration.forward_grid, registration.inverse_grid)
    errors = [float((a - b).abs().max()) for a, b in zip(observed, (forward, backward))]
    finite = all(torch.isfinite(t).all().item() for t in (*observed, forward, backward))
    warped = F.grid_sample(output[0][-1], forward, align_corners=True)
    warp_error = float((warped - output[2][-1]).abs().max())
    return {"passed": finite and max(*errors, warp_error) <= 1e-5, "grid_maximum_errors": errors,
            "warped_image_maximum_error": warp_error, "align_corners": True,
            "units": "Voxel velocity converted to normalized xyz" if registration.method == "voxelmorph" else "Normalized xyz",
            "scope": "Dense and affine grid composition at 96 cubed; no NIfTI physical affine is supplied by the dataset"}
