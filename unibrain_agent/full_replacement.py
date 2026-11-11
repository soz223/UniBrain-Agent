"""Generate and repair an extraction adapter against full UniBrain."""

import ast
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CONTRACT = """Write exactly one Python function: adapt(core, image, if_train).
core is the pinned SynthStrip StripModel with unchanged default architecture:
16 base features, 7 levels, maximum 64 features, distance output.
image has shape [1,1,96,96,96]. Pad 16 voxels on each side to 128 cubed,
run core, crop back to 96 cubed, and convert its signed distance to a soft
brain mask with torch.sigmoid(-distance). Do not resize or change the core.
Return ([image * mask], [mask]). Training must preserve gradients from the
registration loss through the extracted image into core parameters and input.
Inference may threshold the mask. No detach, NumPy, no_grad or frozen parameters
during training. Available names are torch and F (torch.nn.functional).
Allowed calls: core, F.pad, torch.sigmoid, and tensor.to. No imports or helpers.
Keep the function concise. Do not read files, execute tools, or change anything
outside this function. The host, original losses and optimizer are immutable.
"""


class AdapterFailure(Exception):
    pass


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def check_code(code):
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Exactly one adapter function is allowed")
    fn = tree.body[0]
    if (fn.name != "adapt" or [arg.arg for arg in fn.args.args] != ["core", "image", "if_train"]
            or fn.decorator_list or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg
            or fn.returns or any(arg.annotation for arg in fn.args.args)):
        raise ValueError("Adapter signature must be adapt(core, image, if_train)")
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.With, ast.AsyncWith,
                 ast.For, ast.While, ast.Try, ast.Global, ast.Nonlocal, ast.Delete,
                 ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.NamedExpr, ast.AugAssign, ast.AnnAssign)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or isinstance(node, ast.FunctionDef) and node is not fn:
            raise ValueError("Unsupported adapter statement")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Private names are not allowed")
        if isinstance(node, ast.Assign) and any(not isinstance(t, ast.Name) or t.id in {"core", "torch", "F"} for t in node.targets):
            raise ValueError("Only local tensor assignments are allowed")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name not in {"core", "F.pad", "torch.sigmoid"} and not (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "to"):
                raise ValueError("Unsupported adapter call: " + name)
    compile(tree, "adapter.py", "exec")
    return tree


def inject_hard_mask(code):
    tree = check_code(code)
    fn = tree.body[0]
    if not isinstance(fn.body[-1], ast.Return):
        raise ValueError("Fault fixture requires a final return")
    result = ast.unparse(fn.body[-1].value)
    fn.body[-1:] = ast.parse(f"result = {result}\nmask = (result[1][0] >= 0.5).to(image.dtype)\nreturn [image * mask], [mask]").body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def run_loop(folder, source, data, python, provider, gpu=0, fault=False, attempts=3, steps=2):
    from .providers import invoke, provider_settings
    from .native import verify_original
    from .sources import verify
    from .upstream import synthstrip_core

    if steps not in (1, 2) or not 1 <= attempts <= 4:
        raise ValueError("Use one or two updates and at most four attempts")
    folder = Path(folder).resolve()
    source, data, python = Path(source).resolve(), Path(data).resolve(), Path(python).resolve()
    if source == folder or source in folder.parents or data == folder or data in folder.parents:
        raise ValueError("Output must be outside the immutable source and data directories")
    folder.mkdir(parents=True, exist_ok=False)
    provenance = {"unibrain": verify_original(source), "synthstrip": verify(["freesurfer_synthstrip"])}
    spec = synthstrip_core()
    # Send source definitions, never imaging arrays or subject records, to the provider.
    from .paths import SOURCES
    text = (SOURCES / "freesurfer_synthstrip/mri_synthstrip/mri_synthstrip").read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    candidate_source = "\n".join("\n".join(lines[row["line"] - 1:row["end_line"]]) for row in spec.evidence)
    config = provider_settings().snapshot(provider)
    schema = {"type": "object", "properties": {"code": {"type": "string"}, "rationale": {"type": "string"}},
              "required": ["code", "rationale"], "additionalProperties": False}
    record = {"status": "running", "profile": "full_unibrain_synthstrip", "provenance": provenance,
              "fault_injection": "hard_mask" if fault else None, "steps_requested": steps, "attempts": []}
    record["implementation_hashes"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("full_replacement.py", "full_replacement_worker.py", "full_model.py")}
    write_json(folder / "report.json", record)
    previous, feedback = "", None
    try:
        for index in range(attempts):
            trial = folder / f"attempt_{index + 1}"
            trial.mkdir()
            record["phase"] = "repairing" if index else "generating"
            record["active_attempt"] = index + 1
            write_json(folder / "report.json", record)
            prompt = CONTRACT + "\nPinned candidate definitions:\n" + candidate_source
            if previous:
                prompt += "\nRepair this adapter using the measured checker feedback.\n" + previous + "\n" + json.dumps(feedback)
            (trial / "prompt.txt").write_text(prompt, encoding="utf-8")
            result, usage = invoke(config, prompt, schema, trial, "repair" if index else "generate")
            code = result["code"]
            (trial / "proposed.py").write_text(code, encoding="utf-8")
            row = {"attempt": index + 1, "provider": config["provider"], "usage": usage,
                   "rationale": result["rationale"], "fault_injected": fault and index == 0}
            record["attempts"].append(row)
            try:
                check_code(code)
                if fault and index == 0:
                    code = inject_hard_mask(code)
                adapter = trial / "adapter.py"
                adapter.write_text(code, encoding="utf-8")
                (trial / "change.diff").write_text("".join(difflib.unified_diff(previous.splitlines(True), code.splitlines(True),
                                                                      fromfile="previous/adapter.py", tofile="adapter.py")), encoding="utf-8")
                row["sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
                record["phase"] = "checking_and_training"
                write_json(folder / "report.json", record)
                command = [str(python), "-m", "unibrain_agent.full_replacement_worker", "--source", str(Path(source).resolve()),
                           "--data", str(Path(data).resolve()), "--adapter", str(adapter), "--output", str(trial),
                           "--gpu", str(gpu), "--steps", str(steps)]
                with (trial / "worker.log").open("w", encoding="utf-8") as stream:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200,
                                             cwd=Path(__file__).resolve().parents[1],
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                report_path = trial / "validation.json"
                feedback = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
                    "status": "infrastructure_failed", "error": "Worker exited without a validation report"}
                row["worker_exit_code"] = process.returncode
            except (SyntaxError, ValueError) as error:
                feedback = {"status": "adapter_failed", "error": str(error)}
            row["validation"] = feedback
            previous = code
            write_json(folder / "report.json", record)
            if feedback["status"] == "passed":
                record["status"] = "passed"
                record["accepted_adapter"] = str(trial / "adapter.py")
                break
            if feedback.get("updates"):
                record["status"] = "training_failed"
                break
            if feedback["status"] != "adapter_failed":
                record["status"] = feedback["status"]
                break
        else:
            record["status"] = "repair_exhausted"
    except Exception as error:
        record.update(status="failed", error=str(error))
    finally:
        record["phase"] = "finished"
        record["source_after"] = verify_original(source)
        write_json(folder / "report.json", record)
    return record
