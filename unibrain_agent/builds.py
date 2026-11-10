import difflib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .agents import Agent
from .catalog import ADAPTATIONS, required_sources, validate_selection, original_selection
from .data import digest
from .paths import BUILDS, PROJECT
from .runner import write_json
from .sources import verify


def build_path(identifier):
    if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", identifier):
        raise ValueError("Invalid build identifier")
    return BUILDS / identifier


def read_build(identifier):
    path = build_path(identifier) / "build.json"
    if not path.exists():
        raise FileNotFoundError("Build not found")
    record = json.loads(path.read_text(encoding="utf-8"))
    progress = path.with_name("progress.json")
    if progress.exists():
        record["progress"] = json.loads(progress.read_text(encoding="utf-8"))
    return record


def create_build(selection, backend):
    if backend not in {"codex", "rules"}:
        raise ValueError("Invalid agent backend")
    # A catalog choice may be incompatible with the current data; record the failure in prepare().
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    folder = build_path(identifier)
    folder.mkdir(parents=True)
    record = {"id": identifier, "status": "queued", "selection": selection, "backend": backend,
              "profile": "original" if original_selection(selection) else "composed",
              "created": datetime.now(timezone.utc).isoformat(), "events": [], "error": None}
    write_json(folder / "build.json", record)
    return record


def generated_code(selection):
    if original_selection(selection):
        return "from unibrain_agent.native import run_original\n\ndef run_pipeline(epochs=2):\n    return run_original(epochs=epochs)\n"
    return ("from unibrain_agent.composition import ComposedPipeline\n\n"
            "SELECTION = " + repr(selection) + "\n\n\ndef build_pipeline():\n    return ComposedPipeline(SELECTION)\n")


def prepare(identifier, on_event=None):
    folder = build_path(identifier)
    record = read_build(identifier)

    def emit(kind, message, **extra):
        event = {"kind": kind, "message": message, "at": datetime.now(timezone.utc).isoformat(), **extra}
        record["events"].append(event)
        write_json(folder / "build.json", record)
        if on_event:
            on_event(event)

    try:
        record["status"] = "preparing"
        emit("build", "Checking stage contracts and pinned source files", phase="contracts")
        selection = validate_selection(record["selection"])
        from .runtime import settings
        record["runtime"] = settings()
        if original_selection(selection):
            from .native import check_original_runtime, verify_original
            record["sources"] = verify_original(record["runtime"]["original_source"])
            emit("build", "Checking the original CUDA runtime and data paths", phase="audit")
            environment = check_original_runtime(record["runtime"])
            record["plan"] = {"rationale": "Run original main.py and train.py with unchanged architecture, losses and optimizer. Only the explicit epoch budget and data IO wrapper differ."}
            record["agent_calls"] = []
            record["validation"] = {"passed": True, "kind": "prerequisites", "environment": environment,
                                     "scope": "Pinned sources, CUDA runtime and input paths checked. Full training is verified during the run; original gradient boundaries remain."}
        else:
            record["sources"] = verify(required_sources(selection))
        agent = Agent(record["backend"], folder / "agent", emit)
        methods = {x: ADAPTATIONS[x] for x in dict.fromkeys(selection.values()) if x in ADAPTATIONS}
        emit("build", "Original settings preserved" if original_selection(selection) else
             "Agent is reviewing the selected tools and connection changes", phase="planning")
        if not original_selection(selection):
            record["plan"] = agent.compose(selection, methods)
            record["agent_calls"] = agent.calls
        code = generated_code(selection)
        compile(code, "pipeline.py", "exec")
        (folder / "pipeline.py").write_text(code, encoding="utf-8")
        record["code"] = code
        record["code_sha256"] = digest(folder / "pipeline.py")
        original = "from unibrain_agent.pipeline import Pipeline\n\ndef build_pipeline():\n    return Pipeline()\n"
        record["diff"] = "".join(difflib.unified_diff(original.splitlines(True), code.splitlines(True),
                                                    fromfile="baseline/pipeline.py", tofile="candidate/pipeline.py"))
        record["implementation_hashes"] = {str(p.relative_to(PROJECT)): digest(p) for p in sorted((PROJECT / "unibrain_agent").glob("*.py"))}
        record["implementation_hashes"]["scripts/run_upstream.py"] = digest(PROJECT / "scripts/run_upstream.py")
        record["implementation_hashes"].update({str(p.relative_to(PROJECT)): digest(p) for p in sorted((PROJECT / "prompts").glob("*.md"))})
        if original_selection(selection):
            record["status"] = "ready"
            emit("ready", "Original UniBrain sources and runtime are ready. Confirm to train.", phase="ready")
            return record
        record["status"] = "validating"
        emit("build", "Code generated. Loading the selected models for two optimizer steps", phase="training")
        with (folder / "validation.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run([sys.executable, "-m", "unibrain_agent.build_worker", str(folder)],
                                       cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=900,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result_path = folder / "validation.json"
        if result_path.exists():
            record["validation"] = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.returncode or not record.get("validation", {}).get("passed"):
            detail = record.get("validation", {}).get("error") or (folder / "validation.log").read_text(encoding="utf-8")[-2000:]
            raise RuntimeError(detail or "The composed training graph failed validation")
        record["status"] = "ready"
        emit("ready", "Training checks passed. Review the generated code, then build and run.")
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        emit("failed", str(error))
    write_json(folder / "build.json", record)
    return record


def verify_build(folder, ready=True):
    folder = Path(folder)
    record = json.loads((folder / "build.json").read_text(encoding="utf-8"))
    if ready and (record["status"] != "ready" or not record.get("validation", {}).get("passed")):
        raise ValueError("This build has not passed validation")
    validate_selection(record["selection"])
    if (folder / "pipeline.py").read_text(encoding="utf-8") != generated_code(record["selection"]):
        raise ValueError("Generated code differs from the registered composition")
    if digest(folder / "pipeline.py") != record["code_sha256"]:
        raise ValueError("Build code changed after preparation")
    for name, expected in record["implementation_hashes"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError("Implementation changed. Prepare the selection again before running.")
    if original_selection(record["selection"]):
        from .native import verify_original
        from .runtime import settings
        verify_original(record.get("runtime", settings())["original_source"])
    else:
        verify(required_sources(record["selection"]))
    return record


def load_pipeline(folder, ready=True):
    verify_build(folder, ready)
    spec = importlib.util.spec_from_file_location("unibrain_generated_pipeline", Path(folder) / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_pipeline()
