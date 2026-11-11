"""Summarize observed adapter repair evidence without copying private traces."""

import difflib
import hashlib
import json
import math
import re
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_records(records):
    result = []
    for record in records:
        repository, commit = record.get("repository", ""), record.get("commit", "")
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Source provenance must use a public repository URL")
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("Source provenance must identify a full commit")
        result.append({"repository": repository, "commit": commit,
                       "verified_clean": record.get("verified_clean") is True})
    return result


def numeric_fields(value, fields):
    return {key: value[key] for key in fields if type(value.get(key)) in (int, float, bool)
            and math.isfinite(value[key])}


def summarize_trace(directory):
    directory = Path(directory)
    root_path = directory / "report.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    declared = root.get("attempts", [])
    if not declared:
        raise ValueError("A real integration report with recorded attempts is required")
    attempts, transitions, previous_code = [], [], None
    for index, entry in enumerate(declared, 1):
        if entry.get("attempt") != index:
            raise ValueError("Attempt numbers must be consecutive and preserve failed attempts")
        folder = directory / f"attempt_{index}"
        adapter, validation = folder / "adapter.py", folder / "validation.json"
        value = json.loads(validation.read_text(encoding="utf-8"))
        adapter_hash = sha256(adapter)
        if adapter_hash != entry.get("sha256") or adapter_hash != value.get("adapter_sha256"):
            raise ValueError("Adapter hash does not match retained validation evidence")
        if entry.get("validation") != value:
            raise ValueError("Root and attempt validation evidence disagree")
        updates = value.get("updates", [])
        if not isinstance(updates, list) or any(update.get("step") != step for step, update in enumerate(updates, 1)):
            raise ValueError("Optimizer update records are incomplete or unordered")
        budget = root.get("steps_requested")
        if type(budget) is not int or budget < 1 or len(updates) > budget:
            raise ValueError("Observed update count exceeds the declared budget")
        gradient = numeric_fields(value.get("gradient_check", {}), (
            "registration_to_extraction_norm", "connected_tensors", "total_tensors",
            "mask_requires_grad", "input_gradient_norm"))
        status = value.get("status")
        if status not in {"passed", "adapter_failed", "infrastructure_failed"}:
            raise ValueError("Unsupported validation status")
        if status == "passed" and (not updates or not gradient.get("mask_requires_grad")
                                   or gradient.get("connected_tensors", 0) < 1):
            raise ValueError("Passing report lacks actual gradient and update evidence")
        if status == "passed" and any(update.get("finite_gradients") is not True for update in updates):
            raise ValueError("Passing report lacks finite optimizer gradient evidence")
        checkpoint = value.get("checkpoint", {})
        checkpoint_hash = None
        if checkpoint:
            name = checkpoint.get("file", "")
            if Path(name).name != name or not name:
                raise ValueError("Checkpoint must be retained inside its attempt directory")
            checkpoint_hash = sha256(folder / name)
            if checkpoint_hash != checkpoint.get("sha256"):
                raise ValueError("Checkpoint no longer matches its observed hash")
        row = {"attempt": index, "status": status, "adapter_sha256": adapter_hash,
               "validation_sha256": sha256(validation), "checkpoint_sha256": checkpoint_hash,
               "optimizer_updates": len(updates), "gradient": gradient,
               "source_before": source_records(value.get("source_before", [])),
               "candidate_source": source_records(value.get("candidate_source", [])),
               "source_after": source_records(value.get("source_after", [])),
               "fault_injected": entry.get("fault_injected") is True,
               "provider": entry.get("provider") if entry.get("provider") in {"codex", "claude", "openai", "anthropic", "openai_compatible"} else "unspecified",
               "token_usage": numeric_fields(entry.get("usage", {}), ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")),
               "parameter_updates": [{"step": u["step"], "changed_parameter_tensors":
                                      {k: v for k, v in u.get("changed_parameter_tensors", {}).items()
                                       if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) and type(v) is int and v >= 0},
                                      "finite_gradients": u.get("finite_gradients") is True} for u in updates]}
        if value.get("error"):
            row["error_sha256"] = hashlib.sha256(str(value["error"]).encode()).hexdigest()
            row["failure_category"] = ("gradient_disconnection" if gradient.get("mask_requires_grad") is False
                                       or gradient.get("connected_tensors") == 0 else "other_validation_failure")
        code = adapter.read_text(encoding="utf-8").splitlines()
        if previous_code is not None:
            change = list(difflib.ndiff(previous_code, code))
            repair = folder / "repair.json"
            transitions.append({"from_attempt": index - 1, "to_attempt": index,
                                "added_lines": sum(line.startswith("+ ") for line in change),
                                "removed_lines": sum(line.startswith("- ") for line in change),
                                "repair_record_present": repair.is_file(),
                                "repair_record_sha256": sha256(repair) if repair.is_file() else None})
        previous_code = code
        attempts.append(row)
    before, after = attempts[0], attempts[-1]
    repaired = before["status"] == "adapter_failed" and after["status"] == "passed"
    return {"status": "complete", "scope": "Retrospective audit of retained validation evidence; no model or Agent execution",
            "root_report_sha256": sha256(root_path), "controlled_fault_injection": any(a["fault_injected"] for a in attempts),
            "attempts": attempts, "transitions": transitions, "repaired": repaired,
            "metrics": {"attempt_count": len(attempts), "optimizer_updates": sum(a["optimizer_updates"] for a in attempts),
                        "observed_repair": int(repaired)},
            "boundary": "A bounded repair case is not a cohort accuracy result or a natural failure rate estimate"}


def write_trace(directory, output):
    result = summarize_trace(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
