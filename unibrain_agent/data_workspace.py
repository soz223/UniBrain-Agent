import copy
import json
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .data_builder import audit_dataset, build_dataset, inspect_sources, model_contract, read_manifest, eligible_subjects
from .paths import DATA, PROJECT, configured_path

DATA_JOBS = configured_path("data_jobs", PROJECT / "data_jobs")


class DataWorkspace:
    def __init__(self, folder=DATA_JOBS):
        self.folder = Path(folder)
        self.lock = threading.RLock()
        self.active = None
        pointer = self.folder / "workspace.json"
        saved = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else {}
        self.latest = saved.get("latest")
        self.selected = saved.get("selected")

    def save_index(self):
        with self.lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / "workspace.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"latest": self.latest, "selected": self.selected}), encoding="utf-8")
            temporary.replace(path)

    def path(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", identifier):
            raise ValueError("Invalid data job ID")
        return self.folder / identifier / "job.json"

    def read(self, identifier):
        with self.lock:
            record = json.loads(self.path(identifier).read_text(encoding="utf-8"))
            if record["status"] == "running" and self.active != identifier:
                record.update(status="failed", error="Service restarted; submit the data job again")
            return record

    def state(self):
        with self.lock:
            return {"contract": model_contract(), "default_source": str(DATA), "selected": self.selected,
                    "job": self.read(self.latest) if self.latest else None}

    def submit(self, body):
        from .catalog import DEFAULTS, validate_selection
        mode = body.get("mode")
        if mode not in {"inspect", "discuss", "build"}:
            raise ValueError("Unknown data action")
        paths = body.get("sources") or [str(DATA)]
        if not isinstance(paths, list) or len(paths) > 20 or any(not isinstance(p, str) or len(p) > 2000 for p in paths):
            raise ValueError("Supply at most 20 source paths on the backend")
        message = body.get("message", "")
        if not isinstance(message, str) or len(message) > 4000:
            raise ValueError("Keep the data question within 4000 characters")
        selection = validate_selection(body.get("selection", DEFAULTS))
        with self.lock:
            if self.active:
                raise ValueError("A data job is already running")
            identifier = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            record = {"id": identifier, "mode": mode, "status": "running", "sources": paths,
                      "manifest": body.get("manifest", ""), "message": message, "selection": selection,
                      "previous": self.latest, "events": [], "result": None, "error": None, "calls": []}
            if not isinstance(record["manifest"], str) or len(record["manifest"]) > 2000:
                raise ValueError("Invalid manifest path")
            self.path(identifier).parent.mkdir(parents=True)
            self.active = self.latest = identifier
            self.save(record)
            self.save_index()
        threading.Thread(target=self.execute, args=(record,), daemon=True).start()
        return {"id": identifier, "started": True}

    def save(self, record):
        with self.lock:
            path = self.path(record["id"])
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, allow_nan=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def execute(self, record):
        def emit(message):
            record["events"].append({"at": datetime.now(timezone.utc).isoformat(), "message": message})
            self.save(record)
        try:
            if record["mode"] == "build":
                emit("Reading the manifest and selected pipeline contract")
                manifest = read_manifest(record["manifest"])
                destination = self.path(record["id"]).parent / "dataset"
                audit = build_dataset(manifest, destination, emit)
                record["result"] = {"dataset": str(destination.resolve()), "audit": audit}
            else:
                emit("Inspecting backend data sources against the model input contract")
                inventory = inspect_sources(record["sources"])
                record["inventory"] = inventory
                if record["manifest"]:
                    accepted, excluded = eligible_subjects(read_manifest(record["manifest"]))
                    inventory["manifest"] = {"eligible": len(accepted), "excluded": len(excluded),
                                             "reasons": dict(Counter(reason for row in excluded for reason in row["reasons"]))}
                record["result"] = inventory
                if record["mode"] == "discuss":
                    from .agents import Agent, object_schema
                    agent = Agent("configured", self.path(record["id"]).parent / "agent", lambda kind, message, **extra: emit(message))
                    schema = object_schema({"answer": {"type": "string"}, "plan": {"type": "array", "items": {"type": "string"}},
                                            "blockers": {"type": "array", "items": {"type": "string"}}, "model_changes": {"type": "string"}})
                    context = {"question": record["message"], "contract": model_contract(record["selection"]),
                               "inventory_counts": inventory["counts"], "manifest_summary": inventory.get("manifest"),
                               "ready_datasets": [item["audit"]["splits"] for item in inventory["items"] if item.get("audit")],
                               "dicom_converter": inventory["dcm2niix_available"]}
                    if self.selected:
                        context["attached_dataset_audit"] = {key: value for key, value in self.selected["audit"].items()
                                                             if key in {"passed", "splits", "checks", "warnings", "physical_geometry", "excluded"}}
                    if inventory.get("manifest"):
                        manifest = read_manifest(record["manifest"])
                        context["declared_labels"] = manifest["labels"]
                        context["supervision_counts"] = dict(Counter(row.get("supervision", "unspecified") for row in manifest["subjects"]))
                    if record["previous"]:
                        previous = self.read(record["previous"])
                        if previous["mode"] == "build" and previous["status"] == "complete":
                            context["last_conversion_audit"] = {key: value for key, value in previous["result"]["audit"].items() if key != "files"}
                        if previous["mode"] == "discuss" and previous["status"] == "complete":
                            context["previous_answer"] = previous["result"]
                    record["result"] = agent.decide("data_builder", context, schema, {}, instructions=(PROJECT / "prompts/data_builder.md").read_text(encoding="utf-8"))
                    record["calls"] = copy.deepcopy(agent.calls)
            record["status"] = "complete"
            emit("Data job complete; inspect the evidence before attaching a dataset")
        except Exception as error:
            record.update(status="failed", error=str(error))
            emit(str(error))
        finally:
            with self.lock:
                self.save(record)
                self.active = None

    def use(self, identifier, dataset=None):
        record = self.read(identifier)
        if record["status"] != "complete":
            raise ValueError("Complete the data checks before attaching a dataset")
        candidates = [item["path"] for item in record.get("inventory", {}).get("items", []) if item.get("status") == "ready"]
        if record["mode"] == "build":
            candidates.append(record["result"]["dataset"])
        if dataset not in candidates:
            raise ValueError("Select a dataset verified by this data job")
        audit = audit_dataset(dataset)
        with self.lock:
            self.selected = {"job": identifier, "path": dataset, "audit": audit}
            self.save_index()
        return self.selected
