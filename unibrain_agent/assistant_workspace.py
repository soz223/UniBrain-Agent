import copy
import html
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from uuid import uuid4

from .agents import Agent, object_schema, balance_weights
from .catalog import STAGES, ADAPTATIONS, DEFAULTS, validate_selection
from .paths import CATALOG, PROJECT, RUNS, configured_path
from .roles import prompt_for, role_catalog, run_team, measured_tasks

ASSISTANT = configured_path("assistant", PROJECT / "assistant_sessions")
RECOMMENDATION = object_schema({name: {"type": "string"} for name in
                                ("source_id", "stage", "tool_id", "reason", "verification_needed")})
RESEARCH_SCHEMA = object_schema({"answer": {"type": "string"},
    "source_ids": {"type": "array", "items": {"type": "string"}},
    "recommendations": {"type": "array", "maxItems": 3, "items": RECOMMENDATION},
    "next_question": {"type": "string"}})


def library_sources(stage):
    records = json.loads((CATALOG / "research.json").read_text(encoding="utf-8"))["records"]
    allowed = {item[0]: item[3] for item in STAGES}
    sources = []
    for item in records:
        stages = [key for key, choices in allowed.items() if item["id"] in choices]
        if stage != "all" and stage not in stages:
            continue
        sources.append({**item, "source_id": item["id"], "tool_id": item["id"], "allowed_stages": stages,
                        "current_adapter": ADAPTATIONS[item["id"]], "origin": "Project catalog",
                        "verification": "Historical core checks; current composition still requires validation"})
    stages = [key for key, choices in allowed.items() if "unibrain" in choices]
    if stage == "all" or stage in stages:
        sources.append({"source_id": "U01", "tool_id": "unibrain", "allowed_stages": stages,
                        "publication": "UniBrain (2024, KDD)", "origin": "Project reference",
                        "paper": "https://github.com/Anonymous7852/UniBrain",
                        "code": "https://github.com/Anonymous7852/UniBrain/blob/f9f8d906b1725046d532f862a28876482aca2cfa/model.py",
                        "interface": "96 cubed structured sMRI input, image tasks, ROI graph and diagnosis",
                        "limitations": "Original atlas sampling and graph augmentation retain discrete gradient boundaries.",
                        "verification": "Original execution is audited separately from differentiable adapter composition"})
    return sources


def search_publications(stage, question):
    title = next((item[1] for item in STAGES if item[0] == stage), "preprocessing")
    keywords = " ".join(re.findall(r"[A-Za-z][A-Za-z0-9]+", question))[:150]
    query = f"brain MRI deep learning {title} {keywords}".strip()
    url = "https://api.crossref.org/works?" + urlencode({"query.bibliographic": query, "rows": 6})
    request = Request(url, headers={"User-Agent": "UniBrainAgent/0.6 (https://github.com/soz223/UniBrain-Agent)", "Accept": "application/json"})
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)
    records = []
    for item in payload["message"]["items"]:
        if not item.get("DOI") or not item.get("title"):
            continue
        year = item.get("published", item.get("issued", {})).get("date-parts", [[None]])[0][0]
        title = html.unescape(re.sub(r"<[^>]+>", "", item["title"][0]))
        abstract = html.unescape(re.sub(r"<[^>]+>", " ", item.get("abstract", "")))[:4000]
        venue = (item.get("container-title") or ["Venue not supplied"])[0]
        records.append({"source_id": "X" + str(len(records) + 1).zfill(2), "tool_id": "", "allowed_stages": [],
                        "publication": f"{title} ({year or 'Year unavailable'}, {venue})", "paper": "https://doi.org/" + quote(item["DOI"], safe="/"),
                        "code": "", "abstract": abstract, "origin": "Crossref live metadata",
                        "verification": "Publication lead only; implementation and trainability not checked"})
    return records, query


def research_result(result, sources):
    lookup = {source["source_id"]: source for source in sources}
    if not set(result["source_ids"]) <= set(lookup):
        raise ValueError("Research response cited an unknown source")
    stages = {item[0] for item in STAGES}
    for item in result["recommendations"]:
        source = lookup.get(item["source_id"])
        if source is None or item["stage"] not in stages:
            raise ValueError("Research recommendation has an unsupported source or stage")
        if item["tool_id"] and (item["tool_id"] != source["tool_id"] or item["stage"] not in source["allowed_stages"]):
            raise ValueError("Research recommendation cannot select an unregistered tool")
    return result


def latest_context():
    pointer = RUNS / "latest.json"
    if not pointer.exists():
        return {"scope": "No completed training run", "diagnostic": None, "weights": {"extraction": 1., "registration": .1, "segmentation": 1., "classification": 1.}}
    folder = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"]).resolve()
    if folder.parent != RUNS.resolve():
        raise ValueError("Saved run is outside the configured output directory")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    model = report.get("candidate", {})
    context = {"run_id": folder.name, "profile": report.get("profile", "composed"), "scope": "Saved run review; no training changes are applied",
               "diagnostic": model.get("gradients"), "spatial": model.get("spatial"),
               "weights": (model.get("history") or [{}])[-1].get("weights", {"extraction": 1., "registration": .1, "segmentation": 1., "classification": 1.}),
               "validation": {key: value for key, value in model.get("validation", {}).items() if key in {"count", "extraction_dice", "tissue_dice", "accuracy", "reference_loss"}},
               "training_seconds": model.get("training_seconds"), "steps": model.get("steps")}
    if report.get("native"):
        context["observed_module_updates"] = sorted({name for update in report["native"]["updates"] for name in update["modules_with_observed_updates"]})
        context["boundary"] = "Parameter updates were observed. Per-task gradient norms and pairwise conflicts were not measured in this original run."
    return context


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class AssistantWorkspace:
    def __init__(self, folder=ASSISTANT, backend="codex"):
        self.folder, self.backend = Path(folder), backend
        self.lock = threading.Lock()
        self.active = None

    def path(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", identifier):
            raise ValueError("Invalid assistant job identifier")
        return self.folder / identifier / "job.json"

    def read(self, identifier):
        with self.lock:
            record = json.loads(self.path(identifier).read_text(encoding="utf-8"))
            if record["status"] == "running" and self.active != identifier:
                record.update(status="failed", error="The service restarted. Submit the request again.")
            return record

    def state(self):
        latest = self.folder / "latest.json"
        index = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
        return {"roles": role_catalog(), "research_prompt": prompt_for("research"),
                "latest": {key: self.read(identifier) for key, identifier in index.items()}, "context": latest_context()}

    def submit(self, body):
        mode, message, stage = body.get("mode"), body.get("message", ""), body.get("stage", "all")
        if not isinstance(mode, str) or mode not in {"research", "roles"} or not isinstance(message, str) or len(message) > 4000:
            raise ValueError("Invalid assistant request")
        if mode == "research" and not message.strip():
            raise ValueError("Enter a research question")
        role = body.get("role", "all")
        if not isinstance(stage, str) or stage not in {"all", *[item[0] for item in STAGES]} or not isinstance(role, str) or role not in {"all", "coordinator", "extraction", "registration", "segmentation", "classification"}:
            raise ValueError("Unknown stage or role")
        if not isinstance(body.get("selection", DEFAULTS), dict):
            raise ValueError("Expected a pipeline selection object")
        selection = validate_selection(body.get("selection", DEFAULTS))
        previous = self.read(body["previous"]) if body.get("previous") else None
        if previous and (previous["mode"] != "research" or previous["status"] != "complete"):
            raise ValueError("Continue a completed research conversation")
        with self.lock:
            if self.active:
                raise ValueError("An assistant request is already running")
            identifier = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            path = self.path(identifier)
            path.parent.mkdir(parents=True)
            record = {"id": identifier, "mode": mode, "status": "running", "created": datetime.now(timezone.utc).isoformat(),
                      "message": message, "stage": stage, "role": body.get("role", "all"), "selection": selection,
                      "online": body.get("online", True) is True, "backend": self.backend, "events": [], "sources": [],
                      "previous": body.get("previous"),
                      "history": (previous["history"] + [{"question": previous["message"], "answer": previous["result"]["answer"],
                                   "source_ids": list(set(previous["result"]["source_ids"] + [item["source_id"] for item in previous["result"]["recommendations"]])),
                                   "sources": [{k:v for k,v in source.items() if k in {"source_id", "publication", "paper", "code", "origin", "verification"}} for source in previous["sources"]]}])[-6:] if previous else [],
                      "error": None, "result": None}
            save(path, record)
            latest = self.folder / "latest.json"
            index = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
            save(latest, {**index, mode: identifier})
            self.active = identifier
        threading.Thread(target=self.execute, args=(record,), daemon=True).start()
        return {"started": True, "id": identifier}

    def execute(self, record):
        path = self.path(record["id"])
        def emit(kind, message, **extra):
            with self.lock:
                record["events"].append({"kind": kind, "message": message, "at": datetime.now(timezone.utc).isoformat(), **extra})
                save(path, record)
        agent = Agent(self.backend, path.parent / "agent", emit)
        try:
            if record["mode"] == "research":
                emit("phase", "Reading the project literature and adapter catalog")
                sources = library_sources(record["stage"])
                if record["online"]:
                    emit("phase", "Searching publication metadata through Crossref")
                    try:
                        online, query = search_publications(record["stage"], record["message"])
                        sources.extend(online)
                        record["search"] = {"query": query, "provider": "Crossref", "results": len(online), "status": "complete"}
                    except Exception as error:
                        record["search"] = {"provider": "Crossref", "status": "failed", "error": str(error)}
                        emit("warning", "Online retrieval failed. The response will use the project catalog only.")
                else:
                    record["search"] = {"status": "disabled", "provider": "Project catalog"}
                record["sources"] = sources
                context = {"question": record["message"], "conversation": record["history"], "stage": record["stage"],
                           "citation_scope": "Historical source IDs belong to their own turns. Cite current supplied sources in the new response.",
                           "selection": record["selection"], "sources": sources, "search": record["search"],
                           "allowed_stages": [item[0] for item in STAGES]}
                result = agent.decide("research", context, RESEARCH_SCHEMA,
                                      {"answer": "Deterministic test response; no language model used.", "source_ids": [], "recommendations": [], "next_question": ""},
                                      instructions=prompt_for("research"))
                record["result"] = research_result(result, sources)
            else:
                record["context"] = latest_context()
                emit("phase", "Loading aggregate measurements for an advisory role review")
                record["result"] = run_team(agent, {**record["context"], "question": record["message"], "mode": "advisory"}, record["role"])
                result, context = record["result"], record["context"]
                result["weights_before"] = context["weights"]
                result["weights_proposed"] = balance_weights(context["weights"], context["diagnostic"]["task_norms"]) if result["action"] == "balance_gradients" and measured_tasks(context) else context["weights"]
                result["applied"] = False
            record.update(status="complete", calls=copy.deepcopy(agent.calls))
            emit("complete", "Research response saved" if record["mode"] == "research" else "Role review saved. Training weights were not changed.")
        except Exception as error:
            record.update(status="failed", error=str(error))
            emit("failed", str(error))
        finally:
            with self.lock:
                save(path, record)
                self.active = None
