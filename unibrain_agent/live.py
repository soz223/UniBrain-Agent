import copy
import threading
import time

import psutil


class RunMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.sampled_at = 0
        self.resources = {}
        self.reset(0)

    def reset(self, steps):
        with self.lock:
            self.started = time.time()
            self.finished = None
            self.phase = "starting"
            self.device = "Pending"
            self.threads = 4
            self.resources = {}
            self.message = "Starting your pipeline"
            self.steps = steps
            self.history = []
            self.events = []
            self.modules = {}
            self.agents = {}
            self.current_module = None
            self.validation = {"completed": 0, "total": 0}

    def record(self, event):
        with self.lock:
            kind = event["kind"]
            self.message = event["message"]
            if event.get("resources"):
                self.resources = event["resources"]
            if event.get("environment"):
                self.device = event["environment"]["device"]
                self.threads = event["environment"]["threads"]
            if kind == "budget":
                self.steps = event["steps"]
            if kind == "phase":
                self.phase = event["phase"]
                self.current_module = None
            elif kind == "module":
                key = event["module"]
                item = self.modules.setdefault(key, {"module": key, "label": event["label"], "calls": 0, "seconds": 0})
                item["status"] = event["action"]
                if event["action"] == "started":
                    self.current_module = key
                else:
                    item["calls"] += 1
                    item["seconds"] += event["duration"]
                    self.current_module = None
                return
            elif kind == "training":
                self.history.append({key: event[key] for key in ("step", "weighted_loss", "losses", "weights")})
            elif kind in {"agent", "decision"}:
                role = event.get("role", event["message"])
                self.agents[role] = {"role": role, "status": "running" if kind == "agent" else "completed",
                                     "seconds": event.get("duration"), "usage": event.get("usage"),
                                     "decision": event.get("decision")}
                if kind == "agent":
                    self.current_module = None
            elif kind == "validation_progress":
                self.validation = {"completed": event["completed"], "total": event["total"]}
            if kind in {"complete", "failed"}:
                self.phase = kind
                self.finished = time.time()
                self.current_module = None
            self.events.append({key: value for key, value in event.items() if key not in {"subject_id", "config"}})
            self.events = self.events[-120:]

    def snapshot(self):
        with self.lock:
            now = time.time()
            external = self.resources.get("scope") == "Original UniBrain worker process"
            if not external and (not self.resources or (not self.finished and now - self.sampled_at >= 1)):
                self.resources = {"cpu_percent": round(self.process.cpu_percent() / (psutil.cpu_count() or 1), 1),
                                  "rss_mb": round(self.process.memory_info().rss / 1024 ** 2, 1),
                                  "device": self.device.upper(), "threads": self.threads, "scope": "Python worker process"}
                self.sampled_at = now
            return copy.deepcopy({"phase": self.phase, "message": self.message, "started_at": self.started,
                                  "elapsed_seconds": round((self.finished or now) - self.started, 1),
                                  "steps": self.steps, "history": self.history, "events": self.events,
                                  "modules": list(self.modules.values()), "agents": list(self.agents.values()),
                                  "current_module": self.current_module, "validation": self.validation,
                                  "resources": self.resources})
