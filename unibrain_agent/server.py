import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .paths import RUNS
from .live import RunMonitor
from .report import document, landing, previews, volume_slice
from .access import AccessPolicy


def create_server(port=8776, host="127.0.0.1", access=None):
    from .assistant_workspace import AssistantWorkspace
    from .providers import provider_settings
    from .accounts import login_status, logout
    policy = access or AccessPolicy(host)
    assistant = AssistantWorkspace()
    ai = provider_settings()
    state = {"app": "unibrain_agent", "version": "0.8", "api_version": 1, "running": False, "completed": False,
             "message": "Ready", "error": None, "job": None, "build_id": None}
    lock = threading.Lock()
    monitor = RunMonitor()

    def execute(backend, steps, folder=None, coordination="single"):
        from .runner import run, write_json
        try:
            def event(value, run_folder):
                monitor.record(value)
                with lock:
                    state["message"] = value["message"]
                    state["run_dir"] = str(run_folder.resolve())
            from .builds import verify_build
            build = verify_build(folder) if folder else None
            if build and build.get("profile") == "original":
                from .native import run_original
                result = run_original(epochs=steps, build=build, on_event=event)
            else:
                result = run(backend=backend, steps=steps, on_event=event, build_dir=folder, comparison=False, coordination=coordination)
            result["live"] = monitor.snapshot()
            write_json(Path(result["run_dir"]) / "report.json", result)
            with lock:
                state.update(running=False, completed=True, run_dir=result["run_dir"])
        except Exception as error:
            if monitor.snapshot()["phase"] != "failed":
                monitor.record({"kind": "failed", "message": str(error)})
            with lock:
                state.update(running=False, error=str(error))

    def prepare_build(identifier):
        from .builds import prepare
        def event(value):
            with lock:
                state["message"] = value["message"]
        record = prepare(identifier, event)
        with lock:
            state.update(running=False, message="Build " + record["status"], error=record["error"])

    class Handler(BaseHTTPRequestHandler):
        def respond(self, content, status=200, kind="application/json"):
            if not isinstance(content, bytes):
                content = (json.dumps(content, allow_nan=False) if kind == "application/json" else content).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            origin = self.headers.get("Origin")
            if origin and policy.accepts_origin(origin, self.server.server_port):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()
            self.wfile.write(content)

        def local_request(self, preflight=False):
            actual_port = self.server.server_port
            if self.headers.get("Host") not in policy.hosts(actual_port):
                self.respond({"error": "Host rejected"}, 403)
                return False
            origin = self.headers.get("Origin")
            if not policy.accepts_origin(origin, actual_port):
                self.respond({"error": "Origin rejected"}, 403)
                return False
            if not preflight and urlsplit(self.path).path.startswith("/api/") and not policy.authenticated(self.headers.get("Authorization")):
                self.respond({"error": "Backend access token required or invalid"}, 401)
                return False
            return True

        def do_OPTIONS(self):
            if self.local_request(preflight=True):
                self.respond({"allowed": True})

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            try:
                if path == "/api/status":
                    with lock:
                        snapshot = dict(state)
                    if snapshot["job"] == "run":
                        snapshot["live"] = monitor.snapshot()
                    self.respond(snapshot)
                elif path == "/api/catalog":
                    from .catalog import catalog
                    self.respond(catalog())
                elif path == "/api/report":
                    latest = RUNS / "latest.json"
                    report = None
                    if latest.exists():
                        folder = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"]).resolve()
                        if folder.parent != RUNS.resolve():
                            raise ValueError("Report is outside the configured run directory")
                        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                        report["previews"] = previews(folder)
                    self.respond({"report": report})
                elif path == "/api/runtime":
                    from .runtime import settings
                    self.respond(settings())
                elif path == "/api/ai":
                    self.respond(ai.public())
                elif path == "/api/ai/login":
                    query = parse_qs(urlsplit(self.path).query)
                    self.respond(login_status(query.get("provider", [""])[0]))
                elif path == "/api/assistant":
                    self.respond(assistant.state())
                elif path.startswith("/api/assistant/jobs/"):
                    self.respond(assistant.read(path.removeprefix("/api/assistant/jobs/")))
                elif path == "/api/volume":
                    query = parse_qs(urlsplit(self.path).query)
                    if set(query) != {"run", "branch", "stage", "index"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError("Expected one run, branch, stage and index")
                    identifier = query["run"][0]
                    if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
                        raise ValueError("Invalid run identifier")
                    folder = (RUNS / identifier).resolve()
                    if folder.parent != RUNS.resolve():
                        raise ValueError("Invalid run directory")
                    self.respond(volume_slice(folder, query["branch"][0], query["stage"][0], int(query["index"][0])))
                elif path.startswith("/api/builds/"):
                    from .builds import read_build
                    self.respond(read_build(path.removeprefix("/api/builds/")))
                elif path == "/":
                    self.respond(landing(), kind="text/html")
                else:
                    self.respond({"error": "Not found"}, 404)
            except (ValueError, FileNotFoundError) as error:
                self.respond({"error": str(error)}, 404)

        def do_POST(self):
            if not self.local_request():
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Use application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if self.path in {"/api/ai/settings", "/api/ai/test", "/api/ai/logout"}:
                    with lock, assistant.lock:
                        if state["running"] or assistant.active:
                            self.respond({"error": "Wait for the current build, run or assistant request to finish"}, 409)
                            return
                        if self.path.endswith("logout"):
                            if ai.public()["test"]["status"] == "running":
                                raise ValueError("Wait for the connection test to finish")
                            if body.get("confirm") is not True:
                                raise ValueError("Confirm signing out of the backend CLI account")
                            self.respond(logout(body.get("provider")))
                        else:
                            self.respond(ai.update(body) if self.path.endswith("settings") else ai.start_test(),
                                         200 if self.path.endswith("settings") else 202)
                    return
                if self.path == "/api/assistant":
                    if ai.public()["test"]["status"] == "running":
                        raise ValueError("Wait for the connection test to finish")
                    self.respond(assistant.submit(body), 202)
                    return
                backend, steps = body.get("backend", "configured"), body.get("steps", 5)
                if backend not in {"configured", "codex", "rules"} or type(steps) is not int or not 2 <= steps <= 20:
                    raise ValueError("Invalid backend or step budget")
                folder = None
                coordination = body.get("coordination", "single")
                if coordination not in {"single", "roles"}:
                    raise ValueError("Unknown coordination mode")
                if self.path.startswith("/api/builds/") and self.path.endswith("/run"):
                    from .builds import build_path, verify_build
                    folder = build_path(self.path[len("/api/builds/"):-len("/run")])
                    record = verify_build(folder)
                    backend = record["backend"]
                    if record.get("profile") == "original" and coordination == "roles":
                        raise ValueError("Original reproduction preserves its optimizer. Use a custom pipeline for role coordination.")
                elif self.path == "/api/prepare":
                    from .catalog import DEFAULTS
                    selection = body.get("selection")
                    if not isinstance(selection, dict) or set(selection) != set(DEFAULTS):
                        raise ValueError("Select every stage before preparation")
                elif self.path != "/api/run":
                    self.respond({"error": "Not found"}, 404)
                    return
                with lock:
                    if ai.public()["test"]["status"] == "running":
                        raise ValueError("Wait for the connection test to finish")
                    if state["running"]:
                        self.respond({"error": "A build or experiment is already running"}, 409)
                        return
                    if self.path == "/api/prepare":
                        from .builds import create_build
                        record = create_build(selection, backend)
                        state.update(job="prepare", build_id=record["id"])
                        target, args = prepare_build, (record["id"],)
                    else:
                        monitor.reset(steps)
                        state.update(job="run", build_id=folder.name if folder else None)
                        target, args = execute, (backend, steps, folder, coordination)
                    state.update(running=True, completed=False, message="Starting", error=None, run_dir=None)
                threading.Thread(target=target, args=args, daemon=True).start()
                self.respond({"started": True, "build_id": state["build_id"]}, 202)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                self.respond({"error": str(error)}, 400)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state
    return server


def serve(port=8776, open_browser=False, host="127.0.0.1"):
    server = create_server(port, host)
    print(f"UniBrain Agent: http://127.0.0.1:{server.server_port}", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    server.serve_forever()
