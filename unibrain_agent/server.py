import json
import re
import threading
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .paths import RUNS
from .live import RunMonitor
from .report import document, landing, previews, volume_slice
from .access import AccessPolicy


def create_server(port=8777, host="127.0.0.1", access=None):
    from .assistant_workspace import AssistantWorkspace
    from .providers import provider_settings
    from .accounts import login_status, logout
    from .data_workspace import DataWorkspace
    from .cycle_workspace import CycleWorkspace
    policy = access or AccessPolicy(host)
    assistant = AssistantWorkspace()
    ai = provider_settings()
    data_workspace = DataWorkspace()
    cycles = CycleWorkspace()
    ai_sessions = {}
    state = {"app": "unibrain_agent", "version": "0.9", "api_version": 2, "running": False, "completed": False,
             "message": "Ready", "error": None, "job": None, "build_id": None}
    lock = threading.RLock()
    monitor = RunMonitor()

    def execute_cycle(identifier, configuration):
        try:
            cycles.execute(identifier, configuration, ai.path)
        finally:
            record = cycles.read(identifier)
            with lock:
                state.update(running=False, completed=record["status"] in {"ready", "complete"},
                             message="Full research cycle: " + record["status"], error=record.get("error"))

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
                result = run_original(epochs=steps, build=build, on_event=event, coordination=coordination)
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
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-UniBrain-AI-Session")
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

        def require_ai(self):
            with lock:
                grant = ai_sessions.get(self.headers.get("X-UniBrain-AI-Session"))
                if not grant or grant["expires"] < time.monotonic() or grant["provider"] != ai.selected:
                    raise PermissionError("Connect your AI account in AI settings for this browser session")

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
                    elif snapshot["job"] == "research_cycle" and snapshot.get("cycle_id"):
                        snapshot["research_cycle"] = cycles.read(snapshot["cycle_id"])
                    self.respond(snapshot)
                elif path == "/api/research-cycles":
                    from .runtime import settings
                    self.respond({"defaults": cycles.defaults(settings(), data_workspace.selected),
                                  "job": cycles.read(cycles.latest) if cycles.latest else None})
                elif path.startswith("/api/research-cycles/"):
                    identifier = path.removeprefix("/api/research-cycles/")
                    self.respond(cycles.code(identifier[:-5]) if identifier.endswith("/code") else cycles.read(identifier))
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
                    self.require_ai()
                    query = parse_qs(urlsplit(self.path).query)
                    self.respond(login_status(query.get("provider", [""])[0]))
                elif path == "/api/data":
                    self.respond(data_workspace.state())
                elif path.startswith("/api/data/jobs/"):
                    self.respond(data_workspace.read(path.removeprefix("/api/data/jobs/")))
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
                    from .builds import read_build, review_code
                    identifier = path.removeprefix("/api/builds/")
                    self.respond(review_code(identifier[:-5]) if identifier.endswith("/code") else read_build(identifier))
                elif path == "/":
                    self.respond(landing(), kind="text/html")
                else:
                    self.respond({"error": "Not found"}, 404)
            except PermissionError as error:
                self.respond({"error": str(error)}, 403)
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
                if self.path == "/api/objective/preview":
                    from .objective import preview_objective
                    from .builds import read_build
                    build = read_build(body['build_id']) if body.get('build_id') else None
                    if (build and build.get('status') == 'ready' and build.get('selection') == body['selection']
                            and build.get('coordination', 'single') == body.get('coordination', 'single')
                            and build.get('validation', {}).get('objective')):
                        self.respond(build['validation']['objective'])
                        return
                    self.respond(preview_objective(body['selection'], body.get('coordination', 'single'), build))
                    return
                if self.path == "/api/research-cycles":
                    self.require_ai()
                    from .runtime import settings
                    with lock, assistant.lock:
                        if state["running"] or assistant.active or data_workspace.active or ai.public()["test"]["status"] == "running":
                            self.respond({"error": "Wait for the active workspace operation to finish"}, 409)
                            return
                        configuration = ai.snapshot()
                        record = cycles.create(body, settings(), data_workspace.selected, configuration)
                        state.update(running=True, completed=False, job="research_cycle", cycle_id=record["id"],
                                     message="Starting full research cycle", error=None, build_id=None, run_dir=None)
                    threading.Thread(target=execute_cycle, args=(record["id"], configuration), daemon=True).start()
                    self.respond(record, 202)
                    return
                if self.path in {"/api/ai/settings", "/api/ai/test", "/api/ai/logout", "/api/ai/connect", "/api/ai/disconnect"}:
                    with lock, assistant.lock:
                        if state["running"] or assistant.active or data_workspace.active:
                            self.respond({"error": "Wait for the current build, run or assistant request to finish"}, 409)
                            return
                        if self.path.endswith("connect") and not self.path.endswith("disconnect"):
                            if body.get("confirm") is not True:
                                raise ValueError("Confirm using the selected account on this backend")
                            provider = ai.selected
                            account = login_status(provider) if provider in {"codex", "claude"} else {"authenticated": bool(ai.snapshot()["api_key"])}
                            if not account["authenticated"]:
                                raise ValueError("Sign in with the official CLI on your backend or configure your own API key first")
                            now = time.monotonic()
                            for key in list(ai_sessions):
                                if ai_sessions[key]["expires"] < now:
                                    del ai_sessions[key]
                            if len(ai_sessions) >= 64:
                                raise ValueError("Too many connected AI sessions; disconnect an existing session")
                            token = secrets.token_urlsafe(32)
                            ai_sessions[token] = {"provider": provider, "expires": now + 8 * 3600}
                            self.respond({"session": token, "provider": provider, "account": account})
                        elif self.path.endswith("disconnect"):
                            ai_sessions.pop(self.headers.get("X-UniBrain-AI-Session"), None)
                            self.respond({"disconnected": True})
                        elif self.path.endswith("logout"):
                            if ai.public()["test"]["status"] == "running":
                                raise ValueError("Wait for the connection test to finish")
                            if body.get("confirm") is not True:
                                raise ValueError("Confirm signing out of the backend CLI account")
                            self.require_ai()
                            self.respond(logout(body.get("provider")))
                            ai_sessions.clear()
                        else:
                            if self.path.endswith("test"):
                                self.require_ai()
                            self.respond(ai.update(body) if self.path.endswith("settings") else ai.start_test(),
                                         200 if self.path.endswith("settings") else 202)
                            if self.path.endswith("settings"):
                                ai_sessions.clear()
                    return
                if self.path in {"/api/data/jobs", "/api/data/use"}:
                    if self.path.endswith("jobs") and body.get("mode") == "discuss":
                        self.require_ai()
                    with lock, assistant.lock:
                        if state["running"] or assistant.active or ai.public()["test"]["status"] == "running":
                            raise ValueError("Wait for the active workspace operation to finish")
                        if self.path.endswith("use"):
                            self.respond(data_workspace.use(body.get("id"), body.get("dataset")))
                        else:
                            self.respond(data_workspace.submit(body), 202)
                    return
                if self.path == "/api/assistant":
                    self.require_ai()
                    if ai.public()["test"]["status"] == "running":
                        raise ValueError("Wait for the connection test to finish")
                    self.respond(assistant.submit(body), 202)
                    return
                backend, steps = body.get("backend", "configured"), body.get("steps", 5)
                if backend not in {"configured", "codex", "rules"} or type(steps) is not int or not 2 <= steps <= 1000:
                    raise ValueError("Invalid backend or step budget")
                folder = None
                coordination = body.get("coordination", "single")
                if coordination not in {"single", "roles", "taskforce"}:
                    raise ValueError("Unknown coordination mode")
                if self.path.startswith("/api/builds/") and self.path.endswith("/run"):
                    from .builds import build_path, verify_build
                    folder = build_path(self.path[len("/api/builds/"):-len("/run")])
                    record = verify_build(folder)
                    backend = record["backend"]
                    if coordination != record.get("coordination", "single"):
                        raise ValueError("Training coordination changed after preparation. Prepare the pipeline again.")
                    if record.get("profile") != "original" and backend != "rules":
                        self.require_ai()
                    if record.get("profile") == "original" and coordination == "roles":
                        raise ValueError("Prompt roles are available for custom runs; use taskforce for original UniBrain.")
                    if record.get("profile") != "original" and coordination == "taskforce":
                        raise ValueError("The taskforce gradient policy requires the full original UniBrain.")
                elif self.path == "/api/prepare":
                    from .catalog import DEFAULTS
                    selection = body.get("selection")
                    if not isinstance(selection, dict) or set(selection) != set(DEFAULTS):
                        raise ValueError("Select every stage before preparation")
                    if selection != DEFAULTS and backend != "rules":
                        self.require_ai()
                elif self.path != "/api/run":
                    self.respond({"error": "Not found"}, 404)
                    return
                elif backend != "rules":
                    self.require_ai()
                with lock:
                    if ai.public()["test"]["status"] == "running":
                        raise ValueError("Wait for the connection test to finish")
                    if state["running"] or data_workspace.active:
                        self.respond({"error": "A build or experiment is already running"}, 409)
                        return
                    if self.path == "/api/prepare":
                        from .builds import create_build
                        record = create_build(selection, backend, dataset=data_workspace.selected,
                                              coordination=coordination)
                        state.update(job="prepare", build_id=record["id"])
                        target, args = prepare_build, (record["id"],)
                    else:
                        monitor.reset(steps)
                        state.update(job="run", build_id=folder.name if folder else None)
                        target, args = execute, (backend, steps, folder, coordination)
                    state.update(running=True, completed=False, message="Starting", error=None, run_dir=None)
                threading.Thread(target=target, args=args, daemon=True).start()
                self.respond({"started": True, "build_id": state["build_id"]}, 202)
            except PermissionError as error:
                self.respond({"error": str(error)}, 403)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                self.respond({"error": str(error)}, 400)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state
    return server


def serve(port=8777, open_browser=False, host="127.0.0.1"):
    server = create_server(port, host)
    print(f"UniBrain Agent: http://127.0.0.1:{server.server_port}", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    server.serve_forever()
