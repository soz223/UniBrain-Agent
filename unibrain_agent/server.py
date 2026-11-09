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


def create_server(port=8771):
    state = {"app": "unibrain_agent", "version": "0.5", "running": False, "completed": False,
             "message": "Ready", "error": None, "job": None, "build_id": None}
    lock = threading.Lock()
    monitor = RunMonitor()

    def execute(backend, steps, folder=None):
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
                result = run(backend=backend, steps=steps, on_event=event, build_dir=folder, comparison=False)
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
            self.end_headers()
            self.wfile.write(content)

        def local_request(self):
            actual_port = self.server.server_port
            allowed = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
            if self.headers.get("Host") not in allowed:
                self.respond({"error": "Host rejected"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in allowed}:
                self.respond({"error": "Origin rejected"}, 403)
                return False
            return True

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
                elif path == "/api/runtime":
                    from .runtime import settings
                    self.respond(settings())
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
                    latest = RUNS / "latest.json"
                    if latest.exists():
                        folder = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"])
                        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                        self.respond(document({**report, "previews": previews(folder)}), kind="text/html")
                    else:
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
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                backend, steps = body.get("backend", "codex"), body.get("steps", 5)
                if backend not in {"codex", "rules"} or type(steps) is not int or not 2 <= steps <= 20:
                    raise ValueError("Invalid backend or step budget")
                folder = None
                if self.path.startswith("/api/builds/") and self.path.endswith("/run"):
                    from .builds import build_path, verify_build
                    folder = build_path(self.path[len("/api/builds/"):-len("/run")])
                    record = verify_build(folder)
                    backend = record["backend"]
                elif self.path == "/api/prepare":
                    from .catalog import DEFAULTS
                    selection = body.get("selection")
                    if not isinstance(selection, dict) or set(selection) != set(DEFAULTS):
                        raise ValueError("Select every stage before preparation")
                elif self.path != "/api/run":
                    self.respond({"error": "Not found"}, 404)
                    return
                with lock:
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
                        target, args = execute, (backend, steps, folder)
                    state.update(running=True, completed=False, message="Starting", error=None, run_dir=None)
                threading.Thread(target=target, args=args, daemon=True).start()
                self.respond({"started": True, "build_id": state["build_id"]}, 202)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                self.respond({"error": str(error)}, 400)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.state = state
    return server


def serve(port=8771, open_browser=False):
    server = create_server(port)
    print(f"UniBrain Agent: http://127.0.0.1:{server.server_port}", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    server.serve_forever()
