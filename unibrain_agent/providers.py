import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .paths import configured_path, PROJECT

PROVIDERS = {
    "codex": {"name": "Codex CLI", "command": "codex login", "env": "", "base_url": ""},
    "claude": {"name": "Claude Code", "command": "claude auth login", "env": "", "base_url": ""},
    "openai": {"name": "OpenAI API", "command": "", "env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "anthropic": {"name": "Anthropic API", "command": "", "env": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com/v1"},
    "compatible": {"name": "OpenAI compatible API", "command": "", "env": "UNIBRAIN_API_KEY", "base_url": ""},
}


def validate_profile(provider, profile):
    if provider not in PROVIDERS or not isinstance(profile, dict):
        raise ValueError("Unknown AI provider")
    model = profile.get("model", "")
    base = profile.get("base_url", PROVIDERS[provider]["base_url"])
    if not isinstance(model, str) or len(model) > 160 or any(ord(c) < 32 for c in model):
        raise ValueError("Enter a valid model ID")
    if not isinstance(base, str) or len(base) > 500:
        raise ValueError("Enter a valid API base URL")
    model, base = model.strip(), base.strip().rstrip("/")
    if provider in {"codex", "claude"}:
        return {"model": model, "base_url": ""}
    if provider != "compatible":
        base = PROVIDERS[provider]["base_url"]
    parsed = urlsplit(base)
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or any(c.isspace() for c in base)
            or (parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}))):
        raise ValueError("Use an HTTPS base URL, or HTTP on localhost, without credentials or query parameters")
    if not model:
        raise ValueError("Enter the model ID supplied by your API provider")
    return {"model": model, "base_url": base}


class ProviderSettings:
    def __init__(self, path=None):
        self.path = Path(path) if path else configured_path("ai_settings", PROJECT / "ai.local.json")
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.selected = saved.get("provider", "codex")
        self.profiles = {key: {"model": "", "base_url": value["base_url"]} for key, value in PROVIDERS.items()}
        for key, profile in saved.get("profiles", {}).items():
            self.profiles[key] = validate_profile(key, profile)
        if self.selected not in PROVIDERS:
            raise ValueError("Unknown saved AI provider")
        self.secrets = {}
        self.lock = threading.RLock()
        self.test = {"status": "idle"}

    def public(self):
        with self.lock:
            return {"provider": self.selected, "profiles": self.profiles.copy(), "test": self.test.copy(),
                    "providers": [{"id": key, **value, "installed": bool(shutil.which(key)) if key in {"codex", "claude"} else None,
                                   "credential": "session" if key in self.secrets else "environment" if value["env"] and os.environ.get(value["env"]) else "not supplied"}
                                  for key, value in PROVIDERS.items()]}

    def snapshot(self, provider=None):
        with self.lock:
            provider = provider or self.selected
            value = {"provider": provider, **self.profiles[provider]}
            value["api_key"] = self.secrets.get(provider) or os.environ.get(PROVIDERS[provider]["env"], "")
            return value

    def update(self, body):
        provider = body.get("provider")
        if not isinstance(provider, str):
            raise ValueError("Choose an AI provider")
        profile = validate_profile(provider, body)
        key = body.get("api_key", "")
        if not isinstance(key, str) or len(key) > 8192 or any(ord(c) < 32 for c in key):
            raise ValueError("Invalid API key")
        with self.lock:
            if self.test["status"] == "running":
                raise ValueError("Wait for the connection test to finish")
            profiles = {**self.profiles, provider: profile}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"provider": provider, "profiles": {k: v for k, v in profiles.items() if v["model"] or k in {"codex", "claude"}}}, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            if profile["base_url"] != self.profiles[provider]["base_url"] or body.get("clear_key"):
                self.secrets.pop(provider, None)
            if key.strip() and provider not in {"codex", "claude"}:
                self.secrets[provider] = key.strip()
            self.selected, self.profiles = provider, profiles
            self.test = {"status": "idle"}
        return self.public()

    def start_test(self):
        with self.lock:
            if self.test["status"] == "running":
                raise ValueError("A connection test is already running")
            config = self.snapshot()
            self.test = {"status": "running", "provider": config["provider"]}
        threading.Thread(target=self._test, args=(config,), daemon=True).start()
        return self.public()

    def _test(self, config):
        started = time.perf_counter()
        try:
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
            with tempfile.TemporaryDirectory(prefix="unibrain_ai_") as folder:
                result, usage = invoke(config, 'Return JSON with ok set to true.', schema, Path(folder), "connection")
            if not isinstance(result, dict) or set(result) != {"ok"} or result["ok"] is not True:
                raise ValueError("The model returned an unexpected connection response")
            status = {"status": "complete", "message": "Connected. The model returned valid structured output.", "usage": usage}
        except Exception as error:
            status = {"status": "failed", "message": str(error)}
        with self.lock:
            self.test = {**status, "provider": config["provider"], "seconds": round(time.perf_counter() - started, 2)}


from .accounts import login_status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_request(config, prompt, schema):
    provider, model, key = config["provider"], config["model"], config["api_key"]
    if not key:
        raise ValueError("Supply an API key in AI settings or the documented backend environment variable")
    if not model:
        raise ValueError("Choose a model in AI settings")
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        suffix = "/responses"
        headers["Authorization"] = "Bearer " + key
        body = {"model": model, "input": prompt, "store": False,
                "text": {"format": {"type": "json_schema", "name": "unibrain_decision", "schema": schema, "strict": True}}}
    elif provider == "anthropic":
        suffix = "/messages"
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        body = {"model": model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}],
                "output_config": {"format": {"type": "json_schema", "schema": schema}}}
    else:
        suffix = "/chat/completions"
        headers["Authorization"] = "Bearer " + key
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "unibrain_decision", "schema": schema, "strict": True}}}
    request = Request(config["base_url"] + suffix, data=json.dumps(body).encode(), headers=headers)
    try:
        with build_opener(NoRedirect()).open(request, timeout=240) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Provider returned HTTP {error.code}. Check credentials, model access and structured output support.") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Could not reach the configured AI endpoint") from None
    try:
        if provider == "openai":
            if payload.get("status") != "completed":
                raise ValueError("Incomplete response")
            text = "".join(part.get("text", "") for item in payload["output"] for part in item.get("content", []) if part.get("type") == "output_text")
        elif provider == "anthropic":
            if payload.get("stop_reason") != "end_turn":
                raise ValueError("Incomplete response")
            text = "".join(item["text"] for item in payload["content"] if item["type"] == "text")
        else:
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete response")
            text = choice["message"]["content"]
        return json.loads(text), payload.get("usage")
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("The provider returned incomplete or invalid JSON output") from None


def invoke(config, prompt, schema, folder, role):
    provider = config["provider"]
    if provider not in {"codex", "claude"}:
        return api_request(config, prompt, schema)
    executable = shutil.which(provider)
    if not executable:
        raise RuntimeError(f"{PROVIDERS[provider]['name']} is not installed on the backend machine. Open AI settings.")
    folder.mkdir(parents=True, exist_ok=True)
    schema_path, output_path = folder / f"{role}.schema.json", folder / f"{role}.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    if provider == "codex":
        command = [executable, "exec", "--ignore-user-config", "--sandbox", "read-only", "--ephemeral",
                   "--skip-git-repo-check", "--json", "--output-schema", str(schema_path.resolve()),
                   "--output-last-message", str(output_path.resolve())]
        if config["model"]:
            command += ["--model", config["model"]]
        command.append("-")
    else:
        command = [executable, "-p", "--safe-mode", "--tools", "", "--disallowedTools", "mcp__*",
                   "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence",
                   "--output-format", "json", "--json-schema", json.dumps(schema)]
        if config["model"]:
            command += ["--model", config["model"]]
    try:
        completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace",
                                   capture_output=True, timeout=240, cwd=folder,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("CLI request failed or timed out. Check installation and login in AI settings.") from None
    if completed.returncode:
        raise RuntimeError(f"{PROVIDERS[provider]['name']} exited with code {completed.returncode}. Check login, model access and CLI version.")
    if provider == "claude":
        try:
            payload = json.loads(completed.stdout)
            if payload.get("is_error") or not isinstance(payload.get("structured_output"), dict):
                raise ValueError("Missing structured result")
            return payload["structured_output"], payload.get("usage")
        except (ValueError, TypeError):
            raise RuntimeError("Claude Code did not return valid structured output. Check login and CLI version.") from None
    if not output_path.exists():
        raise RuntimeError("Codex returned no result. Check login and model access in AI settings.")
    result = json.loads(output_path.read_text(encoding="utf-8-sig"))
    usage = None
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
        except json.JSONDecodeError:
            pass
    return result, usage


_settings = None
_settings_lock = threading.Lock()


def provider_settings():
    global _settings
    with _settings_lock:
        if _settings is None:
            _settings = ProviderSettings()
        return _settings
