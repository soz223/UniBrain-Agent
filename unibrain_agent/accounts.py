import json
import queue
import shutil
import subprocess
import threading
import time


def codex_account(executable):
    process = subprocess.Popen([executable, "app-server", "--listen", "stdio://"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    inbox = queue.Queue()

    def read():
        for line in process.stdout:
            try:
                inbox.put(json.loads(line))
            except ValueError:
                continue

    def request(identifier, method, params):
        process.stdin.write(json.dumps({"id": identifier, "method": method, "params": params}) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            result = inbox.get(timeout=max(.1, deadline - time.monotonic()))
            if result.get("id") == identifier:
                if "error" in result:
                    raise ValueError("Account metadata unavailable")
                return result["result"]
        raise TimeoutError("Account query timed out")

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        request(0, "initialize", {"clientInfo": {"name": "unibrain_agent", "version": "0.8"}, "capabilities": {}})
        process.stdin.write('{"method":"initialized"}\n')
        process.stdin.flush()
        account = request(1, "account/read", {"refreshToken": False}).get("account")
        return {"authenticated": bool(account), "email": (account or {}).get("email"),
                "plan": (account or {}).get("planType"), "method": (account or {}).get("type")}
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=1)
        process.stdout.close()


def login_status(provider):
    if provider not in {"codex", "claude"}:
        raise ValueError("Choose a CLI provider")
    executable = shutil.which(provider)
    result = {"installed": bool(executable), "authenticated": False, "email": None, "name": None, "plan": None, "method": None}
    if not executable:
        return {**result, "message": "Install the CLI on your backend machine to connect an account."}
    try:
        if provider == "codex":
            result.update(codex_account(executable))
        else:
            process = subprocess.run([executable, "auth", "status"], capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            data = json.loads(process.stdout)
            result.update(authenticated=process.returncode == 0 and data.get("loggedIn", True),
                          email=data.get("email"), name=data.get("name"), plan=data.get("subscriptionType"), method=data.get("authMethod"))
        result["message"] = "Account connected on this backend." if result["authenticated"] else "No account is signed in on this backend."
    except (OSError, ValueError, KeyError, TypeError, queue.Empty, TimeoutError, subprocess.TimeoutExpired):
        result.update(authenticated=None, message="Account status could not be read. Refresh or check the CLI on your backend.")
    return result


def logout(provider):
    if provider not in {"codex", "claude"}:
        raise ValueError("Choose a CLI provider")
    executable = shutil.which(provider)
    if not executable:
        raise ValueError("CLI is not installed on this backend")
    command = [executable, "logout"] if provider == "codex" else [executable, "auth", "logout"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=20,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Sign-out did not finish. Check the CLI on your backend.") from None
    if result.returncode:
        raise ValueError("CLI could not sign out. An environment-managed credential may still be active.")
    return login_status(provider)
