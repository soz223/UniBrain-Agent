"""Install a private loopback backend as a Linux user service."""

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--origin", default="https://soz223.github.io")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this installer on the Linux backend host")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from unibrain_agent.access import origin_url
    origin = origin_url(args.origin)
    if not 1024 <= args.port <= 65535 or args.gpu < 0:
        parser.error("Invalid port or GPU index")
    state = args.state.resolve()
    unit = Path.home() / ".config/systemd/user/unibrain-backend.service"
    if unit.exists() or (state / "backend.env").exists():
        raise FileExistsError("Existing service configuration must be reviewed before replacement")
    state.mkdir(parents=True, exist_ok=True)
    config = {"data": str(args.data.resolve()), "sources": str(project / "sources"),
              "original_source": str(project / "sources/reg_unibrain"),
              "runs": str(state / "runs"), "builds": str(state / "builds"),
              "data_jobs": str(state / "data_jobs"), "assistant": str(state / "assistant_sessions"),
              "original_python": sys.executable, "device": "cuda:0", "threads": 4}
    (state / "runtime.json").write_text(json.dumps(config, indent=2))
    environment = {"UNIBRAIN_CONFIG": str(state / "runtime.json"),
                   "UNIBRAIN_AI_SETTINGS": str(state / "ai.local.json"),
                   "UNIBRAIN_ACCESS_TOKEN": secrets.token_urlsafe(36),
                   "UNIBRAIN_ALLOWED_ORIGINS": origin, "CUDA_VISIBLE_DEVICES": str(args.gpu),
                   "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    env_path = state / "backend.env"
    with os.fdopen(os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write("\n".join(f'{key}={json.dumps(value)}' for key, value in environment.items()) + "\n")
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(f'''[Unit]
Description=UniBrain Agent backend
After=network.target

[Service]
Type=simple
WorkingDirectory={project}
EnvironmentFile={env_path}
ExecStart="{sys.executable}" -m unibrain_agent serve --host 127.0.0.1 --port {args.port}
Restart=on-failure
RestartSec=5
UMask=0077

[Install]
WantedBy=default.target
''')
    subprocess.run(["systemd-analyze", "--user", "verify", str(unit)], check=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit.name], check=True)
    subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit.name], check=True)
    print(json.dumps({"service": unit.name, "state": str(state), "port": args.port,
                      "token_file": str(env_path)}, indent=2))


if __name__ == "__main__":
    main()
