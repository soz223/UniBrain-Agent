#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
runtime="${UNIBRAIN_PYTHON:-.venv/bin/python}"
exec "$runtime" -m unibrain_agent serve --port "${UNIBRAIN_PORT:-8776}"
