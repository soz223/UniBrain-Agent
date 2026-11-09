import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CONFIG = Path(os.environ.get("UNIBRAIN_CONFIG", PROJECT / "local.json")).expanduser().resolve()
LOCAL = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}


def configured_path(name, default):
    value = Path(os.path.expandvars(os.environ.get("UNIBRAIN_" + name.upper(), LOCAL.get(name, str(default))))).expanduser()
    return (value if value.is_absolute() else CONFIG.parent / value).resolve()


SOURCES = configured_path("sources", PROJECT / "sources")
UPSTREAM = SOURCES / "reg_unibrain"
RUNS = configured_path("runs", PROJECT / "runs")
BUILDS = configured_path("builds", PROJECT / "builds")
DATA = configured_path("data", UPSTREAM / "dataset")
ORIGINAL = configured_path("original_source", UPSTREAM)
CATALOG = PROJECT / "catalog"
