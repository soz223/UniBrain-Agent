import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
LOCAL = json.loads((PROJECT / "local.json").read_text(encoding="utf-8")) if (PROJECT / "local.json").exists() else {}
SOURCES = Path(os.environ.get("UNIBRAIN_SOURCES", LOCAL.get("sources", PROJECT / "sources")))
UPSTREAM = SOURCES / "reg_unibrain"
RUNS = Path(os.environ.get("UNIBRAIN_RUNS", LOCAL.get("runs", PROJECT / "runs")))
BUILDS = PROJECT / "builds"
CATALOG = PROJECT / "catalog"
