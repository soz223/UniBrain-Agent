"""Fetch immutable upstream files and optional public UniBrain examples."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unibrain_agent.paths import SOURCES


def fetch(url, target, expected=None):
    if target.exists() and (expected is None or hashlib.sha256(target.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    data = urlopen(url, timeout=120).read()
    if expected and hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != expected:
        raise ValueError(f"Source hash mismatch: {target.name}")
    temporary = target.with_suffix(target.suffix + ".download")
    temporary.write_bytes(data)
    temporary.replace(target)
    print(target.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "catalog/sources.json").read_text(encoding="utf-8"))
    for spec in manifest["repositories"]:
        if args.source and spec["directory"] not in args.source:
            continue
        base = spec["repository"].replace("github.com", "raw.githubusercontent.com") + "/" + spec["commit"] + "/"
        for item in spec["files"]:
            fetch(base + item["path"], SOURCES / spec["directory"] / item["path"], item["sha256"])
        if args.samples and spec["directory"] == "reg_unibrain":
            for name in ["adhd_train_sample.npy", "adhd_val_sample.npy", "adhd_test_sample.npy", "template_img_orig_96.npy",
                         "template_img_gm_mask_orig_96.npy", "template_img_aal_mask_orig_96.npy"]:
                fetch(base + "dataset/" + name, SOURCES / "reg_unibrain/dataset" / name)
    print("Sources ready. Upstream licenses remain in their source folders.")


if __name__ == "__main__":
    main()
