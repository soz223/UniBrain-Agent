"""Replay the public NeuroPilot conversion skill with its agent-selected rules."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(fixture, config, out, validator):
    out.mkdir(parents=True, exist_ok=False)
    bids = out / "bids"
    dicoms = sorted(fixture.glob("*/*.dcm"))
    if len(dicoms) != 15 or any(x.read_bytes()[128:132] != b"DICM" for x in dicoms):
        raise ValueError("Expected 15 DICOM slices from the ACMLab synthetic fixture")

    def command(name, *args):
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(list(map(str, args)), stdout=log, stderr=subprocess.STDOUT, check=True)

    command("inspect", "dcm2bids_helper", "-d", fixture, "-o", out / "inspection")
    metadata = sorted((out / "inspection" / "tmp_dcm2bids" / "helper").glob("*.json"))
    observed = {json.loads(f.read_text())["SeriesDescription"] for f in metadata}
    rules = json.loads(config.read_text())["descriptions"]
    if observed != {entry["criteria"]["SeriesDescription"] for entry in rules}:
        raise ValueError("Observed series descriptions do not match the recorded agent rules")
    command("scaffold", "dcm2bids_scaffold", "-o", bids)
    shutil.copyfile(config, bids / "code" / "dcm2bids_config.json")
    command("convert", "dcm2bids", "-d", fixture, "-p", "sub01", "-c", config,
            "-o", bids, "--auto_extract_entities")
    tmp = bids / "tmp_dcm2bids"
    if tmp.exists():
        tmp.rename(out / "work_after_conversion")
    desc = bids / "dataset_description.json"
    values = json.loads(desc.read_text())
    values.update(Name="ACMLab synthetic anatomical MRI conversion fixture", DatasetType="raw")
    if not any(name.strip() for name in values.get("Authors", [])):
        values.pop("Authors", None)
    desc.write_text(json.dumps(values, indent=2))
    (bids / "participants.tsv").write_text("participant_id\nsub-sub01\n")
    (bids / "README").write_text(
        "Synthetic anatomical MRI from the public ACMLab dcm2bids skill. One generated subject has "
        "three series with five DICOM slices each. These artificial voxel values test DICOM to BIDS "
        "conversion, series assignment and dataset validation. They contain no human imaging data."
    )
    report = subprocess.run(["node", validator, bids, "--json"], capture_output=True)
    (out / "validator.json").write_bytes(report.stdout)
    (out / "validator_stderr.log").write_bytes(report.stderr)
    report.check_returncode()
    issues = json.loads(report.stdout)["issues"]
    anat = bids / "sub-sub01" / "anat"
    names = {f.name for f in anat.iterdir()}
    expected = {f"sub-sub01_{s}.{ext}" for s in ("T1w", "T2w", "FLAIR") for ext in ("nii.gz", "json")}
    if names != expected or issues["errors"]:
        raise ValueError("BIDS output validation failed")
    detail = {
        "upstream_commit": "7e9880588e203404d08652aee576a2c68ee71f87",
        "scope": "Replay of agent-selected DICOM conversion rules on the upstream synthetic fixture",
        "runtime": sys.version.split()[0],
        "input_hashes": {str(f.relative_to(fixture)): hashlib.sha256(f.read_bytes()).hexdigest() for f in dicoms},
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "series": sorted(observed), "outputs": sorted(names),
        "validation_errors": len(issues["errors"]), "warnings": [x["key"] for x in issues["warnings"]],
    }
    (out / "result.json").write_text(json.dumps(detail, indent=2))
    print(json.dumps({key: detail[key] for key in ("scope", "series", "validation_errors", "warnings")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "config", "output", "validator"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    run(args.fixture, args.config, args.output, args.validator)
