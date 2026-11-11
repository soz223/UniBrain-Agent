"""Training fold export and invocation of the official nnU-Net planner."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .data import OriginalDataset, digest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def export_training(args):
    import nibabel as nib

    source = args.data.resolve() / "adhd_train_sample.npy"
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing exports are retained")
    if args.data.resolve() == output or args.data.resolve() in output.parents:
        raise ValueError("Export outside the source data directory")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.dataset) or not 1 <= args.dataset_id <= 999 or args.fold < 1:
        raise ValueError("Use a simple dataset name, dataset ID 1..999 and positive fold")
    if args.coordinates == "physical_mm":
        if not args.spacing or not args.spacing_source or not all(np.isfinite(v) and v > 0 for v in args.spacing):
            raise ValueError("Physical spacing requires three verified values and their source")
        spacing, unit = args.spacing, "mm"
    else:
        if args.spacing or args.spacing_source:
            raise ValueError("Normalized voxel coordinates use unit grid spacing, not physical metadata")
        spacing, unit = [1., 1., 1.], "unknown"
    dataset = OriginalDataset(source)
    ids = [str(value) for value in dataset.array["id"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate subject IDs in the training split")
    dataset_name = f"Dataset{args.dataset_id:03d}_UniBrain_{args.dataset}_fold{args.fold}"
    raw = output / "nnUNet_raw" / dataset_name
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    manifest = {"status": "exporting", "dataset": args.dataset, "fold": args.fold,
                "dataset_id": args.dataset_id, "dataset_name": dataset_name,
                "training_only": True, "validation_read": False, "test_read": False,
                "source_train_sha256": digest(source), "source_shape": [96, 96, 96],
                "reference_kind": "Automatically generated tissue reference labels; not manual annotations",
                "coordinates": args.coordinates, "spacing_array_axes": list(spacing), "nifti_spatial_unit": unit,
                "spacing_source": args.spacing_source, "native_orientation_known": False, "nifti_to_source_axes": [2, 1, 0],
                "geometry_note": "NIfTI axes are reversed so official NibabelIO returns the original array axes. The diagonal affine and zero origin are synthetic. Normalized voxel units are not millimetres.",
                "cases": [], "files": {}}
    write_json(output / "export.json", manifest)
    affine = np.diag([*spacing[::-1], 1.])
    for index, row in enumerate(dataset.array):
        image = np.asarray(row["raw_img_np"], dtype=np.float32)
        labels = np.asarray(row["raw_img_gm_mask_np"])
        if not np.isfinite(image).all() or not np.isfinite(labels).all() or not np.isin(labels, [0, 1, 2, 3]).all():
            raise ValueError(f"Nonfinite image or invalid tissue reference in training case {index}")
        case = f"case{index:06d}"
        paths = ((raw / "imagesTr" / f"{case}_0000.nii.gz", image),
                 (raw / "labelsTr" / f"{case}.nii.gz", labels.astype(np.uint8)))
        for path, value in paths:
            volume = nib.Nifti1Image(value.transpose(2, 1, 0), affine)
            volume.header.set_xyzt_units(unit)
            nib.save(volume, path)
            manifest["files"][path.relative_to(output).as_posix()] = digest(path)
        manifest["cases"].append({"case": case, "source_index": index, "subject_id": ids[index]})
    dataset_json = {"channel_names": {"0": "T1"},
                    "labels": {"background": 0, "tissue_class_1": 1, "tissue_class_2": 2, "tissue_class_3": 3},
                    "numTraining": len(dataset), "file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIO",
                    "description": "UniBrain training fold only; automatically generated tissue references"}
    write_json(raw / "dataset.json", dataset_json)
    manifest["files"][(raw / "dataset.json").relative_to(output).as_posix()] = digest(raw / "dataset.json")
    manifest["status"] = "exported"
    write_json(output / "export.json", manifest)
    return manifest


def validate_export(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "export.json").read_text(encoding="utf-8"))
    if manifest["status"] != "exported" or not manifest["training_only"] or manifest["validation_read"] or manifest["test_read"]:
        raise ValueError("Planner requires a complete training-only export")
    if manifest.get("nifti_to_source_axes") != [2, 1, 0]:
        raise ValueError("Export lacks the verified NibabelIO to source axis mapping")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if output not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError("Export file missing or changed: " + relative)
    raw = output / "nnUNet_raw" / manifest["dataset_name"]
    expected = {str((output / name).resolve()) for name in manifest["files"]}
    actual = {str(path.resolve()) for path in raw.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Untracked files found in the exported training dataset")
    return manifest


def validate_plan_provenance(plans, training_hash, dataset, fold):
    sidecar = Path(plans).with_suffix(".provenance.json")
    if not sidecar.is_file():
        raise ValueError("nnU-Net training requires the planner provenance sidecar from prepare_nnunet.py")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if (provenance.get("status") != "ready" or provenance.get("training_only") is not True
            or not dataset or fold is None or provenance.get("dataset") != dataset or provenance.get("fold") != fold):
        raise ValueError("nnU-Net plan provenance must match the current training dataset and fold")
    if provenance.get("source_train_sha256") != training_hash:
        raise ValueError("nnU-Net plan training array hash does not match the current fold")
    if provenance.get("plans_sha256") != digest(plans):
        raise ValueError("nnU-Net plan changed after the recorded planner invocation")
    environment = provenance.get("planner_environment", {})
    if not environment.get("nnunetv2") or not re.fullmatch(r"[a-f0-9]{64}", environment.get("planner_source_sha256", "")):
        raise ValueError("Official planner version and source hash are required")
    return {"sidecar_sha256": digest(sidecar), **provenance}


def full_volume_contract(plans):
    from .full_candidates import planned_execution
    try:
        contract = planned_execution(plans)
        kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    except (KeyError, TypeError) as error:
        raise ValueError("The native architecture plan is incomplete") from error
    stages, strides = kwargs.get("n_stages", 0), kwargs.get("strides", [])
    features = kwargs.get("features_per_stage", [])
    if stages < 5 or len(strides) != stages or not features or min(features) < 16:
        raise ValueError("A complete full width planned network is required")
    native = contract["native_planned_patch"]
    if not isinstance(native, list) or len(native) != 3:
        raise ValueError("The native 3D patch shape is missing")
    return {**contract, "architecture_unchanged": True,
            "full_volume_adaptation": native != [96, 96, 96],
            "adaptation": "Execute the unchanged fully convolutional architecture on the 96 cubed UniBrain volume; retain the native plan and all stages"}


def run_planner(args):
    if args.processes < 1:
        raise ValueError("The process count must be positive")
    output = args.output.resolve()
    manifest = validate_export(output)
    attempt = output / "planner_attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(nnUNet_raw=str(output / "nnUNet_raw"), nnUNet_preprocessed=str(output / "nnUNet_preprocessed"),
                       nnUNet_results=str(output / "nnUNet_results"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    probe_code = ("import hashlib,inspect,json; from pathlib import Path; from importlib.metadata import version; "
                  "from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner; "
                  "p=Path(inspect.getfile(ExperimentPlanner)); "
                  "print(json.dumps({'nnunetv2':version('nnunetv2'),'dynamic_network_architectures':version('dynamic-network-architectures'),"
                  "'planner_source_sha256':hashlib.sha256(p.read_bytes().replace(bytes([13,10]),bytes([10]))).hexdigest()}))")
    probe = subprocess.run([str(args.python), "-c", probe_code], env=environment, capture_output=True, text=True)
    (attempt / "dependency_check.log").write_text(probe.stdout + probe.stderr, encoding="utf-8")
    report = {"status": "planning", "dataset": manifest["dataset"], "fold": manifest["fold"],
              "source_train_sha256": manifest["source_train_sha256"], "export_sha256": digest(output / "export.json"),
              "coordinates": manifest["coordinates"], "training_only": True, "gpu_used": False,
              "log_directory": str(attempt.relative_to(output))}
    if probe.returncode:
        report.update(status="blocked", reason="Install the official nnunetv2 package and its dependencies in the selected Python environment")
        write_json(output / "plan.json", report)
        write_json(attempt / "report.json", report)
        return report
    report["planner_environment"] = json.loads(probe.stdout.strip().splitlines()[-1])
    command = [str(args.python), "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
               "-d", str(manifest["dataset_id"]), "-pl", "ExperimentPlanner", "-npfp", str(args.processes),
               "--verify_dataset_integrity", "--no_pp", "--clean"]
    report["command"] = command
    preprocessed = output / "nnUNet_preprocessed" / manifest["dataset_name"]
    for existing in preprocessed.glob("*.json"):
        (attempt / ("previous_" + existing.name)).write_bytes(existing.read_bytes())
    with (attempt / "planner.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True)
    report["returncode"] = completed.returncode
    path = output / "nnUNet_preprocessed" / manifest["dataset_name"] / "nnUNetPlans.json"
    if completed.returncode or not path.is_file():
        report.update(status="failed", reason="Official planner failed; inspect the retained log")
    else:
        plans = json.loads(path.read_text(encoding="utf-8"))
        config = plans.get("configurations", {}).get("3d_fullres", {})
        architecture = config.get("architecture", {})
        try:
            report.update(full_volume_contract(plans))
            compatible = True
        except ValueError as error:
            compatible = False
            report["reason"] = str(error) + "; the native plan was preserved unchanged"
        report.update(status="ready" if compatible else "incompatible", plans=str(path.relative_to(output)),
                      plans_sha256=digest(path), patch_size=config.get("patch_size"),
                      transpose_forward=plans.get("transpose_forward"),
                      architecture=architecture.get("network_class_name"),
                      scope="Real planner output; forward/backward and complete UniBrain integration still require execution checks")
        write_json(path.with_suffix(".provenance.json"), report)
    write_json(output / "plan.json", report)
    write_json(attempt / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--dataset", required=True)
    export.add_argument("--fold", type=int, required=True)
    export.add_argument("--dataset-id", type=int, required=True)
    export.add_argument("--coordinates", choices=("normalized_voxel", "physical_mm"), required=True)
    export.add_argument("--spacing", nargs=3, type=float)
    export.add_argument("--spacing-source")
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--python", type=Path, default=Path(sys.executable))
    plan.add_argument("--processes", type=int, default=2)
    return result
