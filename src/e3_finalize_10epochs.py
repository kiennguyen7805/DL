"""Audit and package the existing E3 epoch-10 Kaggle result, without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from kaggle_stage_c import (canonical_hash, effective_args_mismatches,
                            inspect_checkpoint, safe_extract, sha256)

EXPECTED_SOURCE_SHA = "e60c3f490e25636c1a03fbef6e55bc7add534d760ffced84919b95c307646983"
EXPERIMENT = "E3_pretrained"
WEIGHTS = ("best.pt", "last.pt", "resume_epoch010.pt", "recovery_full.pt")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(handle)]
    if [int(row["epoch"]) for row in rows] != list(range(1, 11)):
        raise ValueError("E3 results.csv must contain exactly epochs 1–10")
    return rows


def _load_for_inference(path, yolo_type):
    model = yolo_type(str(path))
    if model.task != "detect" or len(model.names) != 25:
        raise ValueError(f"Checkpoint is not a 25-class detector: {path}")
    return {"load": "PASS", "task": model.task, "classes": len(model.names)}


def audit_source(source_zip, frozen_yaml, torch_module, yaml_module, yolo_type):
    """Read the original Kaggle ZIP and return a validated temporary extraction."""
    source_zip = Path(source_zip)
    if sha256(source_zip) != EXPECTED_SOURCE_SHA:
        raise ValueError("results11.zip SHA256 differs from the audited source")
    temp = tempfile.TemporaryDirectory(prefix="rod_e3_epoch010_")
    try:
        root = Path(temp.name)
        safe_extract(source_zip, root)
        run = root / "rod_main" / EXPERIMENT
        logs = root / "rod_main" / "logs"
        for rel in ("args.yaml", "results.csv"):
            if not (run / rel).is_file():
                raise FileNotFoundError(run / rel)
        for name in WEIGHTS:
            if not (run / "weights" / name).is_file():
                raise FileNotFoundError(run / "weights" / name)
        expected = yaml_module.safe_load(Path(frozen_yaml).read_text(encoding="utf-8"))
        actual = yaml_module.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        if expected["model"] != "yolo11n.pt" or expected["pretrained"] is not True:
            raise ValueError("Frozen E3 initialization differs")
        if expected["epochs"] != 30 or actual["epochs"] != 30:
            raise ValueError("Original training schedule must remain recorded as 30")
        if effective_args_mismatches(expected, actual, {"data", "project", "name", "model"}):
            raise ValueError("Effective E3 args differ from frozen YAML")
        if actual["model"] != "yolo11n.pt" or actual["name"] != EXPERIMENT:
            raise ValueError("Actual model/name differs from E3")
        common = {k: v for k, v in expected.items() if k not in {"model", "pretrained", "name"}}
        config_hash = canonical_hash(common)
        source_config = _json(logs / "config_audit.json")
        if source_config["common_sha256"] != config_hash:
            raise ValueError("Kaggle config audit fingerprint differs")
        rows = _rows(run / "results.csv")
        metrics = {key: float(rows[-1][column]) for key, column in {
            "precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
            "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)"}.items()}
        checkpoint_manifest = _json(logs / "checkpoint_manifest.json")
        resume = run / "weights" / "resume_epoch010.pt"
        info = inspect_checkpoint(resume, 10, torch_module)
        if (checkpoint_manifest["latest_recovery"]["sha256"] != info["sha256"] or
                checkpoint_manifest["latest_recovery"]["epoch"] != 10):
            raise ValueError("Checkpoint manifest differs from full-state checkpoint")
        if sha256(run / "weights" / "recovery_full.pt") != info["sha256"]:
            raise ValueError("Recovery and resume checkpoint hashes differ")
        models = {}
        for name in WEIGHTS:
            path = run / "weights" / name
            raw = torch_module.load(path, map_location="cpu", weights_only=False)
            models[name] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                            "raw_epoch": raw.get("epoch"),
                            "optimizer_present": raw.get("optimizer") is not None,
                            "ema_present": raw.get("ema") is not None,
                            "train_args_epochs": raw.get("train_args", {}).get("epochs")}
            if models[name]["train_args_epochs"] != 30:
                raise ValueError(f"Historical schedule missing from {name}")
            models[name].update(_load_for_inference(path, yolo_type))
            del raw
        for name in ("best.pt", "last.pt"):
            if models[name]["optimizer_present"]:
                raise ValueError(f"Unexpected optimizer in stripped checkpoint {name}")
        if not models["resume_epoch010.pt"]["optimizer_present"]:
            raise ValueError("Resume checkpoint has no optimizer")
        markers = set(re.findall(r"marker_sha256=([0-9a-fA-F]{64})",
                                 (logs / "execution.log").read_text(encoding="utf-8")))
        if len(markers) != 1:
            raise ValueError("Dataset fingerprint missing or ambiguous in execution.log")
        return temp, {"source_archive": {"path": str(source_zip), "sha256": sha256(source_zip),
                                         "bytes": source_zip.stat().st_size, "crc": "PASS"},
                      "experiment": EXPERIMENT, "completed_epochs": 10,
                      "original_schedule_epochs": 30, "current_main_budget_epochs": 10,
                      "dataset_marker_sha256": next(iter(markers)).lower(),
                      "common_config_sha256": config_hash,
                      "frozen_yaml_sha256": sha256(frozen_yaml),
                      "metrics_epoch_10_valid": metrics, "checkpoints": models,
                      "full_state": info, "original_kaggle_error": _json(logs / "errors.json")}
    except Exception:
        temp.cleanup()
        raise


def finalize(source_zip, frozen_yaml, output, torch_module, yaml_module, yolo_type):
    """Create a standalone E3 package; refuses to overwrite an existing artifact."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    temp, audit = audit_source(source_zip, frozen_yaml, torch_module, yaml_module, yolo_type)
    try:
        root = Path(temp.name)
        package_root = root / "e3_epoch010"
        package_root.mkdir()
        source = root / "rod_main"
        shutil.copytree(source / EXPERIMENT, package_root / EXPERIMENT)
        (package_root / "logs").mkdir()
        for name in ("checkpoint_manifest.json", "config_audit.json", "environment.json",
                     "execution.log", "training_status.json", "errors.json"):
            shutil.copy2(source / "logs" / name, package_root / "logs" / name)
        shutil.copy2(frozen_yaml, package_root / "e3_pretrained_frozen.yaml")
        files = {p.relative_to(package_root).as_posix(): sha256(p)
                 for p in package_root.rglob("*") if p.is_file()}
        audit["files_sha256"] = files
        (package_root / "manifest.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".tmp")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(package_root).as_posix())
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None or set(files) | {"manifest.json"} != set(z.namelist()):
                raise ValueError("Standalone E3 artifact ZIP failed CRC/completeness")
            for rel, digest in files.items():
                if hashlib.sha256(z.read(rel)).hexdigest() != digest:
                    raise ValueError(f"Packaged file checksum mismatch: {rel}")
        partial.replace(output)
        sidecar = output.with_suffix(output.suffix + ".sha256")
        sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
        return {"status": "PASS_LOCAL", "artifact": str(output), "bytes": output.stat().st_size,
                "sha256": sha256(output), "manifest": audit}
    finally:
        temp.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--frozen-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml
    from ultralytics import YOLO
    result = finalize(args.source_zip, args.frozen_yaml, args.output, torch, yaml, YOLO)
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
