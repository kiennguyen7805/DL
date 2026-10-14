"""ROD Stage C: resumable, audited Kaggle training (Ultralytics 8.4.165).

The notebook embeds this file so that it runs without a repository checkout.
Importing this module never starts training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_COUNTS = {"train": 18151, "valid": 3318, "test": 1611}
EXPECTED_NAMES = ["Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
                  "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
                  "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
                  "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
                  "Electrical Box", "Chair", "Bicycle Rack"]
EXPERIMENTS = {"E3_pretrained": ("yolo11n.pt", True),
               "E5_scratch": ("yolo11n.yaml", False)}
EXPECTED_ULTRALYTICS = "8.4.165"
STAGES = {1: (1, 10), 2: (11, 20), 3: (21, 30)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_effective_arg(key, value):
    """Normalize values whose YAML/runtime representations are semantically equal."""
    if key == "device":
        if isinstance(value, (list, tuple)):
            return ",".join(str(item).strip() for item in value)
        if value is None:
            return None
        return ",".join(part.strip() for part in str(value).strip().split(","))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [normalize_effective_arg(key, item) for item in value]
    return value


def effective_args_mismatches(expected, actual, ignored=None):
    ignored = set(ignored or ())
    mismatches = {}
    for key, expected_value in expected.items():
        if key in ignored:
            continue
        actual_value = actual.get(key)
        if normalize_effective_arg(key, actual_value) != normalize_effective_arg(key, expected_value):
            mismatches[key] = {"actual": actual_value, "expected": expected_value}
    return mismatches


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def log(root, message):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    with (root / "execution.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def validate_stage(stage):
    if stage not in STAGES:
        raise ValueError("STAGE phải là 1, 2 hoặc 3")
    return STAGES[stage]


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC fail")
        for item in z.infolist():
            name = item.filename.replace("\\", "/")
            if name.startswith("/") or ":" in name or not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe ZIP member: {item.filename}")
        z.extractall(destination)


def find_dataset(input_root):
    markers = []
    for p in Path(input_root).rglob("final_build.json"):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if meta.get("version") == "ROD_FINAL_A4" and meta.get("frozen") and meta.get("verified"):
                markers.append(p)
        except (OSError, ValueError):
            pass
    if len(markers) != 1:
        raise RuntimeError(f"Cần đúng một ROD Final A4; thấy {len(markers)}")
    return markers[0].parent, sha256(markers[0])


def verify_dataset(root, yaml_module):
    cfg = yaml_module.safe_load((Path(root) / "data.yaml").read_text(encoding="utf-8"))
    if cfg["nc"] != 25 or cfg["names"] != EXPECTED_NAMES:
        raise ValueError("ROD Final class mapping khác 25 class đã freeze")
    for split, expected in EXPECTED_COUNTS.items():
        images = [p for p in (Path(root) / split / "images").iterdir() if p.is_file()]
        labels = list((Path(root) / split / "labels").glob("*.txt"))
        if len(images) != expected or len(labels) != expected or {p.stem.casefold() for p in images} != {p.stem.casefold() for p in labels}:
            raise ValueError(f"{split}: image/label count or stems invalid")
    return cfg


def config_audit(configs, yaml_module):
    loaded = {exp: yaml_module.safe_load(text) for exp, text in configs.items()}
    if set(loaded) != set(EXPERIMENTS):
        raise ValueError("Thiếu cấu hình E3/E5")
    allowed_diff = {"model", "pretrained", "name"}
    a, b = (loaded[x] for x in EXPERIMENTS)
    if set(a) != set(b) or {k for k in a if a[k] != b[k]} != allowed_diff:
        raise ValueError("E3/E5 phải chỉ khác model, pretrained, name")
    for exp, cfg in loaded.items():
        spec, pretrained = EXPERIMENTS[exp]
        if cfg["model"] != spec or cfg["pretrained"] is not pretrained or cfg["name"] != exp:
            raise ValueError(f"Initialization sai: {exp}")
        if cfg["epochs"] != 30 or cfg["data"] != "/kaggle/working/rod_final_trainval.yaml":
            raise ValueError(f"epochs/data sai: {exp}")
    common = {k: v for k, v in a.items() if k not in allowed_diff}
    return loaded, {"common_sha256": canonical_hash(common),
                    "yaml_sha256": {exp: hashlib.sha256(text.encode()).hexdigest() for exp, text in configs.items()},
                    "common": common, "different_fields": sorted(allowed_diff)}


def inspect_checkpoint(path, expected_epoch, torch_module):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ck = torch_module.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or ck.get("epoch") != expected_epoch - 1:
        raise ValueError(f"Wrong checkpoint epoch: {path}")
    if not isinstance(ck.get("optimizer"), dict) or not ck["optimizer"].get("state") or not ck["optimizer"].get("param_groups"):
        raise ValueError(f"Optimizer state missing: {path}")
    if ck.get("ema") is None or ck.get("scaler") is None:
        raise ValueError(f"EMA/scaler state missing: {path}")
    if ck.get("train_args", {}).get("epochs") != 30:
        raise ValueError(f"Checkpoint schedule is not 30 epochs: {path}")
    info = {"path": str(path), "epoch": expected_epoch, "raw_epoch": ck.get("epoch"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path), "optimizer_state_entries": len(ck["optimizer"]["state"]),
            "optimizer_param_groups": len(ck["optimizer"]["param_groups"]),
            "ema_present": ck.get("ema") is not None, "ema_type": type(ck.get("ema")).__name__,
            "scaler_present": ck.get("scaler") is not None,
            "scaler_type": type(ck.get("scaler")).__name__,
            "model_present": ck.get("model") is not None,
            "ema_updates": ck.get("updates"), "lr": [x.get("lr") for x in ck["optimizer"]["param_groups"]],
            "train_args_epochs": ck.get("train_args", {}).get("epochs"),
            "train_args_device": ck.get("train_args", {}).get("device"),
            "ultralytics": ck.get("version")}
    del ck
    return info


def atomic_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temp)
    os.replace(temp, target)


def csv_epochs(path, expected_end):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}
                for row in csv.DictReader(f)]
    epochs = [int(float(x["epoch"])) for x in rows]
    if len(epochs) != expected_end or epochs != list(range(1, expected_end + 1)):
        raise ValueError(f"results.csv epoch sequence invalid: {epochs}")
    return rows


def package(root, output, required):
    root, output = Path(root), Path(output)
    for rel in required:
        if not (root / rel).is_file():
            raise FileNotFoundError(root / rel)
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                z.write(p, p.relative_to(root).as_posix())
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None or not set(required).issubset(z.namelist()):
            temp.unlink(missing_ok=True)
            raise ValueError("Packaged ZIP failed CRC or completeness")
    os.replace(temp, output)
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output), "crc": "PASS"}


def build_completed_experiment_artifact(source_root, output, config_texts, torch_module, yaml_module,
                                        *, experiment="E3_pretrained", stage=1,
                                        source_archive=None):
    """Build a verified partial-stage artifact without changing the downloaded source."""
    start, end = validate_stage(stage)
    if start != 1 or experiment != "E3_pretrained":
        raise ValueError("Recovery builder currently accepts only completed Stage 1 E3")
    source_root, output = Path(source_root), Path(output)
    loaded, audit = config_audit(config_texts, yaml_module)
    exp_dir = source_root / experiment
    logs = source_root / "logs"
    resume = exp_dir / "weights" / f"resume_epoch{end:03d}.pt"
    recovery = exp_dir / "weights" / "recovery_full.pt"
    resume_info = inspect_checkpoint(resume, end, torch_module)
    recovery_info = inspect_checkpoint(recovery, end, torch_module)
    if resume_info["sha256"] != recovery_info["sha256"]:
        raise ValueError("resume_epoch010.pt and recovery_full.pt are not identical")
    rows = csv_epochs(exp_dir / "results.csv", end)
    source_args = yaml_module.safe_load((exp_dir / "args.yaml").read_text(encoding="utf-8"))
    mismatches = effective_args_mismatches(
        loaded[experiment], source_args, {"data", "project", "model", "name"})
    if mismatches:
        raise ValueError(f"Effective args mismatch in recovered E3: {mismatches}")
    source_audit = json.loads((logs / "config_audit.json").read_text(encoding="utf-8"))
    if source_audit.get("common_sha256") != audit["common_sha256"]:
        raise ValueError("Recovered E3 config fingerprint differs from frozen YAML")
    execution_log = (logs / "execution.log").read_text(encoding="utf-8")
    markers = sorted(set(re.findall(r"marker_sha256=([0-9a-fA-F]{64})", execution_log)))
    if len(markers) != 1:
        raise ValueError(f"Expected one dataset marker fingerprint in execution.log; found {markers}")
    dataset_sha = markers[0].lower()
    resume_info["path"] = f"{experiment}/weights/resume_epoch{end:03d}.pt"
    recovery_info["path"] = f"{experiment}/weights/recovery_full.pt"
    record = {"experiment": experiment, "stage": stage, "start_epoch": start,
        "end_epoch": end, "elapsed_seconds": float(rows[-1].get("time", 0)),
        "peak_vram_reserved_gib": None, "checkpoint": resume_info,
        "last_metrics": rows[-1],
        "best_epoch_by_map5095": max(rows, key=lambda row: float(
            row.get("metrics/mAP50-95(B)", 0)))["epoch"],
        "recovered_from_downloaded_output": True}
    stage_manifest = {"status": "PARTIAL", "stage": stage, "completed_epoch": end,
        "dataset_marker_sha256": dataset_sha, "common_config_sha256": audit["common_sha256"],
        "checkpoints": {experiment: resume_info}}
    source_files = {
        "results.csv": sha256(exp_dir / "results.csv"),
        "args.yaml": sha256(exp_dir / "args.yaml"),
        "execution.log": sha256(logs / "execution.log"),
        "errors.json": sha256(logs / "errors.json"),
        "checkpoint_manifest.json": sha256(logs / "checkpoint_manifest.json"),
        "resume_epoch010.pt": resume_info["sha256"],
        "recovery_full.pt": recovery_info["sha256"],
    }
    artifact_manifest = {"schema": "rod.completed-experiment-artifact.v1",
        "purpose": "Skip completed E3 in Stage 1 and train only E5 epochs 1-10",
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment": experiment,
        "stage": stage, "completed_epoch": end, "dataset_marker_sha256": dataset_sha,
        "common_config_sha256": audit["common_sha256"], "checkpoint": resume_info,
        "recovery_checkpoint": recovery_info, "last_metrics": rows[-1],
        "source_archive": ({"path": str(source_archive), "sha256": sha256(source_archive)}
                           if source_archive else None),
        "source_file_sha256": source_files}
    with tempfile.TemporaryDirectory(prefix="rod_e3_recovery_") as temp_dir:
        artifact_root = Path(temp_dir) / "rod_main"
        shutil.copytree(exp_dir, artifact_root / experiment)
        shutil.copytree(logs, artifact_root / "logs")
        artifact_logs = artifact_root / "logs"
        for name in ("errors.json", "checkpoint_manifest.json", "training_status.json"):
            path = artifact_logs / name
            if path.is_file():
                shutil.copy2(path, artifact_logs / f"source_{name}")
        write_json(artifact_root / "stage_manifest.json", stage_manifest)
        write_json(artifact_root / "artifact_manifest.json", artifact_manifest)
        write_json(artifact_logs / "stage_summary.json", {experiment: record})
        write_json(artifact_logs / "checkpoint_manifest.json", stage_manifest)
        write_json(artifact_logs / "training_status.json", {"stage": stage,
            "experiments": {"E3_pretrained": "PASS", "E5_scratch": "PENDING"}})
        write_json(artifact_logs / "errors.json", {"errors": []})
        (artifact_logs / "recovery_report.md").write_text(
            "# Stage 1 E3 recovery artifact\n\n"
            f"E3 epochs 1-{end} were recovered without retraining. Checkpoint SHA256: "
            f"`{resume_info['sha256']}`. Dataset marker SHA256: `{dataset_sha}`. "
            f"Common config SHA256: `{audit['common_sha256']}`.\n",
            encoding="utf-8")
        required = ["stage_manifest.json", "artifact_manifest.json",
            f"{experiment}/weights/resume_epoch{end:03d}.pt",
            f"{experiment}/weights/recovery_full.pt", f"{experiment}/results.csv",
            f"{experiment}/args.yaml", "logs/stage_summary.json",
            "logs/config_audit.json", "logs/environment.json", "logs/execution.log",
            "logs/errors.json", "logs/source_errors.json", "logs/recovery_report.md"]
        result = package(artifact_root, output, required)
    result["artifact_manifest"] = artifact_manifest
    return result


def locate_previous(input_root, stage, explicit=None):
    validate_stage(stage)
    if stage == 1:
        return None
    name = f"rod_main_stage{stage-1:02d}_epoch{(stage-1)*10:03d}.zip"
    found = [Path(explicit)] if explicit else list(Path(input_root).rglob(name))
    if len(found) != 1 or not found[0].is_file():
        raise FileNotFoundError(f"Need exactly one {name}; found {found}")
    return found[0]


def validate_previous(previous, stage, dataset_sha, config_sha, torch_module, root):
    start, _ = validate_stage(stage)
    if previous.get("stage") != stage - 1 or previous.get("completed_epoch") != start - 1:
        raise ValueError("Wrong previous stage")
    if previous.get("dataset_marker_sha256") != dataset_sha or previous.get("common_config_sha256") != config_sha:
        raise ValueError("Dataset/config fingerprint mismatch")
    for exp in EXPERIMENTS:
        rel = f"{exp}/weights/resume_epoch{start-1:03d}.pt"
        path = Path(root) / rel
        if previous["checkpoints"][exp]["sha256"] != sha256(path):
            raise ValueError(f"Checkpoint checksum mismatch: {exp}")
        inspect_checkpoint(path, start - 1, torch_module)
        csv_epochs(Path(root) / exp / "results.csv", start - 1)


def make_metrics(root, experiments, end):
    out = Path(root) / "metrics_history.csv"
    rows = []
    for exp in experiments:
        for row in csv_epochs(Path(root) / exp / "results.csv", end):
            rows.append({"experiment": exp, **{k.strip(): v for k, v in row.items()}})
    keys = ["experiment"] + list(dict.fromkeys(k for r in rows for k in r if k != "experiment"))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return out


def run_stage(stage, config_texts, *, input_root=Path("/kaggle/input"), work=Path("/kaggle/working"),
              previous_zip=None, completed_exp_zip=None, armed=False, confirmed_common_sha=None):
    """Explicitly armed entry point. Preflight is available with armed=False."""
    import yaml
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    start, end = validate_stage(stage)
    loaded, audit = config_audit(config_texts, yaml)
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        raise RuntimeError(f"Pin Ultralytics {EXPECTED_ULTRALYTICS}; got {ultralytics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU CUDA chưa bật")
    data_root, data_sha = find_dataset(input_root)
    verify_dataset(data_root, yaml)
    work = Path(work)
    root = work / "rod_main"
    logs = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    environment = {"utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    write_json(logs / "environment.json", environment)
    write_json(logs / "config_audit.json", audit)
    if not (logs / "errors.json").exists():
        write_json(logs / "errors.json", {"errors": []})
    log(logs, f"PREFLIGHT stage={stage} dataset={data_root} marker_sha256={data_sha} config_sha256={audit['common_sha256']}")
    if not armed or confirmed_common_sha != audit["common_sha256"]:
        log(logs, "MAIN TRAINING BLOCKED: set ARMED=True and exact CONFIRMED_COMMON_SHA256 after config review")
        return {"status": "PREFLIGHT_ONLY", "common_sha256": audit["common_sha256"],
                "dataset_marker_sha256": data_sha}
    yaml_path = work / "rod_final_trainval.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(data_root), "train": "train/images",
        "val": "valid/images", "nc": 25, "names": EXPECTED_NAMES}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    previous = None
    if stage > 1:
        archive = locate_previous(input_root, stage, previous_zip)
        if root.joinpath("stage_manifest.json").exists():
            raise FileExistsError("Existing stage state in /kaggle/working; start a clean session")
        safe_extract(archive, root)
        previous = json.loads((root / "stage_manifest.json").read_text(encoding="utf-8"))
        validate_previous(previous, stage, data_sha, audit["common_sha256"], torch, root)
        log(logs, f"RESTORED {archive} sha256={sha256(archive)}")
    elif any((root / exp).exists() for exp in EXPERIMENTS):
        raise FileExistsError("Stage 1 output exists; refusing overwrite")
    status = {exp: "PENDING" for exp in EXPERIMENTS}
    records = {}
    if completed_exp_zip is not None:
        # Independent backup lets E3 survive an E5 failure within the same stage.
        backup = Path(completed_exp_zip)
        if not backup.is_file():
            raise FileNotFoundError(backup)
        safe_extract(backup, root)
        partial = json.loads((root / "stage_manifest.json").read_text(encoding="utf-8"))
        if (partial.get("status") != "PARTIAL" or partial.get("stage") != stage or
            partial.get("dataset_marker_sha256") != data_sha or
            partial.get("common_config_sha256") != audit["common_sha256"]):
            raise ValueError("Partial backup stage/dataset/config mismatch")
        records = json.loads((logs / "stage_summary.json").read_text(encoding="utf-8"))
        if set(records) != set(partial["checkpoints"]):
            raise ValueError("Partial backup records/checkpoints mismatch")
        for completed in records:
            ck = root / completed / "weights" / f"resume_epoch{end:03d}.pt"
            if sha256(ck) != partial["checkpoints"][completed]["sha256"]:
                raise ValueError(f"Partial checkpoint SHA mismatch: {completed}")
            inspect_checkpoint(ck, end, torch)
            csv_epochs(root / completed / "results.csv", end)
            status[completed] = "PASS"
        if stage > 1:
            for incomplete in set(EXPERIMENTS) - set(records):
                prior_ck = root / incomplete / "weights" / f"resume_epoch{start-1:03d}.pt"
                if sha256(prior_ck) != previous["checkpoints"][incomplete]["sha256"]:
                    raise ValueError(f"Partial backup changed prior checkpoint: {incomplete}")
        log(logs, f"RESTORED PARTIAL BACKUP {backup} completed={list(records)}")
    # Extraction happens before attaching logging handlers. Otherwise extractall can
    # replace an open execution.log inode and silently lose current-session logs.
    write_json(logs / "environment.json", environment)
    write_json(logs / "config_audit.json", audit)
    write_json(logs / "errors.json", {"errors": []})
    for logger in (LOGGER, logging.getLogger("py.warnings")):
        for old in list(logger.handlers):
            if getattr(old, "_rod_stage_c", False):
                logger.removeHandler(old)
                old.close()
    handler = logging.FileHandler(logs / "execution.log", encoding="utf-8")
    handler._rod_stage_c = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").addHandler(handler)
    write_json(logs / "training_status.json", {"stage": stage, "experiments": status})
    for exp, (spec, pretrained) in EXPERIMENTS.items():
        if exp in records:
            log(logs, f"SKIP COMPLETED {exp} from verified partial backup")
            continue
        exp_dir = root / exp
        weights = exp_dir / "weights"
        recovery = weights / "recovery_full.pt"
        target = weights / f"resume_epoch{end:03d}.pt"
        expected_start = start
        try:
            log(logs, f"START {exp} stage={stage} epochs={start}-{end}")
            status[exp] = "RUNNING"
            write_json(logs / "training_status.json", {"stage": stage, "experiments": status})
            if stage == 1:
                model = YOLO(spec)
            else:
                resume_path = weights / f"resume_epoch{start-1:03d}.pt"
                inspect_checkpoint(resume_path, start - 1, torch)
                model = YOLO(str(resume_path))
            seen = []
            def on_train_start(trainer):
                if trainer.start_epoch + 1 != expected_start or trainer.epochs != 30:
                    raise RuntimeError(f"Resume epoch/schedule wrong: start={trainer.start_epoch+1}, total={trainer.epochs}")
                if trainer.scheduler.last_epoch != trainer.start_epoch - 1:
                    raise RuntimeError(f"Scheduler discontinuity: {trainer.scheduler.last_epoch}")
                if stage > 1:
                    old_lr = previous["checkpoints"][exp]["lr"]
                    new_lr = [group["lr"] for group in trainer.optimizer.param_groups]
                    if len(old_lr) != len(new_lr) or any(abs(a-b) > 1e-9 for a, b in zip(old_lr, new_lr)):
                        raise RuntimeError(f"Optimizer LR mismatch before resumed epoch: {old_lr} vs {new_lr}")
                log(logs, f"TRAIN_START {exp} actual_epoch={trainer.start_epoch+1} scheduler_last_epoch={trainer.scheduler.last_epoch}")
            def on_save(trainer):
                epoch = int(trainer.epoch) + 1
                # on_model_save follows save_model and precedes final_eval/strip_optimizer in 8.4.165.
                inspect_checkpoint(trainer.last, epoch, torch)
                atomic_copy(trainer.last, recovery)
                info = inspect_checkpoint(recovery, epoch, torch)
                seen.append(epoch)
                write_json(logs / "checkpoint_manifest.json", {"stage": stage, "experiment": exp,
                    "latest_recovery": info, "completed": records})
                if epoch == end:
                    if target.exists():
                        raise FileExistsError(target)
                    atomic_copy(trainer.last, target)
                    inspect_checkpoint(target, epoch, torch)
                log(logs, f"SAVED {exp} epoch={epoch} lr={info['lr']} recovery_sha256={info['sha256']}")
            def on_fit_end(trainer):
                if int(trainer.epoch) + 1 == end:
                    if not target.exists():
                        raise RuntimeError("Stage checkpoint missing")
                    trainer.stop = True
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_model_save", on_save)
            model.add_callback("on_fit_epoch_end", on_fit_end)
            torch.cuda.reset_peak_memory_stats(0)
            t0 = time.perf_counter()
            if stage == 1:
                kwargs = dict(loaded[exp])
                kwargs.pop("model"); kwargs.pop("task"); kwargs.pop("mode")
                kwargs["data"] = str(yaml_path)
                kwargs["project"] = str(root)
                model.train(**kwargs)
            else:
                model.train(resume=True, data=str(yaml_path), device=loaded[exp]["device"],
                            save_dir=str(exp_dir))
            ckinfo = inspect_checkpoint(target, end, torch)
            rows = csv_epochs(exp_dir / "results.csv", end)
            if seen != list(range(start, end + 1)):
                raise ValueError(f"Callback epoch sequence invalid: {seen}")
            # Verify effective arguments rather than assuming resume accepted overrides.
            args = yaml.safe_load((exp_dir / "args.yaml").read_text(encoding="utf-8"))
            mismatches = effective_args_mismatches(
                loaded[exp], args, {"data", "project", "model", "name"})
            if mismatches:
                raise ValueError(f"Effective args mismatch {exp}: {mismatches}")
            record = {"experiment": exp, "stage": stage, "start_epoch": start, "end_epoch": end,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
                "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 2**30, 3),
                "checkpoint": ckinfo, "last_metrics": rows[-1],
                "best_epoch_by_map5095": max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))["epoch"]}
            records[exp] = record
            status[exp] = "PASS"
            write_json(logs / "training_status.json", {"stage": stage, "experiments": status})
            write_json(logs / "stage_summary.json", records)
            make_metrics(root, records, end)
            shutil.copy2(root / "metrics_history.csv", logs / "metrics_history.csv")
            # Independent backup is produced before the next experiment starts.
            write_json(root / "stage_manifest.json", {"status": "PARTIAL", "stage": stage,
                "completed_epoch": end, "dataset_marker_sha256": data_sha,
                "common_config_sha256": audit["common_sha256"], "checkpoints":
                {k: v["checkpoint"] for k, v in records.items()}})
            required = [f"{exp}/weights/resume_epoch{end:03d}.pt", f"{exp}/results.csv",
                        f"{exp}/args.yaml", "stage_manifest.json", "logs/stage_summary.json",
                        "logs/config_audit.json", "logs/environment.json", "logs/execution.log"]
            backup = package(root, work / f"rod_main_stage{stage:02d}_{exp}_epoch{end:03d}.zip", required)
            log(logs, f"BACKUP {exp}: {backup}")
        except Exception as exc:
            status[exp] = "FAIL"
            write_json(logs / "training_status.json", {"stage": stage, "experiments": status})
            write_json(logs / "errors.json", {"experiment": exp, "stage": stage,
                "error": repr(exc), "traceback": traceback.format_exc()})
            log(logs, f"FAIL {exp}: {exc!r}")
            raise
    manifest = {"status": "PASS", "stage": stage, "completed_epoch": end,
        "dataset_marker_sha256": data_sha, "common_config_sha256": audit["common_sha256"],
        "ultralytics": ultralytics.__version__, "checkpoints":
        {k: v["checkpoint"] for k, v in records.items()}, "records": records}
    write_json(root / "stage_manifest.json", manifest)
    write_json(logs / "checkpoint_manifest.json", manifest)
    (logs / "stage_c_report.md").write_text(
        f"# ROD Stage C — Stage {stage}\n\nStatus: PASS in current Kaggle runtime. "
        f"Epochs: {start}–{end}. Dataset marker SHA256: `{data_sha}`. "
        f"Config SHA256: `{audit['common_sha256']}`.\n\n"
        "Back up the stage ZIP through Kaggle Version output and download it before leaving this runtime.\n",
        encoding="utf-8")
    required = ["stage_manifest.json", "logs/training_status.json", "logs/checkpoint_manifest.json",
                "logs/environment.json", "logs/config_audit.json", "logs/metrics_history.csv",
                "logs/stage_summary.json", "logs/execution.log", "logs/stage_c_report.md", "logs/errors.json"]
    # metrics_history.csv is stored at root; keep one standardized copy in logs.
    shutil.copy2(root / "metrics_history.csv", logs / "metrics_history.csv")
    for exp in EXPERIMENTS:
        required += [f"{exp}/weights/resume_epoch{end:03d}.pt", f"{exp}/results.csv", f"{exp}/args.yaml"]
    archive = package(root, work / f"rod_main_stage{stage:02d}_epoch{end:03d}.zip", required)
    log(logs, f"STAGE PASS {stage}: {archive}")
    return {"status": "PASS", "stage": stage, "archive": archive, "experiments": records}
