"""Offline VALID calibration then locked TEST for the supplied E3/E4 Kaggle ZIPs."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from src.kaggle_hf_shared import atomic_json, sha256_file
from scripts.download_hf_checkpoints import DATASET_HASH, MODELS, REQUIRED, safe_filename

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"valid": 3318, "test": 1611}
INFERENCE = {"conf": 0.01, "iou": 0.7, "max_det": 300, "imgsz": 640}


def prepare_checkpoint(source, experiment, output):
    """Select artifacts from the verified latest snapshot, including compact aliases."""
    folder, model, config = MODELS[experiment]
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate ZIP entries")
        manifests = [name for name in names if name.endswith("/last_hf_manifest.json")]
        if len(manifests) != 1:
            raise RuntimeError("Expected one latest training manifest in " + str(source))
        manifest = json.loads(archive.read(manifests[0]))
        if (manifest.get("schema") != 2 or manifest.get("folder") != folder
                or manifest.get("model") != model or manifest.get("config_sha256") != config
                or manifest.get("dataset_sha256") != DATASET_HASH
                or type(manifest.get("completed_epoch")) is not int or manifest["completed_epoch"] != 30
                or manifest.get("is_complete") is not True or manifest.get("target_epochs") != 30):
            raise RuntimeError("Not the expected COMPLETE 30-epoch checkpoint: " + experiment)
        snap = manifest["snapshot"]
        prefix = folder + "/snapshots/"
        if not snap.startswith(prefix):
            raise RuntimeError("Invalid snapshot path")
        safe_filename(snap[len(prefix):])
        hashes = manifest["sha256"]
        aliases = manifest.get("artifact_aliases", {})
        if aliases and (aliases != {"best.pt": "recovery_full.pt"} or "best.pt" in hashes):
            raise RuntimeError("Unexpected checkpoint aliases")
        if not REQUIRED.issubset(set(hashes) | set(aliases)):
            raise RuntimeError("Missing snapshot artifacts")
        destination = Path(output) / experiment / "checkpoint"
        destination.mkdir(parents=True, exist_ok=True)
        for name, expected in hashes.items():
            safe_filename(name)
            # ZIP may also contain source snapshots and stripped native best.pt.
            # Match only this manifest's immutable snapshot directory.
            candidates = [entry for entry in names if entry.endswith("/" + snap + "/" + name)]
            if not candidates:
                raise RuntimeError("Missing snapshot artifact in ZIP: " + name)
            path = destination / name
            if path.is_file() and sha256_file(path) == expected:
                continue
            temp = destination / (name + ".tmp")
            with archive.open(candidates[0]) as src, temp.open("wb") as dest:
                shutil.copyfileobj(src, dest)
            if sha256_file(temp) != expected:
                raise RuntimeError("ZIP artifact SHA256 mismatch: " + name)
            os.replace(temp, path)
        if aliases:
            target = destination / "best.pt"
            expected = hashes["recovery_full.pt"]
            if not target.exists() or sha256_file(target) != expected:
                shutil.copyfile(destination / "recovery_full.pt", target)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in
                csv.DictReader(io.StringIO((destination / "results.csv").read_text(encoding="utf-8-sig")))]
        epochs = [int(float(row["epoch"])) for row in rows]
        if epochs != list(range(1, 31)):
            raise RuntimeError("Training CSV must contain contiguous epochs 1..30")
        atomic_json(destination / "manifest.json", manifest)
        audit = {"model": experiment, "archive": str(Path(source).resolve()),
                 "archive_sha256": sha256_file(source), "completed_epoch": 30,
                 "snapshot": snap, "best_sha256": sha256_file(destination / "best.pt"),
                 "dataset_sha256": DATASET_HASH, "training_last_row": rows[-1],
                 "best_role": "recovery_full.pt alias" if aliases else "best.pt"}
        atomic_json(destination / "audit.json", audit)
        print(f"{experiment}: verified checkpoint 30/30; best SHA256={audit['best_sha256']}", flush=True)
        return destination / "best.pt", audit


def read_cache(path):
    """Recover only a torn final JSONL write; reject corruption in earlier rows."""
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                records.append(json.loads(line))
                if not line.endswith(b"\n"):
                    stream.write(b"\n")
            except (ValueError, UnicodeDecodeError):
                if stream.read(1):
                    raise RuntimeError("Corrupt prediction cache: " + str(path)) from None
                stream.truncate(position)
                print("Recovered interrupted final cache line:", path, flush=True)
                break
    return records


def collect(model, experiment, split, dataset, output, metadata, batch_size, device):
    from app.calibrate_confidence import ground_truth
    images, labels = dataset / split / "images", dataset / split / "labels"
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != COUNTS[split] or any(not (labels / (p.stem + ".txt")).is_file() for p in paths):
        raise RuntimeError(f"{split}: expected {COUNTS[split]} images and matching labels")
    directory = output / experiment / split
    directory.mkdir(parents=True, exist_ok=True)
    cache, meta_path = directory / "predictions.jsonl", directory / "metadata.json"
    expected = {**metadata, "model": experiment, "split": split, "images": len(paths),
                "inference": INFERENCE}
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Prediction cache provenance mismatch: " + str(directory))
    elif cache.exists():
        raise RuntimeError("Prediction cache lacks metadata")
    else:
        atomic_json(meta_path, expected)
    records = read_cache(cache)
    if len(records) > len(paths) or any(row["image"] != path.name for row, path in zip(records, paths)):
        raise RuntimeError("Cached image order mismatch")
    print(f"{experiment} {split.upper()}: resume {len(records)}/{len(paths)}", flush=True)
    started = last_log = time.monotonic()
    initial_count = len(records)
    with cache.open("a", encoding="utf-8") as stream:
        for offset in range(len(records), len(paths), batch_size):
            batch = paths[offset:offset + batch_size]
            results = model.predict(source=[str(p) for p in batch], **INFERENCE,
                                    device=device, batch=batch_size, verbose=False, save=False)
            if len(results) != len(batch):
                raise RuntimeError("Prediction count mismatch")
            for path, result in zip(batch, results, strict=True):
                if Path(result.path).name != path.name:
                    raise RuntimeError("Prediction image mismatch")
                height, width = result.orig_shape
                boxes = result.boxes
                predictions = [] if boxes is None else [
                    [int(cls), float(score), *map(float, box)]
                    for cls, score, box in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                                               boxes.xyxy.cpu().tolist(), strict=True)]
                row = {"image": path.name, "width": width, "height": height,
                       "gt": ground_truth(labels / (path.stem + ".txt"), width, height), "pred": predictions}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(row)
            now = time.monotonic()
            if offset == initial_count or len(records) % 100 < batch_size or now - last_log >= 60 or len(records) == len(paths):
                elapsed = max(now - started, 1e-6)
                rate = (len(records) - initial_count) / elapsed
                eta = (len(paths) - len(records)) / max(rate, 1e-6)
                print(f"{experiment} {split.upper()} {len(records)}/{len(paths)} "
                      f"({100 * len(records) / len(paths):.1f}%) elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
                last_log = now
    return records, cache


def save_metrics(records, threshold, output, provenance, names):
    from app.evaluator import evaluate_records
    print("Computing mAP:", output, flush=True)
    metrics = evaluate_records(records, threshold)
    metrics.update(provenance)
    metrics["f2"] = (5 * metrics["precision"] * metrics["recall"] /
                     (4 * metrics["precision"] + metrics["recall"])) if metrics["precision"] or metrics["recall"] else 0.0
    atomic_json(output / "metrics.json", metrics)
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_id", "class_name", "gt_count", "ap50", "ap50_95", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows({**row, "class_name": names[row["class_id"]]} for row in metrics["per_class"])
    print(f"{provenance['model']} {provenance['split'].upper()} mAP50={metrics['map50']:.6f} "
          f"mAP50-95={metrics['map50_95']:.6f} P={metrics['precision']:.6f} R={metrics['recall']:.6f}", flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-zip", type=Path, default=ROOT / "results_zips" / "results_e3.zip")
    parser.add_argument("--e4-zip", type=Path, default=ROOT / "results_zips" / "results_e4.zip")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["E3", "E4"])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/rod_detection_final")
    parser.add_argument("--output", type=Path, default=ROOT / "results/30epochs")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu or GPU index such as 0")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from app.calibrate_confidence import choose_threshold, score_predictions
    from app.evaluator import METRIC_PROTOCOL

    if ultralytics.__version__ != "8.4.165":
        raise RuntimeError("Use ultralytics==8.4.165 to match these training checkpoints")
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device={device}; torch={torch.__version__}; ultralytics={ultralytics.__version__}", flush=True)
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch installation has no CUDA support")
    yaml_path = args.dataset / "data.yaml"
    if sha256_file(yaml_path) != DATASET_HASH:
        raise RuntimeError("Dataset YAML differs from the frozen training dataset")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw_names = config["names"]
    names = {int(k): v for k, v in raw_names.items()} if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    if config.get("nc") != 25 or set(names) != set(range(25)):
        raise RuntimeError("Expected 25 ROD classes")
    models = list(dict.fromkeys(args.models))
    prepared, summaries = {}, {}
    for experiment in models:
        checkpoint, audit = prepare_checkpoint(args.e3_zip if experiment == "E3" else args.e4_zip,
                                               experiment, args.output)
        metadata = {"checkpoint_sha256": audit["best_sha256"], "data_yaml_sha256": DATASET_HASH,
                    "ultralytics": ultralytics.__version__, "torch": torch.__version__, "device": device}
        model = YOLO(str(checkpoint))
        actual = {int(k): v for k, v in model.names.items()}
        if actual != names or model.task != "detect" or model.model.yaml.get("scale") != ("n" if experiment == "E3" else "s"):
            raise RuntimeError("Checkpoint architecture or class mapping mismatch")
        records, cache = collect(model, experiment, "valid", args.dataset, args.output, metadata, args.batch, device)
        print(f"{experiment}: selecting threshold from VALID only (micro F2)...", flush=True)
        scores = score_predictions(records)
        selected = choose_threshold(scores)
        valid_dir = args.output / experiment / "valid"
        with (valid_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(selected)))
            writer.writeheader()
            writer.writerows(asdict(score) for score in scores)
        evidence = {"model": experiment, "split": "valid", "checkpoint_sha256": audit["best_sha256"],
                    "prediction_sha256": sha256_file(cache), "status": "FULL_30EPOCH"}
        summaries[experiment] = {"valid": save_metrics(records, selected.threshold, valid_dir, evidence, names)}
        lock = {**metadata, "model": experiment, "completed_epoch": 30,
                "threshold": selected.threshold, "threshold_selection": "VALID micro-F2; ties precision then threshold",
                "valid_prediction_sha256": sha256_file(cache), "inference": INFERENCE,
                "metric_protocol": METRIC_PROTOCOL}
        lock_path = args.output / experiment / "evaluation_lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise RuntimeError("Existing 30-epoch TEST lock differs; refusing overwrite")
        atomic_json(lock_path, lock)
        prepared[experiment] = (checkpoint, metadata, lock, sha256_file(lock_path))
        del model, records
    # Complete and lock all requested VALID runs before opening TEST images.
    for experiment, (checkpoint, metadata, lock, lock_sha) in prepared.items():
        lock_path = args.output / experiment / "evaluation_lock.json"
        if sha256_file(lock_path) != lock_sha or sha256_file(checkpoint) != metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint/lock changed before TEST")
        valid_cache = args.output / experiment / "valid/predictions.jsonl"
        if sha256_file(valid_cache) != lock["valid_prediction_sha256"]:
            raise RuntimeError("VALID evidence changed before TEST")
        model = YOLO(str(checkpoint))
        records, cache = collect(model, experiment, "test", args.dataset, args.output,
                                 {**metadata, "lock_sha256": lock_sha}, args.batch, device)
        evidence = {"model": experiment, "split": "test", "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "lock_sha256": lock_sha, "prediction_sha256": sha256_file(cache), "status": "FINAL_LOCKED_30EPOCH"}
        summaries[experiment]["test"] = save_metrics(records, lock["threshold"], args.output / experiment / "test", evidence, names)
        del model, records
    summary_path = args.output / ("summary_" + "_".join(models) + ".json")
    atomic_json(summary_path, summaries)
    with summary_path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["model", "split", "images", "operating_threshold", "map50", "map50_95", "precision", "recall", "f2"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics for split in summaries.values() for metrics in split.values())
    print("DONE:", summary_path, flush=True)


if __name__ == "__main__":
    main()
