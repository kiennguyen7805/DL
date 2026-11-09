"""Common VALID evaluation and frozen TEST protocol for ROD models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.confidence import CONFIG_PATH, auto_confidence
from app.evaluator import IOU_THRESHOLDS, MAX_DET, METRIC_PROTOCOL, OPERATING_MAX_DET, evaluate_records
from app.model_adapters import CHECKPOINTS, DATA_YAML, ROOT, require_checkpoint, rod_names, sha256_file


IDS = {"E3": "YOLO11n (E3, 30 epoch)", "E4": "YOLO11s (E4, 30 epoch)", "E6": "Faster R-CNN (E6)"}
FOLDERS = {"E3": "E3_YOLO11n", "E4": "E4_YOLO11s", "E6": "E6_Faster_RCNN"}
LOCK_PATH = ROOT / "app/evaluation_lock.json"
PROTOCOL = {
    "evaluation_split_order": ["valid", "test"],
    "collection_conf": 0.01, "nms_iou": 0.7, "collection_max_det": 300,
    "evaluation_max_det": MAX_DET, "operating_max_det": OPERATING_MAX_DET,
    "iou_thresholds": list(IOU_THRESHOLDS),
    "ap_recall_points": 101, "operating_match_iou": 0.5,
    "class_mapping": "E3/E4 0..24; E6 1..25 -> ROD 0..24",
    "gt": "ROD Final A4 YOLO xywh converted to pixel xyxy",
    "area": "all", "crowd_ignore": False,
    "ap_scope": "predictions with model score >= 0.01; not the official COCO scorer",
}


def _path(model_id: str, filename: str) -> Path:
    return ROOT / "results" / FOLDERS[model_id] / "metrics" / filename


def cache_path(model_id: str) -> Path:
    current = ROOT / "results/30epochs" / model_id / "valid/predictions.jsonl"
    if current.is_file():
        return current
    return ROOT / "results" / FOLDERS[model_id] / "confidence" / f"{model_id}_valid_predictions.jsonl"


def load_valid_records(model_id: str, *, limit: int | None = None) -> list[dict]:
    if model_id not in IDS:
        raise ValueError("Unknown model")
    path = cache_path(model_id)
    current = path.name == "predictions.jsonl"
    meta_path = path.with_name("metadata.json" if current else f"{model_id}_valid_metadata.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing VALID prediction/metadata for {model_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if current:
        inference = meta.get("inference", {})
        meta = {**meta, "image_count": meta.get("images"),
                "collection_conf": inference.get("conf"),
                "nms_iou": inference.get("iou"), "max_det": inference.get("max_det")}
    if (meta.get("model") != model_id or meta.get("split") != "valid" or
            meta.get("image_count") != 3318 or
            meta.get("checkpoint_sha256") != CHECKPOINTS[IDS[model_id]][1] or
            meta.get("data_yaml_sha256") != sha256_file(DATA_YAML) or
            meta.get("collection_conf") != 0.01 or meta.get("nms_iou") != 0.7 or
            meta.get("max_det") != 300):
        raise RuntimeError(f"VALID cache metadata mismatch: {model_id}")
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != len({row["image"] for row in records}):
        raise RuntimeError(f"Duplicate VALID image in cache: {model_id}")
    if limit is None and len(records) != 3318:
        raise RuntimeError(f"VALID cache incomplete: {model_id}: {len(records)}/3318")
    return records[:limit] if limit is not None else records


def evaluate_valid(model_id: str, *, limit: int | None = None) -> dict:
    option = IDS[model_id]
    require_checkpoint(option)
    setting = auto_confidence(option)
    if setting.checkpoint_sha256 != CHECKPOINTS[option][1]:
        raise RuntimeError("Confidence configuration checkpoint mismatch")
    records = load_valid_records(model_id, limit=limit)
    metrics = evaluate_records(records, setting.threshold)
    metrics.update({
        "model": model_id, "split": "valid", "status": "FULL" if limit is None else "PARTIAL_CHECK",
        "calibrated": setting.calibrated,
        "checkpoint_sha256": CHECKPOINTS[option][1],
        "prediction_sha256": sha256_file(cache_path(model_id)),
        "data_yaml_sha256": sha256_file(DATA_YAML),
    })
    if limit is None:
        output = _path(model_id, "common_valid.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        table = _path(model_id, "common_valid_per_class.csv")
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_id", "class_name", "gt_count", "ap50", "ap50_95", "tp", "fp", "fn"])
            names = rod_names()
            for item in metrics["per_class"]:
                writer.writerow([item["class_id"], names[item["class_id"]], item["gt_count"],
                                 item["ap50"], item["ap50_95"], item["tp"], item["fp"], item["fn"]])
    return metrics


def fingerprint() -> dict:
    return {
        "data_yaml_sha256": sha256_file(DATA_YAML),
        "confidence_config_sha256": sha256_file(CONFIG_PATH),
        "checkpoints": {model_id: CHECKPOINTS[option][1] for model_id, option in IDS.items()},
        "thresholds": {model_id: auto_confidence(option).threshold for model_id, option in IDS.items()},
        "protocol": PROTOCOL,
    }


def freeze() -> dict:
    if LOCK_PATH.exists():
        raise FileExistsError(f"Evaluation lock already exists: {LOCK_PATH}")
    for model_id, option in IDS.items():
        report = _path(model_id, "common_valid.json")
        if not report.is_file():
            raise RuntimeError(f"Missing full VALID evaluation: {model_id}")
        result = json.loads(report.read_text(encoding="utf-8"))
        if result.get("status") != "FULL" or result.get("images") != 3318 or not result.get("calibrated"):
            raise RuntimeError(f"VALID evaluation/calibration incomplete: {model_id}")
        if result.get("protocol") != METRIC_PROTOCOL:
            raise RuntimeError(f"VALID evaluator protocol mismatch: {model_id}")
        if (result.get("checkpoint_sha256") != CHECKPOINTS[option][1] or
                result.get("data_yaml_sha256") != sha256_file(DATA_YAML) or
                result.get("prediction_sha256") != sha256_file(cache_path(model_id)) or
                result.get("operating_threshold") != auto_confidence(option).threshold):
            raise RuntimeError(f"VALID evidence changed: {model_id}")
        require_checkpoint(option)
    locked = fingerprint()
    LOCK_PATH.write_text(json.dumps(locked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return locked


def verify_lock() -> dict:
    if not LOCK_PATH.is_file():
        raise RuntimeError("Missing frozen VALID evaluation lock; TEST is blocked")
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if locked.get("schema") == "imported-per-model-valid-locks-v1":
        # This is a local index of original locks, not a new pre-TEST lock.
        expected = {key: locked.get(key) for key in fingerprint()}
        if expected != fingerprint():
            raise RuntimeError("Evaluation lock fingerprint mismatch; TEST is blocked")
        for model_id in IDS:
            folder = ROOT / "results/30epochs" / model_id
            source = folder / "evaluation_lock.json"
            if sha256_file(source) != locked["source_lock_sha256"][model_id]:
                raise RuntimeError(f"Imported VALID lock changed: {model_id}")
            original = json.loads(source.read_text(encoding="utf-8"))
            if (original["checkpoint_sha256"] != locked["checkpoints"][model_id] or
                    original["threshold"] != locked["thresholds"][model_id] or
                    original["data_yaml_sha256"] != locked["data_yaml_sha256"] or
                    original["valid_prediction_sha256"] != sha256_file(folder / "valid/predictions.jsonl")):
                raise RuntimeError(f"Imported VALID evidence mismatch: {model_id}")
            for field, relative in [("valid_metrics_sha256", "valid/metrics.json"),
                                    ("threshold_sweep_sha256", "valid/threshold_sweep.csv")]:
                if field in original and original[field] != sha256_file(folder / relative):
                    raise RuntimeError(f"Imported VALID evidence changed: {model_id}: {field}")
    elif locked != fingerprint():
        raise RuntimeError("Evaluation lock fingerprint mismatch; TEST is blocked")
    for model_id, option in IDS.items():
        require_checkpoint(option)
        if not auto_confidence(option).calibrated:
            raise RuntimeError(f"Uncalibrated model {model_id}; TEST is blocked")
    return locked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*IDS, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="VALID pipeline check only; no saved metric")
    parser.add_argument("--freeze", action="store_true", help="Lock protocol after three full VALID runs")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(), ensure_ascii=False, indent=2))
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be >= 1")
        for model_id in (IDS if args.model == "all" else (args.model,)):
            metrics = evaluate_valid(model_id, limit=args.limit)
            print(model_id, metrics["status"], metrics["images"],
                  "mAP50", metrics["map50"], "mAP50-95", metrics["map50_95"],
                  "P", metrics["precision"], "R", metrics["recall"])


if __name__ == "__main__":
    main()
