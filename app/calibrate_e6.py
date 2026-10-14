"""Collect E6 predictions on ROD Final VALID and select its operating threshold."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from app.calibrate_confidence import (
    COLLECTION_CONF, MATCH_IOU, THRESHOLDS, choose_threshold,
    ground_truth, score_predictions, sha256,
)
from app.confidence import CONFIG_PATH
from app.model_adapters import CHECKPOINTS, DATA_YAML, load_adapter, require_checkpoint


ROOT = Path(__file__).resolve().parents[1]
OPTION = "Faster R-CNN (E6)"
IMAGES = ROOT / "data/processed/rod_detection_final/valid/images"
LABELS = ROOT / "data/processed/rod_detection_final/valid/labels"
OUT = ROOT / "results/E6_Faster_RCNN/confidence"
PREDICTIONS = OUT / "E6_valid_predictions.jsonl"
METADATA = OUT / "E6_valid_metadata.json"
SWEEP = OUT / "E6_threshold_sweep.csv"


def valid_paths() -> list[Path]:
    paths = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != 3318:
        raise RuntimeError(f"VALID phải có 3318 ảnh, tìm thấy {len(paths)}")
    if any(not (LABELS / f"{p.stem}.txt").is_file() for p in paths):
        raise RuntimeError("VALID có ảnh thiếu nhãn")
    return paths


def collect(max_images: int = 3318) -> list[dict]:
    paths = valid_paths()
    if not 1 <= max_images <= len(paths):
        raise ValueError("max_images ngoài phạm vi")
    checkpoint = require_checkpoint(OPTION)
    metadata = {
        "model": "E6", "split": "valid", "image_count": len(paths),
        "checkpoint_sha256": sha256(checkpoint),
        "data_yaml_sha256": sha256(DATA_YAML),
        "collection_conf": COLLECTION_CONF, "nms_iou": 0.7,
        "max_det": 300, "image_size": [640, 960],
        "class_mapping": "torchvision 1..25 -> ROD 0..24",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if METADATA.exists():
        if json.loads(METADATA.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("Metadata prediction E6 không khớp checkpoint/protocol")
    elif PREDICTIONS.exists():
        raise RuntimeError("Prediction E6 tồn tại nhưng thiếu metadata")
    else:
        METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = []
    if PREDICTIONS.exists():
        with PREDICTIONS.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) > len(paths) or any(r["image"] != p.name for r, p in zip(records, paths)):
            raise RuntimeError("Prediction E6 không khớp thứ tự VALID")
    if len(records) >= max_images:
        return records
    adapter = load_adapter(OPTION)
    started = time.perf_counter()
    with PREDICTIONS.open("a", encoding="utf-8") as stream:
        for index in range(len(records), max_images):
            path = paths[index]
            with Image.open(path) as source:
                image = source.convert("RGB")
            prediction = adapter.predict(image, COLLECTION_CONF)
            record = {
                "image": path.name, "width": image.width, "height": image.height,
                "gt": ground_truth(LABELS / f"{path.stem}.txt", image.width, image.height),
                "pred": [
                    [class_id, score, *box]
                    for class_id, score, box in zip(
                        prediction.class_ids, prediction.scores,
                        prediction.boxes_xyxy, strict=True,
                    )
                ],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
            if (index + 1) % 50 == 0 or index + 1 == max_images:
                print(f"E6 VALID {index + 1}/{len(paths)}; elapsed={time.perf_counter()-started:.0f}s", flush=True)
    return records


def calibrate() -> None:
    records = collect()
    if len(records) != 3318 or len({record["image"] for record in records}) != 3318:
        raise RuntimeError("Calibration E6 chưa đủ 3318 ảnh VALID duy nhất")
    scores = score_predictions(records, THRESHOLDS)
    selected = choose_threshold(scores)
    with SWEEP.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f2"])
        for item in scores:
            writer.writerow([item.threshold, item.tp, item.fp, item.fn,
                             item.precision, item.recall, item.f2])
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["E6"]["checkpoint_sha256"] != CHECKPOINTS[OPTION][1]:
        raise RuntimeError("Checkpoint E6 không khớp confidence config")
    config["E6"].update({
        "threshold": selected.threshold, "calibrated": True,
        "source": "ROD_FINAL_VALID",
        "metrics": {
            "precision": selected.precision, "recall": selected.recall, "f2": selected.f2,
            "tp": selected.tp, "fp": selected.fp, "fn": selected.fn,
            "match_iou": MATCH_IOU, "valid_images": len(records),
        },
        "evidence": str(SWEEP.relative_to(ROOT)).replace("\\", "/"),
        "prediction_sha256": sha256(PREDICTIONS),
        "threshold_sweep_sha256": sha256(SWEEP),
        "data_yaml_sha256": sha256(DATA_YAML),
    })
    temp = CONFIG_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(CONFIG_PATH)
    print(f"E6 VALID threshold={selected.threshold:.2f} P={selected.precision:.4f} "
          f"R={selected.recall:.4f} F2={selected.f2:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=3318)
    args = parser.parse_args()
    if args.max_images == 3318:
        calibrate()
    else:
        collect(args.max_images)


if __name__ == "__main__":
    main()
