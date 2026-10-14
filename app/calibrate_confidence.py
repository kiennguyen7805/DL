"""One-pass VALID predictions, then a reusable micro-F2 threshold sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.confidence import CONFIG_PATH
from app.detector import MODEL_OPTIONS, ROOT, checkpoint_path, load_model


VALID_IMAGES = ROOT / "data/processed/rod_detection_final/valid/images"
VALID_LABELS = ROOT / "data/processed/rod_detection_final/valid/labels"
RESULTS_BY_MODEL = {
    "E3": ROOT / "results/E3_YOLO11n/confidence",
    "E4": ROOT / "results/E4_YOLO11s/confidence",
}
COLLECTION_CONF = 0.01
NMS_IOU = 0.7
MATCH_IOU = 0.5
MAX_DET = 300
IMAGE_SIZE = 640
MODEL_IDS = dict(zip(MODEL_OPTIONS[:2], ("E3", "E4"), strict=True))
THRESHOLDS = [round(i / 100, 2) for i in range(1, 96)]


@dataclass(frozen=True)
class Score:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f2: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    if not label_path.is_file():
        raise RuntimeError(f"Thiếu nhãn VALID: {label_path}")
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 or int(values[0]) != values[0] or not 0 <= values[0] < 25:
            raise RuntimeError(f"Nhãn YOLO không hợp lệ: {label_path}")
        class_id, x, y, w, h = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise RuntimeError(f"Bbox YOLO không hợp lệ: {label_path}")
        rows.append([int(class_id), (x - w / 2) * width, (y - h / 2) * height,
                     (x + w / 2) * width, (y + h / 2) * height])
    return rows


def intersection_over_union(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def score_predictions(records: list[dict], thresholds: list[float] = THRESHOLDS) -> list[Score]:
    """Match each class-correct prediction to at most one GT at IoU >= 0.5."""
    outcomes: list[tuple[float, bool]] = []
    total_gt = 0
    for record in records:
        gt = record["gt"]
        total_gt += len(gt)
        used: set[int] = set()
        for pred in sorted(record["pred"], key=lambda row: -row[1]):
            cls, conf, *box = pred
            eligible = [
                (intersection_over_union(box, item[1:]), index)
                for index, item in enumerate(gt)
                if index not in used and item[0] == cls
            ]
            best_iou, best_index = max(eligible, default=(0.0, -1))
            matched = best_iou >= MATCH_IOU
            if matched:
                used.add(best_index)
            outcomes.append((conf, matched))
    scores = []
    for threshold in thresholds:
        selected = [matched for confidence, matched in outcomes if confidence >= threshold]
        tp = sum(selected)
        fp = len(selected) - tp
        fn = total_gt - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision or recall else 0.0
        scores.append(Score(threshold, tp, fp, fn, precision, recall, f2))
    return scores


def choose_threshold(scores: list[Score]) -> Score:
    if not scores:
        raise RuntimeError("Không có ngưỡng để đánh giá.")
    # F2 weights recall four times as strongly as precision; tie favors fewer FP.
    return max(scores, key=lambda score: (score.f2, score.precision, score.threshold))


def _paths() -> list[Path]:
    paths = sorted(p for p in VALID_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != 3318:
        raise RuntimeError(f"VALID cần 3318 ảnh, tìm thấy {len(paths)}.")
    return paths


def collect(option: str, *, batch_size: int = 8) -> tuple[list[dict], float]:
    model_id = MODEL_IDS[option]
    results = RESULTS_BY_MODEL[model_id]
    checkpoint = checkpoint_path(option)
    dataset_yaml = ROOT / "data/processed/rod_detection_final/data.yaml"
    paths = _paths()
    results.mkdir(parents=True, exist_ok=True)
    prediction_file = results / f"{model_id}_valid_predictions.jsonl"
    metadata_file = results / f"{model_id}_valid_metadata.json"
    metadata = {
        "model": model_id,
        "split": "valid",
        "image_count": len(paths),
        "checkpoint_sha256": sha256(checkpoint),
        "data_yaml_sha256": sha256(dataset_yaml),
        "collection_conf": COLLECTION_CONF,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
        "imgsz": IMAGE_SIZE,
    }
    if metadata_file.exists():
        if json.loads(metadata_file.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError(f"Metadata cache {model_id} không khớp; không ghi đè prediction cũ.")
    else:
        if prediction_file.exists():
            raise RuntimeError(f"Có prediction {model_id} nhưng thiếu metadata.")
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    if prediction_file.exists():
        with prediction_file.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) > len(paths) or any(record["image"] != path.name for record, path in zip(records, paths)):
            raise RuntimeError(f"Cache prediction {model_id} không đúng thứ tự VALID.")
    if len(records) == len(paths):
        print(f"{model_id}: dùng lại {len(records)} prediction VALID đã lưu", flush=True)
        return records, 0.0

    model, _ = load_model(option)
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    with prediction_file.open("a", encoding="utf-8") as stream:
        for offset in range(len(records), len(paths), batch_size):
            batch = paths[offset:offset + batch_size]
            results = model.predict(
                source=[str(path) for path in batch],
                conf=COLLECTION_CONF, iou=NMS_IOU, max_det=MAX_DET,
                imgsz=IMAGE_SIZE, device=device, batch=batch_size,
                verbose=False, save=False,
            )
            if len(results) != len(batch):
                raise RuntimeError("Số prediction không khớp số ảnh trong batch.")
            for path, result in zip(batch, results, strict=True):
                if Path(result.path).name != path.name:
                    raise RuntimeError("Thứ tự prediction VALID không khớp.")
                height, width = result.orig_shape
                boxes = result.boxes
                predictions = [
                    [int(cls), float(conf), *map(float, coords)]
                    for cls, conf, coords in zip(
                        boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                        boxes.xyxy.cpu().tolist(), strict=True,
                    )
                ] if boxes is not None else []
                record = {
                    "image": path.name,
                    "width": width, "height": height,
                    "gt": ground_truth(VALID_LABELS / f"{path.stem}.txt", width, height),
                    "pred": predictions,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            stream.flush()
            if len(records) % 200 < batch_size or len(records) == len(paths):
                print(f"{model_id}: VALID {len(records)}/{len(paths)}; {time.perf_counter()-started:.0f}s", flush=True)
    return records, time.perf_counter() - started


def calibrate(option: str, *, batch_size: int = 8) -> Score:
    model_id = MODEL_IDS[option]
    results = RESULTS_BY_MODEL[model_id]
    records, inference_seconds = collect(option, batch_size=batch_size)
    if len(records) != 3318 or len({record["image"] for record in records}) != 3318:
        raise RuntimeError("Calibration chưa đủ 3318 ảnh VALID duy nhất.")
    scores = score_predictions(records)
    chosen = choose_threshold(scores)
    _, names = load_model(option)
    table = results / f"{model_id}_threshold_sweep.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f2"])
        for score in scores:
            writer.writerow([score.threshold, score.tp, score.fp, score.fn,
                             score.precision, score.recall, score.f2])
    class_table = results / f"{model_id}_per_class_exploration.csv"
    with class_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name", "valid_gt", "global_threshold",
                         "global_precision", "global_recall", "global_f2",
                         "exploratory_threshold", "exploratory_precision",
                         "exploratory_recall", "exploratory_f2", "applied"])
        for class_id in range(25):
            subset = [
                {"gt": [gt for gt in record["gt"] if gt[0] == class_id],
                 "pred": [pred for pred in record["pred"] if pred[0] == class_id]}
                for record in records
            ]
            count = sum(len(record["gt"]) for record in subset)
            global_score = score_predictions(subset, [chosen.threshold])[0]
            exploratory = choose_threshold(score_predictions(subset)) if count else global_score
            writer.writerow([
                class_id, names[class_id], count, chosen.threshold,
                global_score.precision, global_score.recall, global_score.f2,
                exploratory.threshold, exploratory.precision,
                exploratory.recall, exploratory.f2, "NO",
            ])
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config[model_id]["checkpoint_sha256"] != sha256(checkpoint_path(option)):
        raise RuntimeError("Checkpoint đổi so với confidence config.")
    config[model_id].update({
        "threshold": chosen.threshold, "calibrated": True, "source": "ROD_FINAL_VALID",
        "metrics": {
            "precision": chosen.precision, "recall": chosen.recall, "f2": chosen.f2,
            "tp": chosen.tp, "fp": chosen.fp, "fn": chosen.fn,
            "match_iou": MATCH_IOU, "valid_images": len(records),
        },
        "evidence": str(table.relative_to(ROOT)).replace("\\", "/"),
        "prediction_sha256": sha256(results / f"{model_id}_valid_predictions.jsonl"),
        "threshold_sweep_sha256": sha256(table),
        "data_yaml_sha256": sha256(ROOT / "data/processed/rod_detection_final/data.yaml"),
    })
    temp = CONFIG_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(CONFIG_PATH)
    print(
        f"{model_id}: selected={chosen.threshold:.2f} "
        f"P={chosen.precision:.4f} R={chosen.recall:.4f} F2={chosen.f2:.4f} "
        f"TP={chosen.tp} FP={chosen.fp} FN={chosen.fn} "
        f"inference_seconds={inference_seconds:.1f}",
        flush=True,
    )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["E3", "E4", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size phải >= 1")
    selected = MODEL_OPTIONS[:2] if args.model == "both" else (next(
        option for option, model_id in MODEL_IDS.items() if model_id == args.model
    ),)
    for option in selected:
        calibrate(option, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
