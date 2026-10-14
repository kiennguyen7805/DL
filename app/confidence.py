"""Per-model VALID-calibrated confidence selection with an explicit fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("confidence_config.json")
MODEL_IDS = {
    "YOLO11n (E3)": "E3",
    "YOLO11s (E4)": "E4",
    "Faster R-CNN (E6)": "E6",
}


@dataclass(frozen=True)
class AutoConfidence:
    threshold: float
    calibrated: bool
    model_id: str
    precision: float | None
    recall: float | None
    f2: float | None
    checkpoint_sha256: str


def _valid_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RuntimeError("Confidence threshold không hợp lệ trong cấu hình.")
    number = float(value)
    if not 0.001 <= number <= 0.99:
        raise RuntimeError("Confidence threshold ngoài phạm vi 0,001–0,99.")
    return number


def auto_confidence(model_option: str, *, config_path: Path = CONFIG_PATH) -> AutoConfidence:
    if model_option not in MODEL_IDS:
        raise ValueError("Mô hình không hợp lệ.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise RuntimeError("Schema Auto Confidence không hợp lệ.")
    model_id = MODEL_IDS[model_option]
    item = config[model_id]
    if item.get("mode") != "auto":
        raise RuntimeError("Mô hình chưa cấu hình chế độ Auto Confidence.")
    calibrated = item.get("calibrated") is True
    if calibrated:
        threshold = _valid_threshold(item.get("threshold"))
        if item.get("source") != "ROD_FINAL_VALID":
            raise RuntimeError("Thiếu bằng chứng VALID cho ngưỡng đã đánh dấu calibrated.")
        metrics = item.get("metrics") or {}
        if metrics.get("valid_images") != 3318:
            raise RuntimeError("Calibration chưa kiểm đủ 3318 ảnh VALID.")
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f2 = float(metrics["f2"])
        if not all(0 <= value <= 1 for value in (precision, recall, f2)):
            raise RuntimeError("Precision/Recall/F2 trong cấu hình không hợp lệ.")
    else:
        # A declared, low fallback keeps the demo running without claiming calibration.
        threshold = _valid_threshold(item.get("fallback_threshold"))
        precision = recall = f2 = None
    checksum = item.get("checkpoint_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise RuntimeError("Thiếu checksum checkpoint trong cấu hình Auto Confidence.")
    return AutoConfidence(threshold, calibrated, model_id, precision, recall, f2, checksum)
