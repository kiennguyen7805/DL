"""Inference and output handling for the ROD image-upload demo."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from app.confidence import auto_confidence
from app.model_adapters import CHECKPOINTS, load_adapter


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
E3_CHECKPOINT = ROOT / "results/E3_YOLO11n/weights/best.pt"
E3_SHA256 = "41f7943cbab67d1d5920e2852068b1e98d22eda8023f978abcb608a65a2aed02"
E4_CHECKPOINT = ROOT / "results/E4_YOLO11s/checkpoints/best.pt"
E4_SHA256 = "dc09c71052027e9083f014b99144edb65bcdc2ed032ab3591fce0db52b26e2f5"
MODEL_OPTIONS = tuple(CHECKPOINTS)
_MODEL_LOCK = threading.RLock()


@dataclass(frozen=True)
class Detection:
    class_id: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    annotated: Image.Image
    detections: tuple[Detection, ...]
    inference_seconds: float
    output_path: Path
    device: str
    threshold: float
    calibrated: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint fine-tuned: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 checkpoint không khớp: {path.name}")


def checkpoint_path(model_option: str) -> Path:
    if model_option == MODEL_OPTIONS[0]:
        _require_sha(E3_CHECKPOINT, E3_SHA256)
        return E3_CHECKPOINT
    if model_option == MODEL_OPTIONS[1]:
        _require_sha(E4_CHECKPOINT, E4_SHA256)
        return E4_CHECKPOINT
    if model_option == MODEL_OPTIONS[2]:
        path, expected = CHECKPOINTS[model_option]
        _require_sha(path, expected)
        return path
    raise ValueError("Mô hình không hợp lệ.")


def _checkpoint_names(model: YOLO) -> dict[int, str]:
    raw = model.names
    names = {int(i): str(name) for i, name in
             (raw.items() if isinstance(raw, dict) else enumerate(raw))}
    if set(names) != set(range(25)):
        raise RuntimeError("Checkpoint không chứa đúng 25 class ID ROD (0–24).")
    config = yaml.safe_load((ROOT / "data/processed/rod_detection_final/data.yaml").read_text(encoding="utf-8"))
    expected = config["names"]
    if isinstance(expected, dict):
        expected = [expected[i] if i in expected else expected[str(i)] for i in range(25)]
    if config["nc"] != 25 or [names[i] for i in range(25)] != list(expected):
        raise RuntimeError("Tên hoặc thứ tự class trong checkpoint khác ROD Final data.yaml.")
    return names


@lru_cache(maxsize=2)
def load_model(model_option: str) -> tuple[YOLO, dict[int, str]]:
    """Load one verified fine-tuned checkpoint per model for this process."""
    path = checkpoint_path(model_option)
    model = YOLO(str(path))
    if model.task != "detect":
        raise RuntimeError("Checkpoint không phải mô hình object detection.")
    return model, _checkpoint_names(model)


def _validated_image(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("Hãy tải ảnh JPG, JPEG hoặc PNG trước khi phát hiện.")
    if not isinstance(image, Image.Image):
        raise ValueError("Không đọc được ảnh tải lên.")
    if image.format and image.format.upper() not in {"JPEG", "PNG"}:
        raise ValueError("Chỉ hỗ trợ ảnh JPG, JPEG hoặc PNG.")
    try:
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError("Ảnh bị lỗi hoặc không thể giải mã.") from exc


def _color(class_id: int) -> tuple[int, int, int]:
    palette = ((23, 132, 255), (0, 168, 130), (238, 106, 44),
               (148, 82, 224), (231, 68, 102), (24, 149, 166))
    return palette[class_id % len(palette)]


def _annotate(image: Image.Image, detections: tuple[Detection, ...]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    thickness = max(2, round(min(image.size) / 250))
    for item in detections:
        x1, y1, x2, y2 = item.box
        color = _color(item.class_id)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
        label = f"{item.name} {item.confidence:.1%}"
        left, top, right, bottom = draw.textbbox((0, 0), label)
        width, height = right - left + 10, bottom - top + 8
        label_y = max(0, y1 - height)
        draw.rectangle((x1, label_y, min(image.width, x1 + width), label_y + height), fill=color)
        draw.text((x1 + 5, label_y + 3), label, fill="white")
    return result


def detect(
    image: Image.Image,
    model_option: str,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> Prediction:
    setting = auto_confidence(model_option)
    if model_option not in CHECKPOINTS:
        raise ValueError("Mô hình không hợp lệ.")
    expected_checksum = CHECKPOINTS[model_option][1]
    if setting.checkpoint_sha256 != expected_checksum:
        raise RuntimeError("Ngưỡng Auto Confidence không khớp checkpoint đã xác minh.")
    rgb = _validated_image(image)
    with _MODEL_LOCK:
        detections: list[Detection] = []
        if model_option == MODEL_OPTIONS[2]:
            adapter = load_adapter(model_option)
            normalized = adapter.predict(rgb, setting.threshold)
            names = adapter.names
            rows = zip(normalized.boxes_xyxy, normalized.class_ids, normalized.scores, strict=True)
            device = "0" if normalized.device == "cuda" else "cpu"
            elapsed = normalized.inference_time_ms / 1000
        else:
            model, names = load_model(model_option)
            device = "0" if torch.cuda.is_available() else "cpu"
            if device == "0":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.predict(
                source=np.asarray(rgb), conf=setting.threshold, imgsz=640,
                iou=0.7, max_det=300, device=device, verbose=False, save=False,
            )[0]
            if device == "0":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            boxes = result.boxes
            rows = () if boxes is None else zip(
                boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(),
                boxes.conf.cpu().tolist(), strict=True,
            )
        for coords, class_id, score in rows:
            index = int(class_id)
            if index not in names or not 0 <= score <= 1:
                raise RuntimeError("Kết quả dự đoán có class/confidence không hợp lệ.")
            x1, y1, x2, y2 = coords
            x1, x2 = max(0.0, min(x1, rgb.width)), max(0.0, min(x2, rgb.width))
            y1, y2 = max(0.0, min(y1, rgb.height)), max(0.0, min(y2, rgb.height))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError("Kết quả dự đoán có bounding box không hợp lệ.")
            detections.append(Detection(index, names[index], float(score), (x1, y1, x2, y2)))
    items = tuple(detections)
    annotated = _annotate(rgb, items)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rod_{uuid.uuid4().hex}.png"
    annotated.save(path, format="PNG")
    return Prediction(annotated, items, elapsed, path, device, setting.threshold, setting.calibrated)


def summary(prediction: Prediction, model_option: str) -> str:
    count = Counter(item.name for item in prediction.detections)
    lines = [
        f"### Kết quả — {model_option}",
        ("Chế độ nhận diện: **Tự động**" if prediction.calibrated
         else "Chế độ nhận diện: **Tự động (chưa hiệu chuẩn trên VALID)**"),
        f"**Số vật thể:** {len(prediction.detections)}  ·  "
        f"**Thời gian suy luận:** {prediction.inference_seconds * 1000:.1f} ms  ·  "
        f"**Thiết bị:** {'GPU' if prediction.device == '0' else 'CPU'}",
    ]
    if not count:
        lines.append("Không phát hiện vật cản thuộc các lớp mô hình hỗ trợ.")
    else:
        lines += ["", "| Vật thể | Số lượng |", "|---|---:|"]
        lines += [f"| {name} | {number} |" for name, number in sorted(count.items())]
        lines += ["", "**Danh sách phát hiện:**"]
        lines += [
            f"- Vật thể {index}: {item.name}, độ tin cậy {item.confidence:.1%}."
            for index, item in enumerate(prediction.detections, start=1)
        ]
    return "\n".join(lines)
