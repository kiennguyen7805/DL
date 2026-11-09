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

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from app.confidence import auto_confidence
from app.model_adapters import CHECKPOINTS, load_adapter


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
E3_CHECKPOINT = ROOT / "results/30epochs/E3/checkpoint/best.pt"
E3_SHA256 = "d88b24f020a75464d2df2197b8fefeb16be2ebb735f361d19ba569b0dfc33786"
E4_CHECKPOINT = ROOT / "results/30epochs/E4/checkpoint/best.pt"
E4_SHA256 = "a5c32e2fc8be4eb59dd80b67d76ac8abfc4a5ab13437814962ea2c463dffe1ce"
MODEL_OPTIONS = tuple(CHECKPOINTS)
_MODEL_LOCK = threading.RLock()


# --- Phân vùng vị trí (prototype hậu xử lý, KHÔNG thuộc kết quả đánh giá) ---
# Các ngưỡng dưới đây là heuristic đặt tay, chưa được hiệu chỉnh hay kiểm chứng
# trên dữ liệu nào. Chúng giả định ảnh chụp từ góc nhìn người đi bộ (camera
# ngang ngực, đường chân trời gần giữa ảnh); ROD gồm ảnh nhiều nguồn nên giả
# định này không luôn đúng. Phân vùng không ước lượng khoảng cách và không thay
# đổi bbox/confidence của mô hình; mAP/Precision/Recall/F2 không phụ thuộc vào nó.
ZONE_GROUND_MIN_Y = 0.70      # đáy bbox thấp hơn 70% chiều cao ảnh
ZONE_FRONT_MIN_Y = 0.40       # đáy bbox thấp hơn 40% chiều cao ảnh
ZONE_GROUND_X = (0.12, 0.88)  # khoảng tâm ngang của vùng dưới chân
ZONE_FRONT_X = (0.18, 0.82)   # khoảng tâm ngang của vùng trước mặt

ZONE_GROUND = "ground"
ZONE_FRONT = "front"
ZONE_FAR = "far"

ZONE_METADATA = {
    ZONE_GROUND: {
        "label": "DƯỚI CHÂN",
        "icon": "🔴",
        "color": (230, 40, 40),      # Đỏ: nửa dưới, giữa ảnh
        "priority": 1,
    },
    ZONE_FRONT: {
        "label": "TRƯỚC MẶT",
        "icon": "🟡",
        "color": (245, 155, 20),     # Vàng cam: giữa ảnh hoặc sát đáy hai bên
        "priority": 2,
    },
    ZONE_FAR: {
        "label": "Ở XA",
        "icon": "⚪",
        "color": (145, 155, 165),    # Xám: phần trên ảnh hoặc rìa ngoài
        "priority": 3,
    },
}


def classify_zone(
    box: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> str:
    """Gán vùng theo vị trí bbox trong ảnh (heuristic, chưa kiểm chứng).

    - Dưới chân: đáy bbox >= ZONE_GROUND_MIN_Y và tâm ngang trong ZONE_GROUND_X.
    - Trước mặt: đáy bbox trong [ZONE_FRONT_MIN_Y, ZONE_GROUND_MIN_Y) và tâm
      ngang trong ZONE_FRONT_X; hoặc đáy bbox >= ZONE_GROUND_MIN_Y nhưng lệch
      sang hai bên.
    - Ở xa: các trường hợp còn lại.

    Đây là quy tắc vị trí trên ảnh 2D, không phải khoảng cách thật và không xác
    định vật thể có thực sự cản lối đi hay không.
    """
    x1, y1, x2, y2 = box
    norm_y_bottom = y2 / max(1, img_height)
    norm_x_center = ((x1 + x2) / 2.0) / max(1, img_width)

    if norm_y_bottom >= ZONE_GROUND_MIN_Y:
        if ZONE_GROUND_X[0] <= norm_x_center <= ZONE_GROUND_X[1]:
            return ZONE_GROUND
        return ZONE_FRONT
    if norm_y_bottom >= ZONE_FRONT_MIN_Y and ZONE_FRONT_X[0] <= norm_x_center <= ZONE_FRONT_X[1]:
        return ZONE_FRONT
    return ZONE_FAR


@dataclass(frozen=True)
class Detection:
    class_id: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]
    zone: str = ZONE_FAR


@dataclass(frozen=True)
class Prediction:
    annotated: Image.Image
    detections: tuple[Detection, ...]
    inference_seconds: float
    output_path: Path
    device: str
    threshold: float
    calibrated: bool
    threshold_source: str = "auto"
    scan_mode: str = "standard"
    filter_far: bool = False


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


@lru_cache(maxsize=16)
def _get_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a TrueType font supporting Vietnamese Unicode with cross-platform fallbacks."""
    candidates = (
        "arial.ttf",
        "segoeui.ttf",
        "tahoma.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, RuntimeError):
            continue
    return ImageFont.load_default()


def _annotate(
    image: Image.Image,
    detections: tuple[Detection, ...],
    assistive_mode: bool = True,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    thickness = max(2, round(min(image.size) / 250))
    font_size = max(13, round(min(image.size) / 45))
    font = _get_font(font_size)

    # Nếu hệ thống rơi vào font bitmap mặc định (không hỗ trợ Unicode tiếng Việt),
    # chuyển nhãn sang không dấu để tránh bị lỗi ô vuông (tofu).
    is_default_bitmap = not hasattr(font, "font") or font == ImageFont.load_default()

    for item in detections:
        x1, y1, x2, y2 = item.box
        if assistive_mode and item.zone in ZONE_METADATA:
            meta = ZONE_METADATA[item.zone]
            color = meta["color"]
            box_thickness = thickness + 2 if item.zone == ZONE_GROUND else max(2, thickness)
            label_text = meta["label"]
            if is_default_bitmap:
                label_text = {"DƯỚI CHÂN": "DUOI CHAN", "TRƯỚC MẶT": "TRUOC MAT", "Ở XA": "O XA"}.get(label_text, label_text)
            prefix = f"[{label_text}] "
        else:
            color = _color(item.class_id)
            box_thickness = thickness
            prefix = ""
        draw.rectangle((x1, y1, x2, y2), outline=color, width=box_thickness)
        label = f"{prefix}{item.name} {item.confidence:.1%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width, height = right - left + 10, bottom - top + 6
        label_y = max(0, y1 - height)
        draw.rectangle((x1, label_y, min(image.width, x1 + width), label_y + height), fill=color)
        draw.text((x1 + 5, label_y + 2), label, font=font, fill="white")
    return result


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _overlap_of_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return the fraction of the smaller box covered by the other box."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / min(first_area, second_area)


def _classwise_nms(
    detections: list[Detection], iou_threshold: float,
    containment_threshold: float = 0.85,
) -> tuple[Detection, ...]:
    """Keep one same-class detection when overlapping views see the same object.

    This runs for every image and model. A zoomed crop can return only the
    top/centre of an object. Its IoU with the full-image box is then low even
    though it is wholly inside that box, so use containment in addition to
    conventional IoU.
    """
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(
            candidate.class_id != existing.class_id
            or (
                _iou(candidate.box, existing.box) <= iou_threshold
                and _overlap_of_smaller(candidate.box, existing.box) < containment_threshold
            )
            for existing in kept
        ):
            kept.append(candidate)
    return tuple(kept)


def _inference_views(image: Image.Image, high_recall: bool) -> list[tuple[Image.Image, int, int]]:
    """Return full image plus overlapping zoomed regions for small/far objects."""
    views: list[tuple[Image.Image, int, int]] = [(image, 0, 0)]
    if not high_recall:
        return views
    width, height = image.size
    regions = [
        (0, round(height * 0.109), width, round(height * 0.344)),  # far perspective band
        (0, 0, width, round(height * 0.344)),
        (0, round(height * 0.18), width, round(height * 0.68)),
        (0, round(height * 0.52), width, height),
        (0, 0, round(width * 0.60), height),
        (round(width * 0.40), 0, width, height),
    ]
    for left, top, right, bottom in regions:
        if right - left >= 32 and bottom - top >= 32:
            views.append((image.crop((left, top, right, bottom)), left, top))
    return views


def detect(
    image: Image.Image,
    model_option: str,
    *,
    output_dir: Path = OUTPUT_DIR,
    threshold: float | None = None,
    high_recall: bool = False,
    assistive_mode: bool = True,
    filter_far: bool = False,
) -> Prediction:
    setting = auto_confidence(model_option)
    if model_option not in CHECKPOINTS:
        raise ValueError("Mô hình không hợp lệ.")
    expected_checksum = CHECKPOINTS[model_option][1]
    if setting.checkpoint_sha256 != expected_checksum:
        raise RuntimeError("Ngưỡng Auto Confidence không khớp checkpoint đã xác minh.")
    if threshold is None:
        selected_threshold = setting.threshold
        threshold_source = "auto"
    else:
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
            raise ValueError("Ngưỡng confidence tự chỉnh không hợp lệ.")
        selected_threshold = float(threshold)
        if not 0.01 <= selected_threshold <= 0.99:
            raise ValueError("Ngưỡng confidence phải nằm trong khoảng 0,01–0,99.")
        threshold_source = "manual"
    rgb = _validated_image(image)
    views = _inference_views(rgb, high_recall)
    if model_option == MODEL_OPTIONS[2] and high_recall:
        # Faster R-CNN is already strong on this type of small object and is
        # much slower on CPU, so only add the far-perspective zoom.
        views = views[:2]
    with _MODEL_LOCK:
        detections: list[Detection] = []
        if model_option == MODEL_OPTIONS[2]:
            adapter = load_adapter(model_option)
            names = adapter.names
            elapsed = 0.0
            device = "cpu"
            rows = []
            for crop, offset_x, offset_y in views:
                normalized = adapter.predict(crop, selected_threshold)
                elapsed += normalized.inference_time_ms / 1000
                device = "0" if normalized.device == "cuda" else "cpu"
                rows.extend(
                    ((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), class_id, score)
                    for (x1, y1, x2, y2), class_id, score in zip(
                        normalized.boxes_xyxy, normalized.class_ids, normalized.scores, strict=True
                    )
                )
        else:
            model, names = load_model(model_option)
            device = "0" if torch.cuda.is_available() else "cpu"
            if device == "0":
                torch.cuda.synchronize()
            started = time.perf_counter()
            results = model.predict(
                # Pass PIL directly so Ultralytics performs the documented
                # RGB-to-BGR conversion. A raw NumPy RGB array is interpreted
                # as BGR and can change predictions on uploaded images.
                source=[crop for crop, _, _ in views], conf=selected_threshold, imgsz=640,
                iou=0.7, max_det=300, device=device, verbose=False, save=False,
            )
            if device == "0":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows = []
            for result, (_, offset_x, offset_y) in zip(results, views, strict=True):
                boxes = result.boxes
                if boxes is None:
                    continue
                rows.extend(
                    ((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), class_id, score)
                    for (x1, y1, x2, y2), class_id, score in zip(
                        boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(),
                        boxes.conf.cpu().tolist(), strict=True,
                    )
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
            zone = classify_zone((x1, y1, x2, y2), rgb.width, rgb.height)
            detections.append(Detection(index, names[index], float(score), (x1, y1, x2, y2), zone))
    # Every model and every image goes through the same class-wise cleanup.
    # The base models already apply their own NMS, so standard inference keeps
    # that permissive IoU. Multi-crop inference needs a lower threshold because
    # it combines several coordinate views of the same image.
    duplicate_iou = 0.50 if high_recall else 0.70
    items = _classwise_nms(detections, iou_threshold=duplicate_iou)
    if filter_far:
        items = tuple(item for item in items if item.zone != ZONE_FAR)
    annotated = _annotate(rgb, items, assistive_mode=assistive_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rod_{uuid.uuid4().hex}.png"
    annotated.save(path, format="PNG")
    return Prediction(
        annotated, items, elapsed, path, device, selected_threshold,
        setting.calibrated, threshold_source,
        "high_recall" if high_recall else "standard",
        filter_far=filter_far,
    )


def summary(prediction: Prediction, model_option: str) -> str:
    count = Counter(item.name for item in prediction.detections)
    ground_items = [d for d in prediction.detections if d.zone == ZONE_GROUND]
    front_items = [d for d in prediction.detections if d.zone == ZONE_FRONT]
    far_items = [d for d in prediction.detections if d.zone == ZONE_FAR]

    lines = [
        f"### Kết quả — {model_option}",
        ("Chế độ nhận diện: **Tự chỉnh**" if prediction.threshold_source == "manual"
         else ("Chế độ nhận diện: **Tự động theo VALID**" if prediction.calibrated
               else "Chế độ nhận diện: **Tự động (chưa hiệu chuẩn trên VALID)**")),
        f"**Ngưỡng confidence:** {prediction.threshold:.2f}",
        ("**Kiểu quét:** High recall đa vùng" if prediction.scan_mode == "high_recall"
         else "**Kiểu quét:** Chuẩn"),
        f"**Số vật thể:** {len(prediction.detections)}  ·  "
        f"**Thời gian suy luận:** {prediction.inference_seconds * 1000:.1f} ms  ·  "
        f"**Thiết bị:** {'GPU' if prediction.device == '0' else 'CPU'}",
    ]
    if prediction.filter_far:
        lines.append("**Lọc theo vùng:** Đang bật — đã ẩn các phát hiện thuộc vùng Ở XA.")

    if not count:
        lines.append("Không phát hiện vật cản thuộc các lớp mô hình hỗ trợ.")
        return "\n".join(lines)

    def _zone_line(icon: str, label: str, items: list[Detection]) -> str:
        if not items:
            return f"- {icon} **{label}:** không có phát hiện nào trong vùng này."
        names = ", ".join(f"{name} ({c})" for name, c in Counter(d.name for d in items).items())
        return f"- {icon} **{label}:** {len(items)} vật thể [{names}]"

    lines += [
        "",
        "#### Phân vùng theo vị trí trong ảnh (prototype)",
        _zone_line("🔴", "DƯỚI CHÂN", ground_items),
        _zone_line("🟡", "TRƯỚC MẶT", front_items),
    ]
    if not prediction.filter_far:
        lines.append(_zone_line("⚪", "Ở XA", far_items))
    lines.append(
        "\n*Vùng được gán theo vị trí bbox trên ảnh bằng quy tắc heuristic chưa kiểm chứng; "
        "không phải khoảng cách thật. Vùng không có phát hiện không có nghĩa là không có vật cản.*"
    )

    lines += ["", "| Vật thể | Số lượng |", "|---|---:|"]
    lines += [f"| {name} | {number} |" for name, number in sorted(count.items())]
    lines += ["", "**Danh sách phát hiện:**"]
    for index, item in enumerate(prediction.detections, start=1):
        meta = ZONE_METADATA.get(item.zone, {"icon": "•", "label": item.zone.upper()})
        lines.append(f"- Vật thể {index}: {item.name}, độ tin cậy {item.confidence:.1%} — {meta['icon']} {meta['label']}.")

    return "\n".join(lines)
