"""Model-specific adapters with one normalized ROD detection output."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data/processed/rod_detection_final/data.yaml"
E3_CHECKPOINT = ROOT / "results/E3_YOLO11n/weights/best.pt"
E4_CHECKPOINT = ROOT / "results/E4_YOLO11s/checkpoints/best.pt"
E6_CHECKPOINT = ROOT / "results/E6_Faster_RCNN/checkpoints/best_weights.zip"
CHECKPOINTS = {
    "YOLO11n (E3)": (E3_CHECKPOINT, "41f7943cbab67d1d5920e2852068b1e98d22eda8023f978abcb608a65a2aed02"),
    "YOLO11s (E4)": (E4_CHECKPOINT, "dc09c71052027e9083f014b99144edb65bcdc2ed032ab3591fce0db52b26e2f5"),
    "Faster R-CNN (E6)": (E6_CHECKPOINT, "ce459c56afc57efec80c63ab213bab60ecb9a966c78a1396051de6524783f9bd"),
}
MODEL_OPTIONS = tuple(CHECKPOINTS)


@dataclass(frozen=True)
class NormalizedDetections:
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    scores: tuple[float, ...]
    inference_time_ms: float
    device: str


class DetectorAdapter(Protocol):
    names: dict[int, str]
    parameter_count: int

    def predict(self, image: Image.Image, threshold: float) -> NormalizedDetections: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_checkpoint(model_option: str) -> Path:
    try:
        path, expected = CHECKPOINTS[model_option]
    except KeyError as exc:
        raise ValueError("Mô hình không hợp lệ.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint fine-tuned: {path}")
    if sha256_file(path) != expected:
        raise RuntimeError(f"SHA256 checkpoint không khớp: {path.name}")
    return path


@lru_cache(maxsize=1)
def rod_names() -> dict[int, str]:
    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    raw = config.get("names")
    if isinstance(raw, dict):
        raw = [raw[i] if i in raw else raw[str(i)] for i in range(25)]
    if config.get("nc") != 25 or not isinstance(raw, list) or len(raw) != 25:
        raise RuntimeError("ROD Final data.yaml không chứa đúng 25 class.")
    return {index: str(name) for index, name in enumerate(raw)}


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


class YoloAdapter:
    def __init__(self, model_option: str) -> None:
        self.path = require_checkpoint(model_option)
        self.model = YOLO(str(self.path))
        if self.model.task != "detect":
            raise RuntimeError("Checkpoint YOLO không phải object detection.")
        raw = self.model.names
        names = {int(i): str(name) for i, name in
                 (raw.items() if isinstance(raw, dict) else enumerate(raw))}
        if names != rod_names():
            raise RuntimeError("Class checkpoint YOLO khác ROD Final data.yaml.")
        self.names = names
        self.parameter_count = sum(p.numel() for p in self.model.model.parameters())

    def predict(self, image: Image.Image, threshold: float) -> NormalizedDetections:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sync(device)
        started = time.perf_counter()
        result = self.model.predict(
            source=np.asarray(image), conf=threshold, imgsz=640, iou=0.7,
            max_det=300, device="0" if device == "cuda" else "cpu",
            verbose=False, save=False,
        )[0]
        _sync(device)
        elapsed = (time.perf_counter() - started) * 1000
        boxes = result.boxes
        if boxes is None:
            return NormalizedDetections((), (), (), (), elapsed, device)
        coords = tuple(tuple(map(float, row)) for row in boxes.xyxy.cpu().tolist())
        class_ids = tuple(int(value) for value in boxes.cls.cpu().tolist())
        scores = tuple(float(value) for value in boxes.conf.cpu().tolist())
        if any(index not in self.names for index in class_ids):
            raise RuntimeError("YOLO trả về class ngoài ROD 0–24.")
        return NormalizedDetections(
            coords, class_ids, tuple(self.names[i] for i in class_ids), scores,
            elapsed, device,
        )


class FasterRCNNAdapter:
    """Torchvision E6 adapter; model labels 1..25 map to ROD labels 0..24."""

    def __init__(self) -> None:
        self.path = require_checkpoint("Faster R-CNN (E6)")
        self.names = rod_names()
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, min_size=640, max_size=960,
            box_score_thresh=0.0, box_nms_thresh=0.7, box_detections_per_img=300,
        )
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, 26)
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        if tuple(state["roi_heads.box_predictor.cls_score.weight"].shape) != (26, features):
            raise RuntimeError("Checkpoint E6 không có classification head 26 lớp.")
        if tuple(state["roi_heads.box_predictor.bbox_pred.weight"].shape) != (104, features):
            raise RuntimeError("Checkpoint E6 không có bbox head 26 lớp.")
        model.load_state_dict(state, strict=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        self.parameter_count = sum(p.numel() for p in model.parameters())

    @torch.inference_mode()
    def predict(self, image: Image.Image, threshold: float) -> NormalizedDetections:
        device_name = self.device.type
        _sync(device_name)
        started = time.perf_counter()
        tensor = pil_to_tensor(image).float().div_(255).to(self.device)
        raw = self.model([tensor])[0]
        _sync(device_name)
        elapsed = (time.perf_counter() - started) * 1000
        boxes: list[tuple[float, float, float, float]] = []
        class_ids: list[int] = []
        scores: list[float] = []
        for box, label, score in zip(
            raw["boxes"].detach().cpu().tolist(),
            raw["labels"].detach().cpu().tolist(),
            raw["scores"].detach().cpu().tolist(), strict=True,
        ):
            # Torchvision reserves 0 for background. It must never be exposed.
            if int(label) == 0 or float(score) < threshold:
                continue
            rod_id = int(label) - 1
            if rod_id not in self.names:
                raise RuntimeError(f"E6 trả về class Torchvision ngoài 1–25: {label}")
            boxes.append(tuple(map(float, box)))
            class_ids.append(rod_id)
            scores.append(float(score))
        ids = tuple(class_ids)
        return NormalizedDetections(
            tuple(boxes), ids, tuple(self.names[i] for i in ids), tuple(scores),
            elapsed, device_name,
        )


@lru_cache(maxsize=3)
def load_adapter(model_option: str) -> DetectorAdapter:
    if model_option in MODEL_OPTIONS[:2]:
        return YoloAdapter(model_option)
    if model_option == MODEL_OPTIONS[2]:
        return FasterRCNNAdapter()
    raise ValueError("Mô hình không hợp lệ.")
