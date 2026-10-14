"""E6 app integration and common metric protocol; TRAIN/synthetic data only."""

from pathlib import Path

import gradio as gr
import pytest
import torch
from PIL import Image

from app.confidence import auto_confidence
from app.detector import MODEL_OPTIONS, Prediction, detect, summary
from app.evaluate_models import PROTOCOL, verify_lock
from app.evaluate_test import run_final_test
from app.evaluator import evaluate_records
from app.main import build_app, run_detection
from app.model_adapters import FasterRCNNAdapter, load_adapter, rod_names


ROOT = Path(__file__).resolve().parents[1]
E6 = "Faster R-CNN (E6)"
TRAIN_IMAGE = ROOT / "data/processed/rod_detection_final/train/images/IMG_00091.jpg"


def test_e6_checkpoint_head_mapping_and_cpu_inference(tmp_path, monkeypatch):
    monkeypatch.setattr("app.model_adapters.torch.cuda.is_available", lambda: False)
    adapter = load_adapter(E6)
    assert isinstance(adapter, FasterRCNNAdapter)
    assert load_adapter(E6) is adapter
    assert adapter.names == rod_names()
    assert adapter.model.roi_heads.box_predictor.cls_score.out_features == 26
    assert adapter.model.roi_heads.box_predictor.bbox_pred.out_features == 104
    with Image.open(TRAIN_IMAGE) as image:
        low_threshold = adapter.predict(image.convert("RGB"), 0.01)
        result = detect(image, E6, output_dir=tmp_path)
    assert low_threshold.class_ids
    assert result.device == "cpu" and result.output_path.is_file()
    assert result.threshold == auto_confidence(E6).threshold
    assert result.calibrated == auto_confidence(E6).calibrated
    assert all(0 <= item.class_id < 25 and item.name == adapter.names[item.class_id]
               and result.threshold <= item.confidence <= 1 for item in result.detections)
    if not result.calibrated:
        assert "chưa hiệu chuẩn" in summary(result, E6)
    with Image.open(result.output_path) as saved:
        assert saved.format == "PNG" and saved.size == result.annotated.size
    assert isinstance(build_app(), gr.Blocks) and E6 in MODEL_OPTIONS
    with Image.open(TRAIN_IMAGE) as image:
        annotated, message, downloadable = run_detection(image, E6)
    assert isinstance(annotated, Image.Image) and Path(downloadable).is_file()
    assert "Số vật thể" in message


def test_switch_e3_e6_e4_on_same_image(tmp_path, monkeypatch):
    monkeypatch.setattr("app.model_adapters.torch.cuda.is_available", lambda: False)
    with Image.open(TRAIN_IMAGE) as source:
        image = source.convert("RGB")
    for option in MODEL_OPTIONS:
        result = detect(image, option, output_dir=tmp_path)
        assert result.output_path.is_file()
        assert all(item.name and item.class_id in rod_names() for item in result.detections)


def test_common_ap_and_operating_counts():
    rows = [{"image": "a.jpg", "width": 100, "height": 100,
             "gt": [[0, 0, 0, 10, 10]],
             "pred": [[0, 0.9, 0, 0, 10, 10], [0, 0.8, 0, 0, 10, 10],
                      [1, 0.7, 20, 20, 30, 30]]}]
    result = evaluate_records(rows, 0.5)
    assert result["map50"] == pytest.approx(1.0)
    assert result["map50_95"] == pytest.approx(1.0)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["per_class"][1]["ap50"] is None
    assert PROTOCOL["evaluation_max_det"] == 100


def test_float32_rounding_just_outside_image_is_accepted():
    row = {"image": "edge.jpg", "width": 999, "height": 673,
           "gt": [[0, 900, 0, 999, 430]],
           "pred": [[0, 0.9, 900, 0, 999.000061035156, 430]]}
    result = evaluate_records([row], 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)


def test_e6_background_never_reaches_rod_output():
    adapter = FasterRCNNAdapter.__new__(FasterRCNNAdapter)
    adapter.device = torch.device("cpu")
    adapter.names = rod_names()
    adapter.model = lambda images: [{
        "boxes": torch.tensor([[0., 0., 5., 5.], [1., 1., 6., 6.], [2., 2., 7., 7.]]),
        "labels": torch.tensor([0, 1, 25]),
        "scores": torch.tensor([0.99, 0.8, 0.7]),
    }]
    output = adapter.predict(Image.new("RGB", (16, 16)), 0.05)
    assert output.class_ids == (0, 24)
    assert output.class_names == (adapter.names[0], adapter.names[24])


def test_empty_detection_and_wrong_class():
    row = {"image": "blank.jpg", "width": 50, "height": 50,
           "gt": [[0, 1, 1, 20, 20]], "pred": []}
    result = evaluate_records([row], 0.05)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
    row["pred"] = [[25, 0.9, 1, 1, 20, 20]]
    with pytest.raises(ValueError, match="Class ID"):
        evaluate_records([row], 0.05)


def test_e6_no_detection_message_is_not_a_safety_claim(tmp_path):
    empty = Prediction(Image.new("RGB", (20, 20)), (), 0.1,
                       tmp_path / "empty.png", "cpu", 0.05, False)
    message = summary(empty, E6)
    assert "Không phát hiện vật cản thuộc các lớp mô hình hỗ trợ" in message
    assert "an toàn" not in message


def test_test_run_blocked_before_opening_test_split(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="explicit"):
        run_final_test("E3", confirmed=False)
    monkeypatch.setattr("app.evaluate_models.LOCK_PATH", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="Missing frozen"):
        verify_lock()
