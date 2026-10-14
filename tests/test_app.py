"""Local image-upload demo checks; no training or TEST split."""

from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

from app.confidence import auto_confidence
from app.calibrate_confidence import intersection_over_union
from app.detector import MODEL_OPTIONS, checkpoint_path, detect, load_model, summary
from app.main import build_app, run_detection


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGE = ROOT / "data/processed/rod_detection_final/train/images/IMG_00001.jpg"
TRAIN_MULTI_IMAGE = ROOT / "data/processed/rod_detection_final/train/images/IMG_00091.jpg"


@pytest.mark.parametrize("option", MODEL_OPTIONS[:2])
def test_finetuned_checkpoint_and_cpu_inference(option, tmp_path, monkeypatch):
    assert checkpoint_path(option).is_file()
    model, names = load_model(option)
    assert model.task == "detect"
    assert set(names) == set(range(25))
    monkeypatch.setattr("app.detector.torch.cuda.is_available", lambda: False)
    with Image.open(TRAIN_IMAGE) as image:
        result = detect(image, option, output_dir=tmp_path)
    assert result.device == "cpu"
    assert result.threshold == auto_confidence(option).threshold
    assert result.annotated.size == (640, 640)
    assert result.output_path.is_file()
    with Image.open(result.output_path) as saved:
        assert saved.format == "PNG" and saved.size == result.annotated.size
    for item in result.detections:
        x1, y1, x2, y2 = item.box
        assert item.name == names[item.class_id]
        assert result.threshold <= item.confidence <= 1
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640
        assert f"{item.confidence:.1%}" in summary(result, option)
    for index, first in enumerate(result.detections):
        for second in result.detections[index + 1:]:
            if first.class_id == second.class_id:
                assert intersection_over_union(list(first.box), list(second.box)) <= 0.7001
    assert "Thời gian suy luận" in summary(result, option)


def test_gradio_callback_and_input_errors():
    assert isinstance(build_app(), gr.Blocks)
    output, message, downloadable = run_detection(None, MODEL_OPTIONS[0])
    assert output is None and downloadable is None and "Hãy tải ảnh" in message
    with Image.open(TRAIN_IMAGE) as image:
        output, message, downloadable = run_detection(image, MODEL_OPTIONS[0])
    assert isinstance(output, Image.Image)
    assert "Số vật thể" in message
    assert "Thời gian suy luận" in message
    assert "Thời gian xử lý trên máy chủ" in message
    assert Path(downloadable).is_file()
    assert run_detection(output, "Không có model")[0] is None


def test_no_detection_and_no_slider(tmp_path, monkeypatch):
    monkeypatch.setattr("app.detector.torch.cuda.is_available", lambda: False)
    blank = Image.new("RGB", (320, 320), "white")
    result = detect(blank, MODEL_OPTIONS[0], output_dir=tmp_path)
    assert not result.detections
    assert "Không phát hiện vật cản thuộc các lớp mô hình hỗ trợ." in summary(result, MODEL_OPTIONS[0])
    assert all(component.__class__.__name__ != "Slider" for component in build_app().blocks.values())


@pytest.mark.parametrize("option", MODEL_OPTIONS[:2])
def test_multiple_objects_have_readable_confidence(option, tmp_path):
    with Image.open(TRAIN_MULTI_IMAGE) as image:
        result = detect(image, option, output_dir=tmp_path)
    assert len(result.detections) >= 2
    text = summary(result, option)
    assert text.count("- Vật thể ") == len(result.detections)
    assert all(f"{item.confidence:.1%}" in text for item in result.detections)
    for index, first in enumerate(result.detections):
        for second in result.detections[index + 1:]:
            if first.class_id == second.class_id:
                assert intersection_over_union(list(first.box), list(second.box)) <= 0.7001
