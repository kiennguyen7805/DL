"""Local image-upload demo checks; no training or TEST split."""

from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

from app.confidence import auto_confidence
from app.calibrate_confidence import intersection_over_union
from app.detector import (
    MODEL_OPTIONS, Detection, _classwise_nms, checkpoint_path, classify_zone, detect,
    load_model, summary,
)
from app.main import build_app, run_detection


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGE = ROOT / "data/processed/rod_detection_final/train/images/IMG_00001.jpg"
TRAIN_MULTI_IMAGE = ROOT / "data/processed/rod_detection_final/train/images/IMG_00091.jpg"


def test_classwise_nms_removes_contained_duplicate_for_any_class():
    whole_object = Detection(7, "class-seven", 0.924, (89.0, 54.0, 532.0, 339.0))
    nested_duplicate = Detection(7, "class-seven", 0.553, (187.0, 53.0, 507.0, 143.0))
    separate_object = Detection(7, "class-seven", 0.700, (540.0, 54.0, 650.0, 160.0))
    different_class = Detection(8, "class-eight", 0.800, (187.0, 53.0, 507.0, 143.0))

    kept = _classwise_nms(
        [nested_duplicate, separate_object, different_class, whole_object],
        iou_threshold=0.50,
    )

    assert kept == (whole_object, different_class, separate_object)


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


def test_no_detection_and_manual_slider(tmp_path, monkeypatch):
    monkeypatch.setattr("app.detector.torch.cuda.is_available", lambda: False)
    blank = Image.new("RGB", (320, 320), "white")
    result = detect(blank, MODEL_OPTIONS[0], output_dir=tmp_path)
    assert not result.detections
    assert "Không phát hiện vật cản thuộc các lớp mô hình hỗ trợ." in summary(result, MODEL_OPTIONS[0])
    sliders = [component for component in build_app().blocks.values()
               if component.__class__.__name__ == "Slider"]
    assert len(sliders) == 1


def test_manual_threshold_is_applied(tmp_path, monkeypatch):
    monkeypatch.setattr("app.detector.torch.cuda.is_available", lambda: False)
    with Image.open(TRAIN_IMAGE) as image:
        result = detect(image, MODEL_OPTIONS[0], output_dir=tmp_path, threshold=0.12)
    assert result.threshold == 0.12
    assert result.threshold_source == "manual"
    assert "Tự chỉnh" in summary(result, MODEL_OPTIONS[0])
    assert "0.12" in summary(result, MODEL_OPTIONS[0])


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


def test_assistive_spatial_zones_and_filter_far(tmp_path, monkeypatch):
    monkeypatch.setattr("app.detector.torch.cuda.is_available", lambda: False)
    with Image.open(TRAIN_MULTI_IMAGE) as image:
        full_res = detect(image, MODEL_OPTIONS[0], output_dir=tmp_path, filter_far=False)
        filtered_res = detect(image, MODEL_OPTIONS[0], output_dir=tmp_path, filter_far=True)

    # Kiểm tra phân vùng xuất hiện trong summary
    text_full = summary(full_res, MODEL_OPTIONS[0])
    assert "Phân vùng theo vị trí trong ảnh" in text_full
    assert "DƯỚI CHÂN" in text_full
    assert "TRƯỚC MẶT" in text_full
    # Không được khẳng định khoảng cách hay an toàn chưa đo
    assert "bước" not in text_full and "An toàn" not in text_full

    # Số đối tượng sau khi lọc vùng xa phải <= số đối tượng ban đầu
    assert len(filtered_res.detections) <= len(full_res.detections)
    assert all(item.zone != "far" for item in filtered_res.detections)
    assert "đã ẩn các phát hiện thuộc vùng Ở XA" in summary(filtered_res, MODEL_OPTIONS[0])


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((40, 60, 60, 80), "ground"),   # đáy 0,80; tâm giữa
        ((0, 60, 10, 80), "front"),     # đáy 0,80; lệch trái sát rìa
        ((40, 30, 60, 50), "front"),    # đáy 0,50; tâm giữa
        ((0, 30, 10, 50), "far"),       # đáy 0,50; ngoài hành lang
        ((40, 5, 60, 30), "far"),       # đáy 0,30; phần trên ảnh
    ],
)
def test_classify_zone_boundaries(box, expected):
    assert classify_zone(box, 100, 100) == expected
