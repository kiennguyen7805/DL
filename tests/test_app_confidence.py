"""Auto Confidence selection and accessibility checks."""

import json

import gradio as gr
import pytest

from app.calibrate_confidence import ROOT, choose_threshold, score_predictions, sha256
from app.confidence import auto_confidence
from app.detector import MODEL_OPTIONS
from app.main import build_app


def test_one_to_one_matching_and_f2_selection():
    records = [{
        "gt": [[0, 0, 0, 10, 10], [1, 20, 20, 30, 30]],
        "pred": [
            [0, 0.90, 0, 0, 10, 10],
            [2, 0.80, 0, 0, 10, 10],  # wrong class is FP
            [1, 0.20, 20, 20, 30, 30],
            [0, 0.10, 0, 0, 10, 10],  # duplicate cannot match GT twice
        ],
    }]
    scores = score_predictions(records, [0.10, 0.20, 0.80, 0.90])
    chosen = choose_threshold(scores)
    assert chosen.threshold == 0.20
    assert (chosen.tp, chosen.fp, chosen.fn) == (2, 1, 0)
    assert chosen.precision == pytest.approx(2 / 3)
    assert chosen.recall == 1.0


@pytest.mark.parametrize("option", MODEL_OPTIONS[:2])
def test_auto_threshold_is_model_specific_and_never_null(option):
    selected = auto_confidence(option)
    assert 0.001 <= selected.threshold <= 0.99
    assert selected.model_id in {"E3", "E4"}


def test_uncalibrated_fallback_is_explicit(tmp_path):
    path = tmp_path / "confidence.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "E3": {"mode": "auto", "threshold": None, "fallback_threshold": 0.05,
               "calibrated": False, "checkpoint_sha256": "a" * 64},
    }), encoding="utf-8")
    selected = auto_confidence(MODEL_OPTIONS[0], config_path=path)
    assert selected.threshold == 0.05
    assert not selected.calibrated
    assert selected.precision is None


def test_no_slider_and_only_two_user_inputs():
    demo = build_app()
    assert isinstance(demo, gr.Blocks)
    assert all(component.__class__.__name__ != "Slider" for component in demo.blocks.values())
    button = next(component for component in demo.blocks.values() if component.__class__.__name__ == "Button")
    assert button.value == "Phát hiện vật cản"


@pytest.mark.parametrize("model_id", ("E3", "E4"))
def test_full_valid_evidence_reproduces_selected_threshold(model_id):
    folder = ROOT / "results" / ("E3_YOLO11n" if model_id == "E3" else "E4_YOLO11s") / "confidence"
    metadata = json.loads((folder / f"{model_id}_valid_metadata.json").read_text(encoding="utf-8"))
    assert metadata["split"] == "valid" and metadata["image_count"] == 3318
    path = folder / f"{model_id}_valid_predictions.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len({row["image"] for row in records}) == 3318
    assert all(row["image"].startswith("IMG_") for row in records)
    selected = choose_threshold(score_predictions(records))
    config = json.loads((ROOT / "app/confidence_config.json").read_text(encoding="utf-8"))[model_id]
    assert config["calibrated"] and config["source"] == "ROD_FINAL_VALID"
    assert config["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert config["prediction_sha256"] == sha256(path)
    assert config["threshold"] == selected.threshold
    assert config["metrics"]["tp"] == selected.tp
    assert config["metrics"]["fp"] == selected.fp
    assert config["metrics"]["fn"] == selected.fn
