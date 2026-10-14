"""Synthetic-only tests; NEVER modify ROD dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src") if (Path(__file__).resolve().parents[1] / "src").exists()
                else str(Path(__file__).resolve().parent))
from rod_dataset_quality import load_classes, parse_box, scan_dataset
from rod_result_paths import resolve_result_reference


def test_legacy_day2_evidence_paths_resolve_without_editing_manifest(tmp_path: Path):
    evidence = tmp_path / "results/lam_sach_du_lieu/v3/visual_review/001.jpg"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"same evidence")
    for old in ("results/day2/v3/visual_review/001.jpg",
                r"results\day2\v3\visual_review\001.jpg"):
        assert resolve_result_reference(tmp_path, old) == evidence
    assert resolve_result_reference(tmp_path, "results/lam_sach_du_lieu/v3/visual_review/001.jpg") == evidence
    with pytest.raises(ValueError):
        resolve_result_reference(tmp_path, "../outside.jpg")


def make_dataset(tmp_path: Path, *, dup_between_splits=False):
    import yaml
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": {0: "Person", 1: "Car"}}), encoding="utf-8")
    for i, split in enumerate(("train", "valid", "test")):
        imgdir = tmp_path / split / "images"
        lbldir = tmp_path / split / "labels"
        imgdir.mkdir(parents=True)
        lbldir.mkdir(parents=True)
        image = Image.new("RGB", (40, 30), "white" if dup_between_splits else (i * 50, 40, 20))
        image.save(imgdir / f"pic_{split}.png")
        (lbldir / f"pic_{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return tmp_path


def test_load_yaml_dict_names(tmp_path):
    make_dataset(tmp_path)
    assert load_classes(tmp_path) == ["Person", "Car"]


def test_parse_valid_box():
    assert parse_box("1 0.5 0.5 0.2 0.4", 2) == (1, 0.5, 0.5, 0.2, 0.4)


@pytest.mark.parametrize("line", [
    "0 0.5 0.5 0.2", "2 0.5 0.5 0.2 0.2", "0 0.9 0.5 0.5 0.5",
    "0 0.5 0.5 nan 0.1", "0 0.5 0.5 0 0.1", "abc 0.5 0.5 0.1 0.1",
])
def test_parse_reject_invalid(line):
    with pytest.raises(ValueError):
        parse_box(line, 2)


def test_scan_counts_synthetic(tmp_path):
    make_dataset(tmp_path)
    report = scan_dataset(tmp_path, check_image_integrity=True, hash_images=True)
    assert sum(r["num_objects"] for r in report["records"]) == 3
    assert report["split_stats"]["train"]["images"] == 1
    assert not [r for r in report["issues"] if r["severity"] == "ERROR"]


def test_detect_missing_label(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "train" / "labels" / "pic_train.txt").unlink()
    report = scan_dataset(tmp_path)
    assert any(issue["issue_type"] == "missing_label" for issue in report["issues"])


def test_detect_cross_split_duplicate(tmp_path):
    make_dataset(tmp_path, dup_between_splits=True)
    report = scan_dataset(tmp_path, hash_images=True)
    assert report["overlap"]
    assert any(issue["issue_type"] == "cross_split_duplicate" for issue in report["issues"])


def test_detect_invalid_box(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "valid" / "labels" / "pic_valid.txt").write_text("1 1.1 0.5 0.2 0.2\n")
    report = scan_dataset(tmp_path)
    assert any(issue["issue_type"] == "invalid_label" for issue in report["issues"])


def test_render_and_review_pending(tmp_path):
    import importlib.util
    make_dataset(tmp_path / "dataset")
    module_path = Path(__file__).resolve().parents[1] / "src" / "02_verify_rod.py"
    if not module_path.is_file():
        module_path = Path(__file__).resolve().parent / "02_verify_rod.py"
    spec = importlib.util.spec_from_file_location("verify_rod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(tmp_path / "dataset", tmp_path / "results", n=3, seed=42,
                        candidates=tmp_path / "nonexistent.csv")
    assert report["status_auto"] == "PASS"
    assert report["human_visual_review"] == "PENDING"
    assert report["ready_for_training"] is False
    assert report["review_images_created"] == 3
    assert (tmp_path / "results" / "viz").exists()


def test_inspect_outputs(tmp_path):
    import importlib.util
    make_dataset(tmp_path / "dataset")
    module_path = Path(__file__).resolve().parents[1] / "src" / "01_inspect_rod.py"
    if not module_path.is_file():
        module_path = Path(__file__).resolve().parent / "01_inspect_rod.py"
    spec = importlib.util.spec_from_file_location("inspect_rod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(tmp_path / "dataset", tmp_path / "analysis")
    assert report["total_images"] == 3
    assert (tmp_path / "analysis" / "class_counts.csv").exists()
    assert (tmp_path / "analysis" / "fig_class_distribution.png").exists()


def test_audit_stem_candidate_prioritizes_clipped(tmp_path):
    import importlib.util
    root = make_dataset(tmp_path / "dataset")
    (root / "train" / "labels" / "pic_train.txt").write_text(
        "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n", encoding="utf-8")
    for stem in ("polygon", "clipped"):
        Image.new("RGB", (40, 30), "white").save(root / "train" / "images" / f"{stem}.png")
        (root / "train" / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("split,image_stem,reason\ntrain,polygon,polygon_converted\n"
                          "train,clipped,raw_box_or_polygon_clipped\n", encoding="utf-8")
    module_path = Path(__file__).resolve().parents[1] / "src" / "02_verify_rod.py"
    spec = importlib.util.spec_from_file_location("verify_rod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.prioritized_records(scan_dataset(root), 4, 42, candidates)
    assert len(selected) == 4
    assert selected[-1][0]["image"].stem == "clipped"
    assert selected[-1][1] == "audit_raw_box_or_polygon_clipped"
