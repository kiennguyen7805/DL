from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "build_rod_detection_final.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_rod_detection_final", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


@pytest.mark.parametrize(
    ("input_path", "expected"),
    [
        ("train/images/IMG_00001.jpg", "train/images/IMG_00001.jpg"),
        (r"train\images\IMG_00001.jpg", "train/images/IMG_00001.jpg"),
        (r"valid\images/IMG_00002.jpg", "valid/images/IMG_00002.jpg"),
    ],
)
def test_normalize_rel_accepts_windows_and_unix(input_path: str, expected: str) -> None:
    assert module.normalize_rel(input_path) == expected


@pytest.mark.parametrize("bad", ["/test/images/a.jpg", "train/../a.jpg", "train/images/a/b.jpg"])
def test_normalize_rel_rejects_unsafe_paths(bad: str) -> None:
    with pytest.raises(ValueError):
        module.normalize_rel(bad)


def test_label_path_is_platform_independent() -> None:
    expected = "train/labels/IMG_00001.txt"
    assert module.label_for_image("train/images/IMG_00001.jpg") == expected
    assert module.label_for_image(r"train\images\IMG_00001.jpg") == expected


def test_scan_counts_uses_real_scan_dataset_schema() -> None:
    report = {
        "split_stats": {
            "train": {"images": 3, "objects_parsed": 5},
            "valid": {"images": 2, "objects_parsed": 4},
            "test": {"images": 1, "objects_parsed": 2},
        }
    }
    images, objects = module.scan_counts(report)
    assert images == {"train": 3, "valid": 2, "test": 1}
    assert objects == {"train": 5, "valid": 4, "test": 2}
    assert "image_count" not in report
