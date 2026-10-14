"""Validate the three locked TEST reports and write one presentation summary.

This script never performs inference.  It only accepts reports produced by
``python -m app.evaluate_test --confirm-locked-test`` and fails closed when a
report, class row, lock hash, or evaluation protocol is inconsistent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "app/evaluation_lock.json"
REPORTS = {
    "E3": ROOT / "results/E3_YOLO11n/metrics/common_test.json",
    "E4": ROOT / "results/E4_YOLO11s/metrics/common_test.json",
    "E6": ROOT / "results/E6_Faster_RCNN/metrics/common_test.json",
}
EXPECTED_IMAGES = 1611
EXPECTED_OBJECTS = 2354
EXPECTED_CLASSES = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> dict:
    lock_hash = sha256(LOCK)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    models: dict[str, dict] = {}
    protocols = []
    for model_id, path in REPORTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked TEST report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "model": report.get("model") == model_id,
            "split": report.get("split") == "test",
            "status": report.get("status") == "FINAL_LOCKED",
            "images": report.get("images") == EXPECTED_IMAGES,
            "objects": report.get("ground_truth_objects") == EXPECTED_OBJECTS,
            "lock": report.get("lock_sha256") == lock_hash,
            "per_class": len(report.get("per_class", [])) == EXPECTED_CLASSES,
            "class_ids": [row.get("class_id") for row in report.get("per_class", [])]
            == list(range(EXPECTED_CLASSES)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Invalid {model_id} TEST report ({', '.join(failed)}): {path}")
        expected_threshold = lock["thresholds"][model_id]
        if abs(float(report["operating_threshold"]) - float(expected_threshold)) > 1e-12:
            raise RuntimeError(f"{model_id} TEST threshold differs from evaluation lock")
        protocol = report.get("protocol")
        if protocol is not None:
            protocols.append(json.dumps(protocol, sort_keys=True))
        models[model_id] = report
    if protocols and len(set(protocols)) != 1:
        raise RuntimeError("TEST reports do not use the same evaluation protocol")
    return {
        "status": "FINAL_LOCKED",
        "split": "test",
        "evaluation_lock": str(LOCK.relative_to(ROOT)).replace("\\", "/"),
        "evaluation_lock_sha256": lock_hash,
        "images": EXPECTED_IMAGES,
        "ground_truth_objects": EXPECTED_OBJECTS,
        "models": models,
    }


def main() -> None:
    summary = load_and_validate()
    output = ROOT / "results/final_test_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: three locked TEST reports verified -> {output}")
    for model_id, report in summary["models"].items():
        print(
            f"{model_id}: P={report['precision']:.6f} R={report['recall']:.6f} "
            f"mAP50={report['map50']:.6f} mAP50-95={report['map50_95']:.6f}"
        )


if __name__ == "__main__":
    main()
