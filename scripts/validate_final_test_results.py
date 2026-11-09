"""Validate the three locked TEST reports and write one presentation summary.

This script never performs inference. It validates current imported Kaggle
reports against their original per-model VALID locks, and also supports legacy
local FINAL_LOCKED reports. Inconsistent provenance or metric protocols fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "app/evaluation_lock.json"
REPORTS = {
    model: ROOT / "results/30epochs" / model / "test/metrics.json"
    for model in ("E3", "E4", "E6")
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
        imported = lock.get("schema") == "imported-per-model-valid-locks-v1"
        expected_lock = lock["source_lock_sha256"][model_id] if imported else lock_hash
        checks = {
            "model": report.get("model") == model_id,
            "split": report.get("split") == "test",
            "status": report.get("status") in ({"FINAL_AFTER_VALID_LOCK", "FINAL_LOCKED_30EPOCH"} if imported else {"FINAL_LOCKED"}),
            "images": report.get("images") == EXPECTED_IMAGES,
            "objects": report.get("ground_truth_objects") == EXPECTED_OBJECTS,
            "lock": report.get("lock_sha256") == expected_lock,
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
        if imported:
            folder = path.parent.parent
            original_lock = folder / "evaluation_lock.json"
            metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
            if sha256(original_lock) != expected_lock:
                raise RuntimeError(f"{model_id} original VALID lock changed")
            if (report.get("checkpoint_sha256") != lock["checkpoints"][model_id] or
                    metadata.get("checkpoint_sha256") != lock["checkpoints"][model_id] or
                    metadata.get("data_yaml_sha256") != lock["data_yaml_sha256"] or
                    metadata.get("lock_sha256") != expected_lock or
                    report.get("data_yaml_sha256", metadata.get("data_yaml_sha256")) != lock["data_yaml_sha256"] or
                    report.get("prediction_sha256") != sha256(path.parent / "predictions.jsonl")):
                raise RuntimeError(f"{model_id} TEST provenance mismatch")
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
