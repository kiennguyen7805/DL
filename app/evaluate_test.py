"""Final TEST pass. Refuses to read TEST until the VALID protocol is locked."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from app.calibrate_confidence import ground_truth
from app.confidence import auto_confidence
from app.evaluate_models import FOLDERS, IDS, ROOT, verify_lock
from app.evaluator import evaluate_records
from app.model_adapters import load_adapter, rod_names, sha256_file


def run_final_test(model_id: str, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise RuntimeError("TEST blocked: explicit --confirm-locked-test is required")
    verify_lock()  # No TEST path is opened before this gate.
    if model_id not in IDS:
        raise ValueError("Unknown model ID")
    option = IDS[model_id]
    images = ROOT / "data/processed/rod_detection_final/test/images"
    labels = ROOT / "data/processed/rod_detection_final/test/labels"
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) != 1611 or any(not (labels / f"{p.stem}.txt").is_file() for p in paths):
        raise RuntimeError("ROD Final TEST image/label count mismatch")
    output = ROOT / "results" / FOLDERS[model_id] / "metrics"
    cache = output / "common_test_predictions.jsonl"
    meta_path = output / "common_test_metadata.json"
    meta = {"model": model_id, "split": "test", "image_count": len(paths),
            "lock_sha256": sha256_file(ROOT / "app/evaluation_lock.json")}
    if (output / "common_test.json").exists():
        raise RuntimeError("TEST result already exists; refusing a second evaluation")
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != meta:
            raise RuntimeError("Partial TEST cache has a different evaluation lock")
    elif cache.exists():
        raise RuntimeError("Partial TEST cache lacks metadata")
    else:
        output.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    records = []
    if cache.exists():
        with cache.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) > len(paths) or any(row["image"] != path.name for row, path in zip(records, paths)):
            raise RuntimeError("Partial TEST cache does not match frozen split")
    adapter = load_adapter(option) if len(records) < len(paths) else None
    with cache.open("a", encoding="utf-8") as stream:
        for index in range(len(records), len(paths)):
            path = paths[index]
            with Image.open(path) as source:
                image = source.convert("RGB")
            pred = adapter.predict(image, 0.01)
            row = {
                "image": path.name, "width": image.width, "height": image.height,
                "gt": ground_truth(labels / f"{path.stem}.txt", image.width, image.height),
                "pred": [[cls, score, *box] for cls, score, box in zip(
                    pred.class_ids, pred.scores, pred.boxes_xyxy, strict=True)],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(row)
            if (index + 1) % 50 == 0:
                print(f"{model_id} TEST {index + 1}/{len(paths)}", flush=True)
    result = evaluate_records(records, auto_confidence(option).threshold)
    result.update({"model": model_id, "split": "test", "status": "FINAL_LOCKED",
                   "prediction_sha256": sha256_file(cache),
                   "lock_sha256": sha256_file(ROOT / "app/evaluation_lock.json")})
    (output / "common_test.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "common_test_per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name", "gt_count", "ap50", "ap50_95", "tp", "fp", "fn"])
        names = rod_names()
        for item in result["per_class"]:
            writer.writerow([item["class_id"], names[item["class_id"]], item["gt_count"],
                             item["ap50"], item["ap50_95"], item["tp"], item["fp"], item["fn"]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=IDS)
    parser.add_argument("--confirm-locked-test", action="store_true")
    args = parser.parse_args()
    print(run_final_test(args.model, confirmed=args.confirm_locked_test))


if __name__ == "__main__":
    main()
