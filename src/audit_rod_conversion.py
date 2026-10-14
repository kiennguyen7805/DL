"""Independent, read-only consistency audit of raw ROD annotations vs YOLO detection.

The converter implementation is deliberately NOT imported here. This module checks
its output against independently computed enclosing boxes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_names(yaml_path: Path) -> dict[int, str]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "names" not in cfg:
        raise ValueError(f"YAML thiếu names: {yaml_path}")
    names = cfg["names"]
    if isinstance(names, list):
        mapping = dict(enumerate(map(str, names)))
    elif isinstance(names, dict):
        mapping = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError("names phải là list hoặc dict")
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("class ID trong data.yaml phải liên tục từ 0")
    if "nc" in cfg and int(cfg["nc"]) != len(mapping):
        raise ValueError("nc không khớp names")
    return mapping


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def raw_to_box(line: str, nc: int, slack: float = 0.02):
    """Return (class_id, normalized xyxy, source_kind, was_clipped)."""
    parts = line.split()
    if not parts:
        raise ValueError("Dòng nhãn rỗng")
    try:
        cls = int(parts[0])
    except ValueError as exc:
        raise ValueError("Class ID phải là số nguyên") from exc
    if not 0 <= cls < nc:
        raise ValueError(f"Class ID ngoài phạm vi: {cls}")
    try:
        vals = [float(x) for x in parts[1:]]
    except ValueError as exc:
        raise ValueError("Tọa độ không phải số") from exc
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Tọa độ NaN/Inf")
    if len(vals) == 4:
        xc, yc, w, h = vals
        if w <= 0 or h <= 0:
            raise ValueError("Box không có diện tích dương")
        corners = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        kind = "box"
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        corners = (min(xs), min(ys), max(xs), max(ys))
        kind = "polygon"
    else:
        raise ValueError(f"Số trường nhãn không hợp lệ: {len(parts)}")
    x1, y1, x2, y2 = corners
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Hình gốc có diện tích bao ngoài bằng 0")
    if min(corners) < -slack or max(corners) > 1 + slack:
        raise ValueError(f"Tọa độ vượt ảnh quá giới hạn slack={slack}")
    fixed = tuple(clip(x) for x in corners)
    if fixed[0] >= fixed[2] or fixed[1] >= fixed[3]:
        raise ValueError("Box sau clip không còn diện tích dương")
    return cls, fixed, kind, any(abs(a - b) > 1e-10 for a, b in zip(corners, fixed))


def processed_to_box(line: str, nc: int, tol: float = 0.0001):
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Processed label phải có đúng 5 trường, thực tế {len(parts)}")
    try:
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
    except ValueError as exc:
        raise ValueError("Class/tọa độ processed không hợp lệ") from exc
    if cls < 0 or cls >= nc:
        raise ValueError(f"Class processed ngoài phạm vi: {cls}")
    if not all(math.isfinite(x) for x in (xc, yc, w, h)):
        raise ValueError("Tọa độ processed NaN/Inf")
    if w <= 0 or h <= 0:
        raise ValueError("Processed box không có diện tích dương")
    bbox = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
    if min(bbox) < -tol or max(bbox) > 1 + tol:
        raise ValueError("Processed box vượt biên ảnh")
    return cls, bbox


def index_stems(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Thiếu thư mục: {folder}")
    index = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            if p.stem in index:
                raise ValueError(f"Trùng basename trong {folder}: {p.stem}")
            index[p.stem] = p
    return index


def read_rows(path: Path, parser, nc: int, issues: list, split: str, label: str, **kw):
    rows = []
    for line_no, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            rows.append(parser(text, nc, **kw))
        except ValueError as exc:
            issues.append((split, label, "invalid_label", f"{path.name}:{line_no}: {exc}"))
    return rows


def compare_boxes(raw_rows: list, processed_rows: list, tol: float) -> tuple[int, list[str]]:
    """One-to-one matching by class and closeness; independent of label line order."""
    available = set(range(len(processed_rows)))
    errors = []
    matches = 0
    for cls, expected, kind, was_clipped in raw_rows:
        choices = [(max(abs(a - b) for a, b in zip(expected, found)), j)
                   for j, (other_cls, found) in enumerate(processed_rows)
                   if j in available and other_cls == cls]
        if not choices:
            errors.append(f"Không có box processed cho class {cls} ({kind})")
            continue
        distance, j = min(choices)
        if distance > tol:
            errors.append(f"Box class {cls} sai lệch {distance:.7f} > {tol}: raw={expected}, processed={processed_rows[j][1]}")
            continue
        available.remove(j)
        matches += 1
    for j in sorted(available):
        errors.append(f"Box processed dư: class={processed_rows[j][0]}, xyxy={processed_rows[j][1]}")
    return matches, errors


def audit(raw: Path, processed: Path, output: Path, tol: float, slack: float, n_samples: int, seed: int) -> dict:
    raw, processed, output = raw.resolve(), processed.resolve(), output.resolve()
    for source in (raw, processed):
        if output == source or source in output.parents:
            raise ValueError("--out không được nằm bên trong thư mục dataset gốc/processed")
    if not raw.is_dir() or not processed.is_dir():
        raise FileNotFoundError("Không tìm thấy raw hoặc processed dataset")
    raw_names = class_names(raw / "data.yaml")
    proc_names = class_names(processed / "data.yaml")
    if raw_names != proc_names:
        raise ValueError("Class mapping raw và processed data.yaml khác nhau")
    nc = len(proc_names)
    issues = []
    totals = Counter()
    per_split = {}
    viz_candidates = []
    for split in SPLITS:
        raw_images = index_stems(raw / split / "images", IMAGE_EXTS)
        proc_images = index_stems(processed / split / "images", IMAGE_EXTS)
        raw_labels = index_stems(raw / split / "labels", {".txt"})
        proc_labels = index_stems(processed / split / "labels", {".txt"})
        for group, left, right in (("image_stems", raw_images, proc_images),
                                   ("label_stems", raw_labels, proc_labels),
                                   ("raw_image_label", raw_images, raw_labels),
                                   ("processed_image_label", proc_images, proc_labels)):
            for missing in sorted(set(left) ^ set(right)):
                issues.append((split, missing, group + "_mismatch", "File chỉ tồn tại một phía"))
        matching = sorted(set(raw_labels) & set(proc_labels))
        counts = Counter(raw_images=len(raw_images), processed_images=len(proc_images),
                         raw_labels=len(raw_labels), processed_labels=len(proc_labels),
                         inspected_labels=len(matching))
        for stem in matching:
            src = raw_labels[stem]
            dst = proc_labels[stem]
            raw_rows = read_rows(src, raw_to_box, nc, issues, split, stem, slack=slack)
            converted = read_rows(dst, processed_to_box, nc, issues, split, stem, tol=tol)
            counts["raw_objects"] += len(raw_rows)
            counts["processed_objects"] += len(converted)
            counts["polygons"] += sum(r[2] == "polygon" for r in raw_rows)
            counts["raw_boxes"] += sum(r[2] == "box" for r in raw_rows)
            counts["clipped_objects"] += sum(r[3] for r in raw_rows)
            matched, errors = compare_boxes(raw_rows, converted, tol)
            counts["matched_objects"] += matched
            for error in errors:
                issues.append((split, stem, "conversion_mismatch", error))
            if any(r[2] == "polygon" for r in raw_rows):
                viz_candidates.append((split, stem, "polygon_converted"))
            if any(r[3] for r in raw_rows):
                viz_candidates.append((split, stem, "raw_box_or_polygon_clipped"))
        per_split[split] = dict(counts)
        totals.update(counts)

    summary = {"status": "PASS" if not issues else "FAIL",
               "raw_root": str(raw), "processed_root": str(processed),
               "classes": nc, "class_names": proc_names,
               "tolerance": tol, "source_clip_slack": slack,
               "splits": per_split, "totals": dict(totals),
               "issue_count": len(issues), "issues_preview": issues[:20],
               "note": "Đối chiếu bằng bộ kiểm tra độc lập, không sửa nhãn. PASS tự động không thay thế visual review."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversion_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "conversion_issues.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(("split", "label", "issue", "details"))
        writer.writerows(issues)
    rng = random.Random(seed)
    rng.shuffle(viz_candidates)
    with (output / "review_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(("split", "image_stem", "reason"))
        writer.writerows(viz_candidates[:n_samples])
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only audit of ROD polygon/box -> detection conversion")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/archive/ROD-Dataset/dataset"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed/rod_detection"))
    ap.add_argument("--out", type=Path, default=Path("results/lam_sach_du_lieu/conversion_audit"))
    ap.add_argument("--tol", type=float, default=0.0001, help="Max normalized coordinate error")
    ap.add_argument("--slack", type=float, default=0.02, help="Maximum raw coordinate overshoot allowed for clipping")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.tol <= 0 or args.slack < 0 or args.samples < 0:
        ap.error("tol > 0, slack >= 0, samples >= 0 là bắt buộc")
    try:
        result = audit(args.raw, args.processed, args.out, args.tol, args.slack, args.samples, args.seed)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print("STATUS:", result["status"])
    print("REPORT:", args.out / "conversion_audit.json")
    print("CLASSES:", result["classes"])
    for split, counts in result["splits"].items():
        print(f"{split}: {counts['raw_images']} raw images, {counts['processed_images']} processed images, "
              f"{counts['raw_objects']} raw objects, {counts['matched_objects']} matched objects")
    print("ISSUES:", result["issue_count"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

