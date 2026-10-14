"""Read-only inspection/validation helpers for the processed ROD detection dataset."""
from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SMALL_AREA_WARNING = 1e-6  # project heuristic, NOT a COCO size threshold
EPSILON = 1e-4


def load_classes(root: Path) -> list[str]:
    cfg_file = root / "data.yaml"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy {cfg_file}")
    config = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError("data.yaml không có 'names'")
    names = config["names"]
    if isinstance(names, list):
        ordered = [str(x) for x in names]
    elif isinstance(names, dict):
        mapping = {int(k): str(v) for k, v in names.items()}
        if sorted(mapping) != list(range(len(mapping))):
            raise ValueError("Class ID trong names phải liên tục từ 0")
        ordered = [mapping[i] for i in range(len(mapping))]
    else:
        raise ValueError("'names' phải là list hoặc dict")
    if not ordered or any(not x.strip() for x in ordered):
        raise ValueError("Class names rỗng hoặc không hợp lệ")
    if "nc" in config and int(config["nc"]) != len(ordered):
        raise ValueError(f"nc={config['nc']} nhưng names={len(ordered)}")
    return ordered


def index_paths(directory: Path, *, images: bool) -> tuple[dict[str, Path], list[str]]:
    """Index by case-insensitive relative stem; detect ambiguous image/label pairs."""
    indexed: dict[str, Path] = {}
    collisions: list[str] = []
    if not directory.is_dir():
        return indexed, collisions
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (IMAGE_EXTS if images else {".txt"}):
            continue
        key = path.relative_to(directory).with_suffix("").as_posix().casefold()
        if key in indexed:
            collisions.append(f"{indexed[key]} | {path}")
        else:
            indexed[key] = path
    return indexed, collisions


def parse_box(line: str, nc: int, eps: float = EPSILON) -> tuple[int, float, float, float, float]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Nhãn detection phải có đúng 5 trường, thực tế {len(fields)}")
    try:
        class_id = int(fields[0])
    except ValueError as exc:
        raise ValueError("class_id không phải số nguyên") from exc
    if not 0 <= class_id < nc:
        raise ValueError(f"class_id {class_id} ngoài [0, {nc-1}]")
    try:
        xc, yc, w, h = (float(s) for s in fields[1:])
    except ValueError as exc:
        raise ValueError("Tọa độ không phải số") from exc
    if not all(math.isfinite(v) for v in (xc, yc, w, h)):
        raise ValueError("Tọa độ có NaN/Inf")
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError("Tâm/kích thước bbox nằm ngoài khoảng [0,1]")
    if xc - w/2 < -eps or xc + w/2 > 1 + eps or yc - h/2 < -eps or yc + h/2 > 1 + eps:
        raise ValueError("Góc bbox vượt biên ảnh")
    return class_id, xc, yc, w, h


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # for identical-file detection, not for authentication
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def add_issue(issues: list[dict], severity: str, kind: str, split: str, path: Path | str, detail: str, line: int | None = None):
    issues.append({"severity": severity, "issue_type": kind, "split": split,
                   "file": str(path), "line": line or "", "detail": str(detail)})


def scan_dataset(root: Path, *, check_image_integrity: bool = False, hash_images: bool = False) -> dict:
    """Inspect processed YOLO boxes. Never writes inside root."""
    root = root.resolve()
    names = load_classes(root)
    issues: list[dict] = []
    records: list[dict] = []
    split_stats: dict[str, dict] = {}
    class_objects = {s: Counter() for s in SPLITS}
    class_images = {s: Counter() for s in SPLITS}
    digest_paths: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    per_class_areas: dict[int, list[float]] = defaultdict(list)

    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.is_dir():
            add_issue(issues, "ERROR", "missing_image_directory", split, image_dir, "Thiếu thư mục ảnh")
        if not label_dir.is_dir():
            add_issue(issues, "ERROR", "missing_label_directory", split, label_dir, "Thiếu thư mục nhãn")
        images, image_collisions = index_paths(image_dir, images=True)
        labels, label_collisions = index_paths(label_dir, images=False)
        for detail in image_collisions:
            add_issue(issues, "ERROR", "ambiguous_image_stem", split, image_dir, detail)
        for detail in label_collisions:
            add_issue(issues, "ERROR", "ambiguous_label_stem", split, label_dir, detail)
        for key in sorted(images.keys() - labels.keys()):
            add_issue(issues, "ERROR", "missing_label", split, images[key], "Không có file nhãn")
        for key in sorted(labels.keys() - images.keys()):
            add_issue(issues, "ERROR", "missing_image", split, labels[key], "Không có file ảnh")
        num_empty = 0
        num_objects = 0
        for key, img in sorted(images.items()):
            lbl = labels.get(key)
            boxes = []
            if lbl:
                try:
                    contents = lbl.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError) as exc:
                    add_issue(issues, "ERROR", "unreadable_label", split, lbl, str(exc))
                    contents = ""
                seen_lines: set[str] = set()
                for line_number, line in enumerate(contents.splitlines(), start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped in seen_lines:
                        add_issue(issues, "ERROR", "duplicate_annotation", split, lbl,
                                  "Dòng nhãn trùng hoàn toàn", line_number)
                    seen_lines.add(stripped)
                    try:
                        box = parse_box(stripped, len(names))
                    except ValueError as exc:
                        add_issue(issues, "ERROR", "invalid_label", split, lbl, str(exc), line_number)
                        continue
                    boxes.append(box)
                    class_objects[split][box[0]] += 1
                    per_class_areas[box[0]].append(box[3] * box[4])
                    if box[3] * box[4] < SMALL_AREA_WARNING:
                        add_issue(issues, "WARNING", "tiny_box", split, lbl,
                                  f"bbox area_ratio {box[3] * box[4]:.9f}; cần xem lại", line_number)
            if not boxes:
                num_empty += 1
            for cls in set(b[0] for b in boxes):
                class_images[split][cls] += 1
            num_objects += len(boxes)
            width = height = None
            try:
                with Image.open(img) as im:
                    width, height = im.size
                    if width <= 0 or height <= 0:
                        raise ValueError("Image dimensions không hợp lệ")
                    if check_image_integrity:
                        im.verify()
            except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
                add_issue(issues, "ERROR", "unreadable_image", split, img, str(exc))
            if hash_images:
                try:
                    digest_paths[md5_file(img)].append((split, img))
                except OSError as exc:
                    add_issue(issues, "ERROR", "cannot_hash_image", split, img, str(exc))
            records.append({"split": split, "key": key, "image": img, "label": lbl,
                            "width": width, "height": height, "boxes": boxes,
                            "num_objects": len(boxes), "classes": set(b[0] for b in boxes)})
        split_stats[split] = {"images": len(images), "label_files": len(labels),
                              "objects_parsed": num_objects, "empty_or_invalid_labels": num_empty,
                              "missing_labels": len(images.keys() - labels.keys()),
                              "orphan_labels": len(labels.keys() - images.keys())}

    overlap = []
    within_split_duplicate_groups = 0
    if hash_images:
        for digest, entries in digest_paths.items():
            splits_here = {s for s, _ in entries}
            if len(splits_here) > 1:
                overlap.append({"md5": digest, "paths": [{"split": s, "file": str(p)} for s, p in entries]})
                add_issue(issues, "ERROR", "cross_split_duplicate", "/".join(sorted(splits_here)),
                          entries[0][1], f"MD5 {digest}: " + " | ".join(str(p) for _, p in entries))
            elif len(entries) > 1:
                within_split_duplicate_groups += 1
                add_issue(issues, "WARNING", "within_split_duplicate", entries[0][0], entries[0][1],
                          f"MD5 {digest}, {len(entries)} file giống nhau trong split")

    return {"root": root, "names": names, "issues": issues, "records": records,
            "split_stats": split_stats, "class_objects": class_objects,
            "class_images": class_images, "per_class_areas": per_class_areas,
            "overlap": overlap, "within_split_duplicate_groups": within_split_duplicate_groups,
            "image_hashes_evaluated": hash_images}


def write_csv(path: Path, header: list[str], rows: list[list | tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
