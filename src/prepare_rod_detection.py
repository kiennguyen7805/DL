"""Convert mixed ROD box/polygon labels to YOLO detection labels.

The source dataset is never modified. Minor source box overflow is clipped to
the image boundary. Images are hard-linked where possible and copied otherwise.
Run this once before training a detection model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


DEFAULT_SOURCE = Path("data/raw/archive/ROD-Dataset/dataset")
DEFAULT_OUTPUT = Path("data/processed/rod_detection")
SPLITS = ("train", "valid", "test")


def convert_line(line: str, nc: int) -> tuple[str, bool, bool]:
    parts = line.split()
    if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 != 1):
        raise ValueError(f"Unexpected number of fields: {len(parts)}")
    class_id = int(parts[0])
    if str(class_id) != parts[0] or not 0 <= class_id < nc:
        raise ValueError(f"Invalid class ID: {parts[0]}")
    values = [float(value) for value in parts[1:]]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("Coordinates must be in [0, 1]")

    polygon = len(parts) != 5
    if polygon:
        xs, ys = values[0::2], values[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        values = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    xc, yc, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("Box width and height must be positive")
    tolerance = 1e-3
    if (xc - width / 2 < -tolerance or xc + width / 2 > 1 + tolerance
            or yc - height / 2 < -tolerance or yc + height / 2 > 1 + tolerance):
        raise ValueError("Box extends outside image")
    x1, x2 = max(0, xc - width / 2), min(1, xc + width / 2)
    y1, y2 = max(0, yc - height / 2), min(1, yc + height / 2)
    clipped = not polygon and (x1 != xc - width / 2 or x2 != xc + width / 2
                               or y1 != yc - height / 2 or y2 != yc + height / 2)
    values = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return " ".join([str(class_id), *(f"{value:.10g}" for value in values)]), polygon, clipped


def prepare(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(source / "data.yaml")
    if output == source or source in output.parents:
        raise ValueError("Output must not be inside the source dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    nc = int(config["nc"])
    if len(names) != nc:
        raise ValueError("data.yaml: nc and names disagree")
    summary = {"source": str(source), "output": str(output), "splits": {}}

    for split in SPLITS:
        src_images, src_labels = source / split / "images", source / split / "labels"
        if not src_images.is_dir() or not src_labels.is_dir():
            raise FileNotFoundError(f"Missing images or labels for {split}")
        images = sorted(p for p in src_images.iterdir() if p.is_file())
        labels = {p.stem: p for p in src_labels.glob("*.txt")}
        if {p.stem for p in images} != set(labels):
            raise ValueError(f"Image/label filenames do not match in {split}")

        dst_images, dst_labels = output / split / "images", output / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        counts = {"images": len(images), "objects": 0, "boxes_kept": 0,
                  "polygons_converted": 0, "boxes_clipped": 0, "images_copied": 0}
        for image in images:
            label = labels[image.stem]
            converted = []
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    result, polygon, clipped = convert_line(line, nc)
                except (ValueError, OverflowError) as error:
                    raise ValueError(f"{label}:{line_number}: {error}") from error
                converted.append(result)
                counts["objects"] += 1
                counts["polygons_converted" if polygon else "boxes_kept"] += 1
                counts["boxes_clipped"] += clipped
            (dst_labels / label.name).write_text(
                "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
            )
            destination = dst_images / image.name
            try:
                os.link(image, destination)
            except OSError:
                shutil.copy2(image, destination)
                counts["images_copied"] += 1
        summary["splits"][split] = counts

    prepared_config = {"path": str(output), "train": "train/images",
                       "val": "valid/images", "test": "test/images",
                       "nc": nc, "names": names}
    (output / "data.yaml").write_text(
        yaml.safe_dump(prepared_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
