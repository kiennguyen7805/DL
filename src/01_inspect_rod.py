"""ROD exploratory data analysis, processed detection (read-only)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rod_dataset_quality import SPLITS, scan_dataset, write_csv


def median_or_none(seq):
    return median(seq) if seq else None


def run(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = scan_dataset(root, check_image_integrity=False, hash_images=False)
    names = data["names"]
    records = data["records"]
    all_counts = Counter()
    for split in SPLITS:
        all_counts.update(data["class_objects"][split])
    rows = []
    for cls, name in enumerate(names):
        rows.append([cls, name, *[data["class_objects"][s][cls] for s in SPLITS],
                     *[data["class_images"][s][cls] for s in SPLITS],
                     median_or_none(data["per_class_areas"][cls])])
    write_csv(out / "class_counts.csv",
              ["class_id", "class_name", "train_objects", "valid_objects", "test_objects",
               "train_images", "valid_images", "test_images", "median_area_ratio"], rows)

    write_csv(out / "objects_per_image.csv", ["split", "image", "num_objects", "width_px", "height_px"],
              [[r["split"], str(r["image"]), r["num_objects"], r["width"], r["height"]] for r in records])
    bbox_rows = []
    for r in records:
        for cls, _, _, w, h in r["boxes"]:
            bbox_rows.append([r["split"], cls, names[cls], w, h, w*h])
    write_csv(out / "bbox_stats.csv", ["split", "class_id", "class_name", "w", "h", "area_ratio"], bbox_rows)

    sizes = Counter((r["width"], r["height"]) for r in records if r["width"] and r["height"])
    vals = [n for n in all_counts.values() if n > 0]
    report = {
        "source": "Independent read-only scan of processed ROD, NOT model metrics",
        "root": str(data["root"]), "num_classes": len(names),
        "splits": data["split_stats"],
        "total_images": sum(x["images"] for x in data["split_stats"].values()),
        "total_objects_parsed": sum(x["objects_parsed"] for x in data["split_stats"].values()),
        "class_imbalance_max_min_nonzero": (max(vals)/min(vals) if vals else None),
        "class_top3_object_percent": (100 * sum(sorted(all_counts.values(), reverse=True)[:3]) / sum(all_counts.values())
                                      if sum(all_counts.values()) else None),
        "frequent_image_dimensions": [{"size": f"{w}x{h}", "count": n} for (w,h), n in sizes.most_common(10)],
        "annotation_error_count": sum(x["severity"] == "ERROR" for x in data["issues"]),
        "notes": ["Số liệu từ nhãn YOLO processed; scan ảnh/label không thay thế human visual review.",
                  "Kiểm tra ảnh hỏng và MD5 cross-split thuộc src/02_verify_rod.py."]
    }
    (out / "dataset_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(11, 7))
    plt.barh(range(len(names)), [all_counts[i] for i in range(len(names))])
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Number of annotated objects")
    plt.title("ROD class distribution (processed dataset)")
    plt.tight_layout()
    plt.savefig(out / "fig_class_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(11, 7))
    total_images_per_class = [sum(data["class_images"][s][i] for s in SPLITS) for i in range(len(names))]
    plt.barh(range(len(names)), total_images_per_class)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Images containing class")
    plt.title("ROD images per class")
    plt.tight_layout()
    plt.savefig(out / "fig_images_per_class.png", dpi=140)
    plt.close()

    bins_labels = ("0", "1", "2", "3-5", "6+")
    object_freq = Counter("0" if r["num_objects"] == 0 else "1" if r["num_objects"] == 1
                          else "2" if r["num_objects"] == 2 else "3-5" if r["num_objects"] <= 5 else "6+"
                          for r in records)
    plt.figure(figsize=(7, 4))
    plt.bar(bins_labels, [object_freq[x] for x in bins_labels])
    plt.xlabel("Objects per image")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(out / "fig_objects_per_image.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 5))
    areas = [row[5] for row in bbox_rows]
    if areas:
        plt.hist(areas, bins=40)
    plt.xlabel("Normalized bbox area ratio (w*h)")
    plt.ylabel("Number of objects")
    plt.title("ROD bbox area ratio; not COCO small/medium/large")
    plt.tight_layout()
    plt.savefig(out / "fig_bbox_area_ratio.png", dpi=140)
    plt.close()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/processed/rod_detection"))
    p.add_argument("--out", type=Path, default=Path("results/lam_sach_du_lieu/analysis"))
    p.add_argument("--full", action="store_true", help="Compatibility flag; always perform full EDA")
    args = p.parse_args()
    try:
        report = run(args.root, args.out)
    except (OSError, ValueError) as exc:
        print(f"EDA FAIL: {exc}")
        return 2
    print("EDA:", "PASS" if report["annotation_error_count"] == 0 else "FAIL")
    print("SUMMARY:", args.out / "dataset_summary.json")
    print("Images:", report["total_images"], "Objects:", report["total_objects_parsed"])
    return 0 if report["annotation_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
