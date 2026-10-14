"""Full ROD integrity/overlap verification and human-review image gallery (read-only)."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rod_dataset_quality import IMAGE_EXTS, SPLITS, scan_dataset, write_csv


def prioritized_records(data: dict, n: int, seed: int, candidates: Path | None = None) -> list[tuple[dict, str]]:
    """Coverage-oriented human visual review, with audit candidate support."""
    rng = random.Random(seed)
    records = [r for r in data["records"] if r["width"] and r["height"]]
    by_path = {str(r["image"].resolve()).casefold(): r for r in records}
    by_name: dict[str, list[dict]] = {}
    by_split_stem: dict[tuple[str, str], dict] = {}
    for r in records:
        by_name.setdefault(r["image"].name.casefold(), []).append(r)
        by_split_stem[(r["split"], r["image"].stem.casefold())] = r
    selected: list[tuple[dict, str]] = []
    chosen: set[tuple] = set()

    def choose(r: dict | None, reason: str):
        if r is None or len(selected) >= n:
            return
        key = (r["split"], r["key"])
        if key not in chosen:
            chosen.add(key)
            selected.append((r, reason))

    # Always include train/valid/test even if audit candidates list is dominated by train.
    for split in SPLITS:
        sample = [r for r in records if r["split"] == split and r["boxes"]]
        choose(max(sample, key=lambda r: r["num_objects"], default=None), "split_coverage")

    if candidates and candidates.is_file():
        with candidates.open("r", encoding="utf-8-sig", newline="") as f:
            candidate_rows = list(csv.DictReader(f))
            candidate_rows.sort(key=lambda row: "clipped" not in row.get("reason", ""))
            for row in candidate_rows:
                # Column names aren't assumed: search any cell containing an image filename.
                split = row.get("split", "").strip().lower()
                stem = Path(row.get("image_stem", "").strip()).stem.casefold()
                found = by_split_stem.get((split, stem)) if split and stem else None
                for cell in row.values() if found is None else ():
                    if not cell:
                        continue
                    value = str(cell).strip().replace("\\", "/")
                    if Path(value).suffix.lower() not in IMAGE_EXTS:
                        continue
                    candidate = by_path.get(str(Path(value).resolve()).casefold())
                    if candidate is None:
                        alternatives = by_name.get(Path(value).name.casefold(), [])
                        if len(alternatives) == 1:
                            candidate = alternatives[0]
                        elif len(alternatives) > 1:
                            # Resolve same filenames across splits using explicit path components.
                            path_parts = {p.casefold() for p in Path(value).parts}
                            matches = [r for r in alternatives if r["split"] in path_parts]
                            candidate = matches[0] if len(matches) == 1 else None
                    if candidate:
                        found = candidate
                        break
                if found is not None:
                    choose(found, "audit_" + row.get("reason", "candidate"))
                if sum(reason.startswith("audit_") for _, reason in selected) >= min(8, n//4):
                    break

    class_counts = Counter()
    for r in records:
        class_counts.update(r["classes"])
    for cls in sorted(class_counts, key=lambda i: (class_counts[i], i)):
        examples = [r for r in records if cls in r["classes"]]
        choose(max(examples, key=lambda r: r["num_objects"] == 1, default=None), f"class_coverage_{cls}")

    def small_area(r):
        return min((b[3] * b[4] for b in r["boxes"]), default=float("inf"))
    for r in sorted(records, key=small_area)[:20]:
        choose(r, "small_bbox")
    for r in sorted(records, key=lambda r: r["num_objects"], reverse=True)[:30]:
        choose(r, "crowded")
    remaining = [r for r in records if (r["split"], r["key"]) not in chosen]
    rng.shuffle(remaining)
    for r in remaining:
        choose(r, "seeded_random")
        if len(selected) >= n:
            break
    return selected


def render_review_image(rec: dict, names: list[str], out: Path) -> None:
    with Image.open(rec["image"]) as img:
        img = img.convert("RGB")
        canvas = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        width, height = img.size
        for cls, xc, yc, bw, bh in rec["boxes"]:
            x0 = max(0, min(width - 1, round((xc - bw/2) * width)))
            y0 = max(0, min(height - 1, round((yc - bh/2) * height)))
            x1 = max(0, min(width - 1, round((xc + bw/2) * width)))
            y1 = max(0, min(height - 1, round((yc + bh/2) * height)))
            canvas.rectangle([x0,y0,x1,y1], outline="lime", width=3)
            canvas.text((x0, max(0, y0 - 12)), f"{cls}: {names[cls]}", fill="yellow", font=font,
                        stroke_width=1, stroke_fill="black")
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=90)


def run(root: Path, out: Path, *, n: int, seed: int, candidates: Path | None = None,
        overlap_out: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = scan_dataset(root, check_image_integrity=True, hash_images=True)
    problems = data["issues"]
    errors = [p for p in problems if p["severity"] == "ERROR"]
    warn = [p for p in problems if p["severity"] == "WARNING"]
    write_csv(out / "bad_labels.csv", ["severity", "issue_type", "split", "file", "line", "detail"],
              [[p[k] for k in ("severity", "issue_type", "split", "file", "line", "detail")] for p in problems])
    overlap_report = {"cross_split_duplicate_groups": len(data["overlap"]),
                      "cross_split_duplicates": data["overlap"],
                      "within_split_duplicate_groups": data["within_split_duplicate_groups"],
                      "method": "MD5 exact file byte hash, not near-duplicate image similarity"}
    (out / "split_overlap.json").write_text(json.dumps(overlap_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if overlap_out is not None:
        overlap_out.parent.mkdir(parents=True, exist_ok=True)
        overlap_out.write_text(json.dumps(overlap_report, indent=2, ensure_ascii=False), encoding="utf-8")

    selected = prioritized_records(data, n, seed, candidates)
    review_entries = []
    for idx, (r, reason) in enumerate(selected, 1):
        filename = f"{idx:03d}_{r['split']}_{r['key'].replace('/', '__')}.jpg"
        dest = out / "viz" / filename
        try:
            render_review_image(r, data["names"], dest)
        except (OSError, ValueError) as exc:
            errors.append({"severity": "ERROR", "issue_type": "render_failed", "split": r["split"],
                           "file": str(r["image"]), "line": "", "detail": str(exc)})
            continue
        review_entries.append({"image": str(r["image"]), "overlay": str(dest),
                               "split": r["split"], "reason": reason})
    write_csv(out / "review_manifest.csv", ["image", "overlay", "split", "reason", "review_status"],
              [[x["image"], x["overlay"], x["split"], x["reason"], "PENDING"] for x in review_entries])

    lines = ["# HUMAN VISUAL REVIEW — BẮT BUỘC", "",
             "**Trạng thái: PENDING.** Codex không được tự ghi PASS bằng mắt thay người dùng.", "",
             "- Mở từng ảnh trong `viz/`, đối chiếu ảnh thật với bbox và class name.",
             "- Ghi `OK`, `LECH`, hoặc `NGHI_NGO` cho từng ảnh, bổ sung lý do nếu có.",
             "- Kiểm tra nghiêm ngặt ảnh từ audit (polygon/box clipped), vật nhỏ, class hiếm và các split.",
             "- Metadata clip count: converter summary=1.328, audit độc lập=190; geometry match 40.195/40.195.",
             "  Đây là discrepancy metadata còn mở; không tự sửa summary hoặc khẳng định nguyên nhân.",
             "- Nếu có lệch nhãn mang tính hệ thống: STOP, điều tra rồi mới train.", "",
             "| # | Split | Lý do chọn | Ảnh overlay | Kết quả | Ghi chú |",
             "|---|---|---|---|---|---|"]
    for idx, r in enumerate(review_entries, 1):
        lines.append(f"| {idx} | {r['split']} | {r['reason']} | `{Path(r['overlay']).name}` | PENDING | |")
    (out / "visual_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"status_auto": "PASS" if not errors else "FAIL",
              "human_visual_review": "PENDING", "ready_for_training": False,
              "total_images": sum(s["images"] for s in data["split_stats"].values()),
              "total_objects_parsed": sum(s["objects_parsed"] for s in data["split_stats"].values()),
              "num_classes": len(data["names"]), "splits": data["split_stats"],
              "error_count": len(errors), "warning_count": len(warn),
              "cross_split_duplicate_groups": len(data["overlap"]),
              "review_images_created": len(review_entries),
              "expected_review_count": n, "audit_clip_discrepancy": "1,328 vs 190; unresolved metadata difference",
              "notes": ["status_auto không bao hàm human visual review.",
                        "Không đào sâu test khi quyết định class subset/HPO; việc quét integrity test được phép.",
                        "MD5 chỉ phát hiện ảnh trùng hệt nội dung byte, không phát hiện near duplicates."]}
    (out / "verify_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/processed/rod_detection"))
    p.add_argument("--out", type=Path, default=Path("results/lam_sach_du_lieu/verify"))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates", type=Path, default=Path("results/lam_sach_du_lieu/conversion_audit/review_candidates.csv"))
    p.add_argument("--overlap-out", type=Path, default=Path("results/lam_sach_du_lieu/analysis/split_overlap.json"))
    args = p.parse_args()
    if not 20 <= args.n <= 50:
        p.error("--n phải nằm trong khoảng [20,50]")
    try:
        report = run(args.root, args.out, n=args.n, seed=args.seed,
                     candidates=args.candidates, overlap_out=args.overlap_out)
    except (OSError, ValueError) as exc:
        print("VERIFY FAIL:", exc)
        return 2
    print("AUTO STATUS:", report["status_auto"])
    print("HUMAN VISUAL REVIEW:", report["human_visual_review"])
    print("ERRORS:", report["error_count"], "WARNINGS:", report["warning_count"])
    print("CROSS-SPLIT DUPLICATE GROUPS:", report["cross_split_duplicate_groups"])
    print("REPORT:", args.out / "verify_report.json")
    return 0 if report["status_auto"] == "PASS" and report["review_images_created"] >= args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
