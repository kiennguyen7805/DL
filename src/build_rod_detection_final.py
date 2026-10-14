"""Build the A4 ROD Final dataset from immutable V4 Candidate.

Only image/label pairs selected by direct multi-metric evidence or the A4
visual review allowlist are excluded. TEST is never modified. Pending bbox and
taxonomy edits are deliberately not applied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

from rod_dataset_quality import load_classes, scan_dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/rod_detection_v4_candidate"
DEST = ROOT / "data/processed/rod_detection_final"
STAGE = DEST.with_name(DEST.name + ".staging")
V3 = ROOT / "results/lam_sach_du_lieu/v3"
V4 = ROOT / "results/lam_sach_du_lieu/v4"
OUT = ROOT / "results/lam_sach_du_lieu/final"
SPLITS = ("train", "valid", "test")
PRIORITY = {"test": 2, "valid": 1, "train": 0}

# Direct visual inspection in A4 found these weak-metric groups to contain the
# same source scene/template. Quarantine only non-TEST members.
VISUAL_DUPLICATE_GROUPS = {
    "N0152", "N0180", "N0223", "N0235", "N0239", "N0240",
    "N0241", "N0242", "N0245", "N0246", "N0265",
}
VISUAL_FALSE_POSITIVE_GROUPS = {"N0173", "N0213"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def normalize_rel(value: str) -> str:
    """Return a safe, platform-independent relative manifest path."""
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or len(path.parts) != 3 or any(x in ("", ".", "..") for x in path.parts):
        raise ValueError(f"Unsafe relative path: {value}")
    return path.as_posix()


def label_for_image(image: str) -> str:
    path = PurePosixPath(normalize_rel(image))
    if path.parts[1] != "images":
        raise ValueError(f"Expected image path: {image}")
    return PurePosixPath(path.parts[0], "labels", path.stem + ".txt").as_posix()


def split_of(image: str) -> str:
    split = PurePosixPath(normalize_rel(image)).parts[0]
    if split not in SPLITS:
        raise ValueError(f"Unsafe split: {image}")
    return split


def strong(row: dict[str, str]) -> bool:
    geom = (int(row["orb_inliers"]) >= 40
            and float(row["orb_spread"]) >= 0.15
            and float(row["orb_inlier_ratio"]) >= 0.45)
    pixel = float(row["ssim_gray"]) >= 0.95 and float(row["rgb_mae"]) <= 15
    return geom or pixel


def present(image: str) -> bool:
    return (SOURCE / normalize_rel(image)).is_file()


def add(selected: dict[str, dict], image: str, group: str, reason: str,
        witness: str, metrics: str, panel: str) -> None:
    image = normalize_rel(image)
    witness = normalize_rel(witness) if witness else ""
    if split_of(image) == "test":
        raise ValueError(f"Refusing to exclude TEST: {image}")
    image_path = SOURCE / image
    label = label_for_image(image)
    label_path = SOURCE / label
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(image)
    lines = [x for x in label_path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    row = dict(group_id=group, split=split_of(image), image=image, label=label,
               image_sha256=digest(image_path), label_sha256=digest(label_path),
               object_count=len(lines), object_classes=" | ".join(x.split()[0] for x in lines),
               reason=reason, witness_image=witness,
               witness_sha256=digest(SOURCE / witness) if witness and present(witness) else "",
               metrics=metrics, original_panel=panel)
    old = selected.get(image)
    if old is None or reason.startswith("DIRECT_STRONG"):
        selected[image] = row


def preflight() -> tuple[list[dict], str, list[dict]]:
    if load_classes(SOURCE)[:25] != load_classes(SOURCE) or len(load_classes(SOURCE)) != 25:
        raise ValueError("Expected exactly 25 classes")
    triage = read_csv(V3 / "leakage_triage.csv")
    pairs = read_csv(V3 / "leakage_pair_metrics.csv")
    priority = read_csv(V4 / "priority_leakage_review.csv")
    secondary = read_csv(V4 / "secondary_test_leakage_review.csv")
    indirect = read_csv(V4 / "indirect_survivors_direct_check.csv")
    status = {r["group_id"]: r["status"] for r in triage}
    panel = {r["case_id"]: r["original_panel"] for r in priority + secondary}
    selected: dict[str, dict] = {}

    # Resolve every remaining POSSIBLE group only through strong direct
    # cross-split edges. Lower-priority endpoint is excluded; no transitive
    # propagation is allowed.
    for p in pairs:
        gid, a, b = p["group_id"], p["image_a"], p["image_b"]
        if status.get(gid) != "POSSIBLE_DUPLICATE" or not present(a) or not present(b):
            continue
        sa, sb = split_of(a), split_of(b)
        if sa == sb or not strong(p):
            continue
        if PRIORITY[sa] == PRIORITY[sb]:
            continue
        remove, keep = (a, b) if PRIORITY[sa] < PRIORITY[sb] else (b, a)
        add(selected, remove, gid, "DIRECT_STRONG_POSSIBLE_GROUP_KEEP_HIGHER_SPLIT", keep,
            f"ssim={p['ssim_gray']};rgb_mae={p['rgb_mae']};orb_inliers={p['orb_inliers']};orb_spread={p['orb_spread']};orb_ratio={p['orb_inlier_ratio']}",
            panel.get(gid, p.get("original_panel", "")))

    # A4 visual decisions for 13 weak-metric TEST groups. Two are explicit
    # false positives; all non-TEST members in the other 11 are quarantined.
    sec = {r["case_id"]: r for r in secondary}
    if set(sec) != VISUAL_DUPLICATE_GROUPS | VISUAL_FALSE_POSITIVE_GROUPS:
        raise ValueError("Secondary review group set changed")
    for gid in sorted(VISUAL_DUPLICATE_GROUPS):
        row = sec[gid]
        members = [x.strip() for x in row["all_members"].split("|")]
        tests = [x for x in members if split_of(x) == "test" and present(x)]
        if not tests:
            raise ValueError(f"No retained TEST in {gid}")
        for image in members:
            if present(image) and split_of(image) != "test":
                add(selected, image, gid, "A4_VISUAL_SAME_SOURCE_RISK_KEEP_TEST", tests[0],
                    f"best_ssim={row['best_ssim']};best_orb_inliers={row['best_orb_inliers']}", row["original_panel"])

    # The 13 A3 survivors were compared directly against their retained TEST
    # image and visually confirmed as same-scene/source risk in A4.
    for row in indirect:
        image, keep = row["image"], row["retained_test"]
        if present(image):
            add(selected, image, row["group_id"], "A4_VISUAL_INDIRECT_SURVIVOR_SAME_SOURCE_RISK", keep,
                f"ssim={row['direct_ssim']};rgb_mae={row['direct_rgb_mae']};orb_inliers={row['direct_orb_inliers']};orb_spread={row['direct_orb_spread']}",
                row["original_panel"])

    decisions = []
    for row in priority:
        gid = row["case_id"]
        count = sum(x["group_id"] == gid for x in selected.values())
        decisions.append(dict(group_id=gid, category=row["category"],
                              decision="RESOLVED_DIRECT_EDGES" if count else "NO_DIRECT_DUPLICATE_RETAINED",
                              excluded_images=count,
                              reason="Only direct strong cross-split edges were acted on; no whole-component exclusion",
                              original_panel=row["original_panel"]))
    for row in secondary:
        gid = row["case_id"]
        decision = "FALSE_POSITIVE_RETAIN" if gid in VISUAL_FALSE_POSITIVE_GROUPS else "VISUAL_SAME_SOURCE_QUARANTINE_NON_TEST"
        decisions.append(dict(group_id=gid, category=row["category"], decision=decision,
                              excluded_images=sum(x["group_id"] == gid for x in selected.values()),
                              reason="A4 original-panel visual inspection", original_panel=row["original_panel"]))

    inputs = [V3 / "leakage_triage.csv", V3 / "leakage_pair_metrics.csv",
              V4 / "priority_leakage_review.csv", V4 / "secondary_test_leakage_review.csv",
              V4 / "indirect_survivors_direct_check.csv", SOURCE / "data.yaml",
              SOURCE / "candidate_build.json"]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return sorted(selected.values(), key=lambda r: (r["split"], r["image"])), fingerprint, decisions


def build(selected: list[dict], fingerprint: str) -> None:
    if DEST.exists() or STAGE.exists():
        raise FileExistsError(f"Output exists: {DEST} or {STAGE}")
    excluded_images = {normalize_rel(r["image"]) for r in selected}
    excluded_labels = {normalize_rel(r["label"]) for r in selected}
    for split in SPLITS:
        for folder in ("images", "labels"):
            dst = STAGE / split / folder
            dst.mkdir(parents=True, exist_ok=True)
            excluded = excluded_images if folder == "images" else excluded_labels
            for src in (SOURCE / split / folder).iterdir():
                rel = (Path(split) / folder / src.name).as_posix()
                if rel not in excluded:
                    shutil.copy2(src, dst / src.name)
    cfg = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(DEST.resolve())
    (STAGE / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (STAGE / "final_build.json").write_text(json.dumps({
        "version": "ROD_FINAL_A4", "source": str(SOURCE), "input_fingerprint": fingerprint,
        "a4_excluded_images": len(selected), "pending_bbox_edits_applied": 0,
        "pending_taxonomy_edits_applied": 0, "test_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "frozen": False,
    }, indent=2), encoding="utf-8")


def scan_counts(scan: dict) -> tuple[dict[str, int], dict[str, int]]:
    stats = scan["split_stats"]
    return ({s: stats[s]["images"] for s in SPLITS},
            {s: stats[s]["objects_parsed"] for s in SPLITS})


def v4_baseline() -> tuple[dict[str, dict], list[dict[str, str]]]:
    report = json.loads((V4 / "verify_report.json").read_text(encoding="utf-8"))
    return report["after_split"], read_csv(V4 / "class_counts_before_after.csv")


def verify_test_unchanged(root: Path) -> int:
    checked = 0
    for folder in ("images", "labels"):
        source = SOURCE / "test" / folder
        target = root / "test" / folder
        source_names = {p.name for p in source.iterdir() if p.is_file()}
        target_names = {p.name for p in target.iterdir() if p.is_file()}
        if source_names != target_names:
            raise ValueError(f"TEST {folder} file set changed")
        for name in sorted(source_names):
            if digest(source / name) != digest(target / name):
                raise ValueError(f"TEST checksum changed: {folder}/{name}")
            checked += 1
    return checked


def verify(selected: list[dict], fingerprint: str, decisions: list[dict], root: Path) -> dict:
    baseline, baseline_classes = v4_baseline()
    after = scan_dataset(root, check_image_integrity=True, hash_images=True)
    errors = [x for x in after["issues"] if x["severity"] == "ERROR"]
    if errors:
        raise ValueError(f"Dataset scan errors: {errors[:3]}")
    if after["names"] != load_classes(SOURCE) or len(after["names"]) != 25:
        raise ValueError("Class mapping changed")
    after_images, after_objects = scan_counts(after)
    expected = Counter(r["split"] for r in selected)
    for split in SPLITS:
        if after_images[split] != baseline[split]["images"] - expected[split]:
            raise ValueError(f"Image delta mismatch: {split}")
        if after["split_stats"][split]["label_files"] != after_images[split]:
            raise ValueError(f"Image/label count mismatch: {split}")
    checked_test_files = verify_test_unchanged(root)
    report = {
        "source": str(SOURCE), "dataset": str(DEST), "verified_staging": str(root),
        "input_fingerprint": fingerprint,
        "before_images": {s: baseline[s]["images"] for s in SPLITS}, "after_images": after_images,
        "before_objects": {s: baseline[s]["objects_parsed"] for s in SPLITS}, "after_objects": after_objects,
        "excluded_images": len(selected), "excluded_by_split": dict(expected),
        "class_count": len(after["names"]), "errors": len(errors),
        "missing_images": sum(after["split_stats"][s]["orphan_labels"] for s in SPLITS),
        "missing_labels": sum(after["split_stats"][s]["missing_labels"] for s in SPLITS),
        "exact_cross_split_groups": len(after["overlap"]),
        "same_split_exact_groups": after["within_split_duplicate_groups"],
        "test_files_checksum_verified": checked_test_files,
        "test_images_and_labels_unchanged": checked_test_files == 3222,
        "pending_bbox_edits_applied": 0, "pending_motorcycle_relabels_applied": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(OUT / "quarantined_images.csv", selected, list(selected[0]) if selected else ["image"])
    write_csv(OUT / "leakage_decisions.csv", decisions,
              ["group_id", "category", "decision", "excluded_images", "reason", "original_panel"])
    rows = []
    for i, name in enumerate(after["names"]):
        row = {"class_id": i, "class_name": name}
        base = baseline_classes[i]
        for split in SPLITS:
            row[f"v4_{split}"] = int(base[f"after_{split}"])
            row[f"final_{split}"] = after["class_objects"][split][i]
        row["v4_total"] = sum(int(base[f"after_{s}"]) for s in SPLITS)
        row["final_total"] = sum(after["class_objects"][s][i] for s in SPLITS)
        row["delta"] = row["final_total"] - row["v4_total"]
        rows.append(row)
    write_csv(OUT / "class_counts_v4_final.csv", rows, list(rows[0]))
    return report


def repair_staging_labels(selected: list[dict]) -> list[dict]:
    """Remove only checksum-verified orphan labels caused by the path bug."""
    expected = {normalize_rel(r["label"]): r for r in selected}
    orphaned: set[str] = set()
    for split in SPLITS:
        image_stems = {p.stem.casefold() for p in (STAGE / split / "images").iterdir() if p.is_file()}
        for label in (STAGE / split / "labels").iterdir():
            if label.is_file() and label.stem.casefold() not in image_stems:
                orphaned.add((PurePosixPath(split) / "labels" / label.name).as_posix())
    if orphaned != set(expected) or len(orphaned) != 178:
        raise ValueError(f"Staging orphan set mismatch: found={len(orphaned)}, expected={len(expected)}")
    log = []
    for rel in sorted(orphaned):
        row = expected[rel]
        stage_path, source_path = STAGE / rel, SOURCE / rel
        stage_hash = digest(stage_path)
        source_hash = digest(source_path)
        if stage_hash != row["label_sha256"] or source_hash != row["label_sha256"]:
            raise ValueError(f"Checksum mismatch; refusing removal: {rel}")
        log.append({"label": rel, "sha256": stage_hash, "manifest_image": row["image"],
                    "reason": "ORPHAN_FROM_WINDOWS_PATH_SEPARATOR_BUG"})
    # Mutation happens only after every one of the 178 entries passed.
    for item in log:
        (STAGE / item["label"]).unlink()
    write_csv(OUT / "staging_orphan_label_repair.csv", log,
              ["label", "sha256", "manifest_image", "reason"])
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--resume-staging", action="store_true")
    args = ap.parse_args()
    selected, fingerprint, decisions = preflight()
    print(json.dumps({"selected": len(selected), "by_split": Counter(r["split"] for r in selected),
                      "fingerprint": fingerprint}, default=dict))
    if args.dry_run:
        return
    if args.resume_staging:
        if not STAGE.is_dir() or DEST.exists():
            raise FileNotFoundError("Expected staging and no Final dataset")
        repaired = repair_staging_labels(selected)
        print(json.dumps({"repaired_orphan_labels": len(repaired)}))
    else:
        build(selected, fingerprint)
    report = verify(selected, fingerprint, decisions, STAGE)
    STAGE.rename(DEST)
    metadata = json.loads((DEST / "final_build.json").read_text(encoding="utf-8"))
    metadata["frozen"] = True
    metadata["verified"] = True
    (DEST / "final_build.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
