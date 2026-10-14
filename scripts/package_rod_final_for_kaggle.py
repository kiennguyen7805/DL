"""Verify and package immutable ROD Final as one ZIP for Kaggle upload.

The dataset itself is read-only. Output is written outside data/processed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/rod_detection_final"
OUT = ROOT / "artifacts/kaggle"
SPLITS = ("train", "valid", "test")
EXPECTED = {"train": 18151, "valid": 3318, "test": 1611}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight() -> dict:
    cfg = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    if cfg.get("nc") != 25 or len(cfg.get("names", [])) != 25:
        raise ValueError("Expected exactly 25 classes")
    build = json.loads((DATASET / "final_build.json").read_text(encoding="utf-8"))
    if not build.get("frozen") or not build.get("verified"):
        raise ValueError("Dataset is not frozen and verified")
    counts = {}
    for split in SPLITS:
        images = list((DATASET / split / "images").glob("*"))
        labels = list((DATASET / split / "labels").glob("*.txt"))
        if len(images) != EXPECTED[split] or len(labels) != EXPECTED[split]:
            raise ValueError(f"Count mismatch in {split}")
        if {p.stem.casefold() for p in images} != {p.stem.casefold() for p in labels}:
            raise ValueError(f"Image/label stem mismatch in {split}")
        counts[split] = {"images": len(images), "labels": len(labels)}
    return {
        "version": build["version"], "input_fingerprint": build["input_fingerprint"],
        "counts": counts, "classes": cfg["names"],
        "data_yaml_sha256": sha256(DATASET / "data.yaml"),
        "final_build_sha256": sha256(DATASET / "final_build.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-zip", action="store_true", help="Create the large ZIP after preflight")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = preflight()
    (OUT / "rod_final_upload_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.write_zip:
        print("PREFLIGHT_PASS; add --write-zip to create the upload archive")
        return
    target = OUT / "rod_detection_final.zip"
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in sorted(DATASET.rglob("*")):
            if path.is_file():
                zf.write(path, (Path("rod_detection_final") / path.relative_to(DATASET)).as_posix())
    manifest["archive"] = str(target.relative_to(ROOT))
    manifest["archive_sha256"] = sha256(target)
    (OUT / "rod_final_upload_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote artifacts/kaggle/rod_detection_final.zip")


if __name__ == "__main__":
    main()
