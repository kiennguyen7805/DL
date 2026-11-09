"""Download completed E3/E4 snapshots from the new HF repo, without loading torch.

Run from the project root: python -m scripts.download_hf_checkpoints
Uses HF_TOKEN or the Hugging Face login cache; never modifies remote files.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from src.kaggle_hf_shared import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "13ffa/deep"
DATASET_HASH = "84a776e32c59063930b3e036d91876921163f48de70a9e85eb8c2b3cc2ddcf52"
MODELS = {
    "E3": ("YOLO 11n", "yolo11n.pt", "707ac9a6ed2eb1545007c8dd4ac752429731ca9b29cd585f8319c375d3de0d70"),
    "E4": ("YOLO 11s", "yolo11s.pt", "684c8c281633c537094601c84bf3cdea5b674126230a69b74004380d6276aa70"),
}
REQUIRED = {"recovery_full.pt", "best.pt", "results.csv", "args.yaml", "config.json", "environment.json"}


def safe_filename(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
        raise RuntimeError("Invalid artifact filename")
    return name


def download_model(fetch, experiment, output, epochs, provenance):
    """fetch reads only from the single commit selected by main()."""
    folder, model, config_hash = MODELS[experiment]
    pointer_path = fetch(folder + "/latest.json")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    snapshot = pointer.get("snapshot")
    prefix = folder + "/snapshots/"
    if not isinstance(snapshot, str) or not snapshot.startswith(prefix):
        raise RuntimeError("Invalid latest snapshot pointer")
    snapshot_id = safe_filename(snapshot[len(prefix):])
    manifest_path = fetch(snapshot + "/manifest.json")
    if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise RuntimeError("Manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != 2 or manifest.get("snapshot") != snapshot
            or manifest.get("folder") != folder or manifest.get("model") != model
            or manifest.get("dataset_sha256") != DATASET_HASH
            or manifest.get("config_sha256") != config_hash):
        raise RuntimeError("Snapshot model/config/dataset identity mismatch")
    if (type(manifest.get("completed_epoch")) is not int
            or manifest["completed_epoch"] != epochs
            or manifest.get("is_complete") is not True
            or manifest.get("target_epochs", epochs) != epochs
            or manifest.get("next_sample", 0) != 0):
        raise RuntimeError(f"{experiment}: latest snapshot is not COMPLETE at epoch {epochs}")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Missing artifact checksums")
    for name, checksum in hashes.items():
        safe_filename(name)
        if name in {"manifest.json", "latest.json", "download_receipt.json"}:
            raise RuntimeError("Reserved artifact filename")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise RuntimeError("Invalid SHA256")
    aliases = manifest.get("artifact_aliases", {})
    if not isinstance(aliases, dict) or any(
        alias != "best.pt" or target != "recovery_full.pt" or alias in hashes or target not in hashes
        for alias, target in aliases.items()
    ):
        raise RuntimeError("Invalid best checkpoint alias")
    if not REQUIRED.issubset(set(hashes) | set(aliases)):
        raise RuntimeError("Snapshot missing checkpoint/metrics/config artifacts")

    # Verify every artifact before publishing a usable local checkpoint directory.
    downloaded = {}
    for name, checksum in hashes.items():
        path = fetch(snapshot + "/" + name)
        if sha256_file(path) != checksum:
            raise RuntimeError("Artifact SHA256 mismatch: " + name)
        downloaded[name] = path
    destination = Path(output) / experiment / f"epoch{epochs:03d}" / snapshot_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in {**downloaded, "manifest.json": manifest_path, "latest.json": pointer_path}.items():
        temp = destination / (name + ".tmp")
        shutil.copyfile(path, temp)
        os.replace(temp, destination / name)
    for alias, target in aliases.items():
        temp = destination / (alias + ".tmp")
        shutil.copyfile(destination / target, temp)
        os.replace(temp, destination / alias)
    receipt = {**provenance, "experiment": experiment, "snapshot": snapshot,
               "completed_epoch": epochs, "verified": True,
               "best_sha256": sha256_file(destination / "best.pt"),
               "manifest_sha256": pointer["manifest_sha256"]}
    atomic_json(destination / "download_receipt.json", receipt)
    print(f"{experiment}: COMPLETE {epochs}/{epochs}, SHA256 PASS: {destination}", flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["E3", "E4"])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-type", choices=("auto", "dataset", "model"), default="auto")
    parser.add_argument("--epochs", type=int, choices=(10, 30), default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hf_checkpoints")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    kinds = ("dataset", "model") if args.repo_type == "auto" else (args.repo_type,)
    for kind in kinds:
        try:
            info = api.repo_info(repo_id=args.repo_id, repo_type=kind)
            break
        except RepositoryNotFoundError:
            if kind == kinds[-1]:
                raise RuntimeError("Cannot access HF repo. Set HF_TOKEN with read access or use hf auth login.") from None
    revision = info.sha
    print(f"Read-only download: {args.repo_id} ({kind}), commit {revision}", flush=True)

    def fetch(filename):
        return Path(hf_hub_download(repo_id=args.repo_id, repo_type=kind, revision=revision,
                                     filename=filename, token=token))

    for experiment in dict.fromkeys(args.models):
        download_model(fetch, experiment, args.output, args.epochs,
                       {"repo_id": args.repo_id, "repo_type": kind, "revision": revision})


if __name__ == "__main__":
    main()
