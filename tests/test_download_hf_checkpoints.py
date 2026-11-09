import json

import pytest

from scripts.download_hf_checkpoints import DATASET_HASH, MODELS, REQUIRED, download_model
from src.kaggle_hf_shared import sha256_file


def snapshot(tmp_path, *, alias=False, epoch=30):
    remote = tmp_path / "remote"
    folder, model, config = MODELS["E3"]
    snap = folder + "/snapshots/test123"
    directory = remote / snap
    directory.mkdir(parents=True)
    names = REQUIRED - {"best.pt"} if alias else REQUIRED
    hashes = {}
    for name in names:
        path = directory / name
        path.write_bytes(name.encode())
        hashes[name] = sha256_file(path)
    manifest = {"schema": 2, "folder": folder, "model": model, "snapshot": snap,
                "config_sha256": config, "dataset_sha256": DATASET_HASH,
                "completed_epoch": epoch, "is_complete": epoch == 30,
                "target_epochs": 30, "sha256": hashes,
                "artifact_aliases": {"best.pt": "recovery_full.pt"} if alias else {}}

    def publish():
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        (remote / folder / "latest.json").write_text(json.dumps(
            {"snapshot": snap, "manifest_sha256": sha256_file(path)}), encoding="utf-8")

    publish()
    return remote, directory, manifest, publish


def test_complete_snapshot_with_best_alias(tmp_path):
    remote, _, _, _ = snapshot(tmp_path, alias=True)
    dest = download_model(lambda name: remote / name, "E3", tmp_path / "out", 30,
                          {"repo_id": "13ffa/deep", "revision": "pinned"})
    assert (dest / "best.pt").read_bytes() == (dest / "recovery_full.pt").read_bytes()
    receipt = json.loads((dest / "download_receipt.json").read_text())
    assert receipt["revision"] == "pinned" and receipt["verified"]


@pytest.mark.parametrize("failure", ["partial", "hash", "manifest", "model", "path", "alias", "missing"])
def test_reject_invalid_snapshot_before_publishing(tmp_path, failure):
    remote, directory, manifest, publish = snapshot(tmp_path)
    if failure == "partial":
        manifest["completed_epoch"] = 29
    elif failure == "hash":
        (directory / "best.pt").write_bytes(b"corrupt")
    elif failure == "manifest":
        (directory / "manifest.json").write_bytes(b"corrupt")
    elif failure == "model":
        manifest["model"] = "yolo11s.pt"
    elif failure == "path":
        manifest["sha256"]["../outside.pt"] = "a" * 64
    elif failure == "alias":
        manifest["artifact_aliases"] = {"best.pt": "args.yaml"}
    elif failure == "missing":
        del manifest["sha256"]["recovery_full.pt"]
    if failure not in ("hash", "manifest"):
        publish()
    with pytest.raises(RuntimeError):
        download_model(lambda name: remote / name, "E3", tmp_path / "out", 30, {})
    assert not (tmp_path / "out").exists()
