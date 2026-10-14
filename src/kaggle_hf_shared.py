"""Embedded in both Kaggle notebooks. No local import is required at runtime."""

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path


HF_REPO_ID = "iraprp/deeplearning"
HF_SECRET_NAME = "HF_TOKEN2"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def validate_progress(manifest, folder, model, config_hash, dataset_hash, max_epoch=10):
    if manifest.get("schema") != 2 or manifest.get("folder") != folder or manifest.get("model") != model:
        raise RuntimeError("HF manifest thuộc mô hình/folder khác hoặc sai schema")
    if manifest.get("config_sha256") != config_hash or manifest.get("dataset_sha256") != dataset_hash:
        raise RuntimeError("HF checkpoint khác config/dataset; dừng, không train lại")
    epoch = manifest.get("completed_epoch")
    if type(epoch) is not int or not 0 <= epoch <= max_epoch:
        raise RuntimeError("HF completed_epoch không hợp lệ")
    if manifest.get("is_complete") is not (epoch == max_epoch):
        raise RuntimeError("HF trạng thái COMPLETE không khớp epoch")
    if not isinstance(manifest.get("sha256"), dict) or not manifest["sha256"]:
        raise RuntimeError("HF manifest thiếu checksum")
    return epoch


class HFStore:
    """Immutable snapshot first, latest pointer second. Failed upload cannot advance latest."""

    def __init__(self, api, token, repo_type, folder, model, config_hash, dataset_hash, work):
        assert folder in ("YOLO 11s", "Faster R-CNN")
        self.api, self.token, self.repo_type = api, token, repo_type
        self.folder, self.model = folder, model
        self.config_hash, self.dataset_hash = config_hash, dataset_hash
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    def _files(self):
        return set(self.api.list_repo_files(repo_id=HF_REPO_ID, repo_type=self.repo_type, token=self.token))

    def _download(self, remote):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type=self.repo_type,
                                    filename=remote, token=self.token, local_dir=str(self.work / "download"),
                                    force_download=remote.endswith("latest.json")))

    def restore(self):
        files = self._files()  # connection/auth errors must propagate
        prefix = self.folder + "/"
        relevant = [name for name in files if name.startswith(prefix)]
        pointer_name = prefix + "latest.json"
        if pointer_name not in files:
            if relevant:
                raise RuntimeError("HF folder có dữ liệu nhưng thiếu latest.json; dừng để kiểm tra thủ công")
            return None
        pointer = json.loads(self._download(pointer_name).read_text(encoding="utf-8"))
        snap = pointer.get("snapshot")
        if not isinstance(snap, str) or not snap.startswith(prefix + "snapshots/") or ".." in snap:
            raise RuntimeError("HF latest pointer không hợp lệ")
        manifest_name = snap + "/manifest.json"
        if manifest_name not in files:
            raise RuntimeError("HF latest pointer trỏ tới snapshot thiếu manifest")
        manifest_path = self._download(manifest_name)
        if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
            raise RuntimeError("HF manifest SHA256 không khớp latest pointer")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_progress(manifest, self.folder, self.model, self.config_hash, self.dataset_hash)
        if manifest.get("snapshot") != snap:
            raise RuntimeError("HF snapshot ID không nhất quán")
        local = {}
        for name, expected in manifest["sha256"].items():
            if name != Path(name).name or snap + "/" + name not in files:
                raise RuntimeError("HF snapshot thiếu artifact hoặc tên file không hợp lệ")
            path = self._download(snap + "/" + name)
            if sha256_file(path) != expected:
                raise RuntimeError("HF artifact SHA256 không khớp: " + name)
            local[name] = path
        return manifest, local

    def _retry(self, action):
        for attempt in range(3):
            try:
                return action()
            except Exception as exc:
                print("HF upload lỗi", attempt + 1, type(exc).__name__, "; bản local còn ở", self.work)
                if attempt == 2:
                    raise RuntimeError("HF upload thất bại; giữ Kaggle Output và KHÔNG đánh dấu đã đồng bộ") from None
                time.sleep(5 * (attempt + 1))

    def save(self, files, completed_epoch, extra=None):
        snap = self.folder + "/snapshots/" + uuid.uuid4().hex
        stage = self.work / "snapshots" / snap.rsplit("/", 1)[-1]
        stage.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name, source in files.items():
            if name != Path(name).name or not Path(source).is_file():
                raise RuntimeError("Artifact không hợp lệ: " + name)
            target = stage / name
            # Kaggle checkpoint and stage live on the same filesystem. Hardlinks
            # avoid duplicating a large optimizer checkpoint during HF upload.
            # Fall back to a copy if the filesystem does not support hardlinks.
            try:
                os.link(Path(source).resolve(), target)
            except OSError:
                shutil.copy2(source, target)
            hashes[name] = sha256_file(target)
        manifest = {**(extra or {}), "schema": 2, "folder": self.folder, "model": self.model,
                    "snapshot": snap, "config_sha256": self.config_hash,
                    "dataset_sha256": self.dataset_hash,
                    "completed_epoch": int(completed_epoch), "is_complete": int(completed_epoch) == 10,
                    "sha256": hashes}
        validate_progress(manifest, self.folder, self.model, self.config_hash, self.dataset_hash)
        atomic_json(stage / "manifest.json", manifest)
        self._retry(lambda: self.api.upload_folder(folder_path=str(stage), path_in_repo=snap,
                     repo_id=HF_REPO_ID, repo_type=self.repo_type, token=self.token,
                     commit_message=f"{self.folder}: snapshot epoch {completed_epoch}"))
        # Verify server snapshot before moving pointer. A failed pointer commit leaves previous version valid.
        remote_manifest = self._download(snap + "/manifest.json")
        if sha256_file(remote_manifest) != sha256_file(stage / "manifest.json"):
            raise RuntimeError("HF upload manifest verification failed")
        pointer = self.work / "latest.json"
        atomic_json(pointer, {"snapshot": snap, "manifest_sha256": sha256_file(stage / "manifest.json")})
        self._retry(lambda: self.api.upload_file(path_or_fileobj=str(pointer), path_in_repo=self.folder + "/latest.json",
                    repo_id=HF_REPO_ID, repo_type=self.repo_type, token=self.token,
                    commit_message=f"{self.folder}: advance epoch {completed_epoch}"))
        remote_pointer = json.loads(self._download(self.folder + "/latest.json").read_text(encoding="utf-8"))
        if remote_pointer != json.loads(pointer.read_text(encoding="utf-8")):
            raise RuntimeError("HF latest pointer chưa được xác nhận sau upload")
        # Only temporary staging files are removed; the source checkpoint in
        # Kaggle Output and the verified immutable HF snapshot remain intact.
        if stage.resolve().parent != (self.work / "snapshots").resolve():
            raise RuntimeError("Local staging path outside expected snapshots directory")
        for staged_file in stage.iterdir():
            if not staged_file.is_file():
                raise RuntimeError("Unexpected directory in local staging snapshot")
            staged_file.unlink()
        stage.rmdir()
        print("HF snapshot verified:", self.folder, "epoch", completed_epoch,
              "next_sample", manifest.get("next_sample", 0), "local staging cleared; free_GB=",
              round(shutil.disk_usage(self.work).free / (1024 ** 3), 2))
        return manifest


def connect_hf(token):
    from huggingface_hub import HfApi
    if not token or not token.strip():
        raise RuntimeError("Thiếu Kaggle Secret HF_TOKEN2")
    api = HfApi(token=token)
    matches = []
    for kind in ("dataset", "model"):
        try:
            api.repo_info(repo_id=HF_REPO_ID, repo_type=kind)
            matches.append(kind)
        except Exception:
            pass
    if len(matches) != 1:
        raise RuntimeError("Không xác định duy nhất repo_type của iraprp/deeplearning; kiểm tra token/quyền/repo")
    return api, matches[0]
