"""Embedded in both Kaggle notebooks. No local import is required at runtime."""

import hashlib
import json
import os
import re
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

    def __init__(self, api, token, repo_type, folder, model, config_hash, dataset_hash, work, max_epoch=10, repo_id=None, compact=False):
        if folder not in ("YOLO 11s", "Faster R-CNN", "E3_30epochs", "E4_30epochs", "E6_30epochs"):
            raise ValueError("Unknown experiment folder")
        if type(max_epoch) is not int or max_epoch not in (10, 30):
            raise ValueError("Invalid epoch budget")
        self.max_epoch = max_epoch
        self.compact = compact
        self.repo_id = HF_REPO_ID if repo_id is None else repo_id
        self.api, self.token, self.repo_type = api, token, repo_type
        self.folder, self.model = folder, model
        self.config_hash, self.dataset_hash = config_hash, dataset_hash
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    def _files(self):
        return set(self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type, token=self.token))

    def _download(self, remote):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type,
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
        validate_progress(manifest, self.folder, self.model, self.config_hash, self.dataset_hash, self.max_epoch)
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
        for alias, target in manifest.get("artifact_aliases", {}).items():
            if (alias not in ("best.pt", "best_weights.pth") or target not in ("checkpoint.pt", "recovery_full.pt")
                or target not in local or alias in local):
                raise RuntimeError("HF artifact alias không hợp lệ")
            local[alias] = local[target]
        return manifest, local

    def _compact_files(self, files):
        """Keep the full resume payload; remove only redundant/unneeded uploads."""
        recovery = "checkpoint.pt" if self.model == "fasterrcnn_resnet50_fpn_v2" else "recovery_full.pt"
        required = ({"checkpoint.pt", "metrics.csv", "config.json", "environment.json"}
                    if recovery == "checkpoint.pt" else
                    {"recovery_full.pt", "results.csv", "args.yaml", "config.json", "environment.json"})
        best = "best_weights.pth" if recovery == "checkpoint.pt" else "best.pt"
        if not required.issubset(files):
            raise RuntimeError("Compact upload thiếu file resume/config/metrics bắt buộc")
        selected = {name: source for name, source in files.items() if name in required | {best}}
        aliases = {}
        if best in selected and sha256_file(selected[best]) == sha256_file(selected[recovery]):
            aliases[best] = recovery
            del selected[best]
        return selected, aliases

    def _prune_compact(self, current):
        """Bound the live tree, never squash shared repo history during training.

        Only compact-v1 snapshots of this exact folder/config/dataset are owned.
        Unknown snapshots, source repositories, other models, and future progress
        are never deleted. A conditional commit fails if another writer commits.
        """
        if self.repo_id != HF_REPO_ID:
            raise RuntimeError("Refusing retention writes to source repository")
        head = self.api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type, token=self.token).sha
        files = set(self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type,
                                             token=self.token, revision=head))
        pointer_name = self.folder + "/latest.json"
        pointer = json.loads(self._download(pointer_name).read_text(encoding="utf-8"))
        if (pointer.get("snapshot") != current["snapshot"] or
            pointer.get("manifest_sha256") != sha256_file(self._download(current["snapshot"] + "/manifest.json"))):
            raise RuntimeError("Latest changed; retention skipped")
        prefix = self.folder + "/snapshots/"
        records = {}
        for name in sorted(files):
            if not name.startswith(prefix) or not name.endswith("/manifest.json"):
                continue
            snap = name.rsplit("/", 1)[0]
            if snap.count("/") != 2:
                continue
            m = json.loads(self._download(name).read_text(encoding="utf-8"))
            if (m.get("storage_policy") != "compact-v1" or m.get("folder") != self.folder or
                m.get("snapshot") != snap or m.get("model") != self.model or
                m.get("config_sha256") != self.config_hash or m.get("dataset_sha256") != self.dataset_hash):
                continue
            validate_progress(m, self.folder, self.model, self.config_hash, self.dataset_hash, self.max_epoch)
            if type(m.get("next_sample", 0)) is not int or m.get("next_sample", 0) < 0:
                raise RuntimeError("Invalid progress during retention")
            actual = {p for p in files if p.startswith(snap + "/")}
            expected = {snap + "/manifest.json"}
            for artifact in m["sha256"]:
                if artifact != Path(artifact).name or "/" in artifact or "\\" in artifact:
                    raise RuntimeError("Unsafe artifact path during retention")
                expected.add(snap + "/" + artifact)
            if actual == expected:
                records[snap] = m
        if current["snapshot"] not in records:
            raise RuntimeError("Current snapshot incomplete; no retention deletion")
        def progress(s):
            return records[s]["completed_epoch"], records[s].get("next_sample", 0)
        keep = {current["snapshot"]}
        for epoch in (10, 20, 30):
            complete = [s for s, m in records.items() if m["completed_epoch"] == epoch and m.get("next_sample", 0) == 0]
            if complete:
                keep.add(max(complete, key=lambda s: (s == current["snapshot"], s)))
        # Defensive fallback if an early/partial current snapshot has no best yet.
        best = [s for s, m in records.items() if
                {"best.pt", "best_weights.pth"} & (set(m["sha256"]) | set(m.get("artifact_aliases", {})))]
        if best and not ({"best.pt", "best_weights.pth"} &
                         (set(current["sha256"]) | set(current.get("artifact_aliases", {})))):
            keep.add(max(best, key=lambda s: (progress(s), s)))
        # Never prune a snapshot at more advanced progress, even if not latest yet.
        obsolete = {s for s in records if s not in keep and progress(s) <= progress(current["snapshot"])}
        paths = sorted(p for p in files if p.rsplit("/", 1)[0] in obsolete)
        if not paths:
            return
        from huggingface_hub import CommitOperationDelete
        self.api.create_commit(repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
            parent_commit=head, revision="main", operations=[CommitOperationDelete(path_in_repo=p) for p in paths],
            commit_message=f"{self.folder}: prune verified compact snapshots; keep latest/best/milestones")
        print("HF retention:", self.folder, "removed snapshots=", len(obsolete),
              "; retained latest/best/milestones. Git history unchanged.", flush=True)

    def _retry(self, action):
        for attempt in range(3):
            try:
                return action()
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                detail = getattr(exc, "server_message", None)
                if not detail and response is not None:
                    try:
                        body = response.json()
                        if isinstance(body, dict):
                            detail = body.get("error") or body.get("message")
                    except (ValueError, AttributeError):
                        pass
                # Some HF/Xet errors have only a plain-text HTTP response.
                body_text = getattr(response, "text", "") if response is not None else ""
                detail = str(detail or exc)
                if body_text and str(body_text) not in detail:
                    detail += "\nResponse: " + str(body_text)
                if self.token:
                    detail = detail.replace(self.token, "[REDACTED]")
                detail = re.sub(r"hf_[A-Za-z0-9_]+", "[REDACTED]", detail)
                detail = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", detail, flags=re.I)
                detail = re.sub(r"https?://[^\s]+", "[URL REDACTED]", detail)[:6000]
                diagnostic = f"{type(exc).__name__}; HTTP {status}; {detail}"
                record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "folder": self.folder, "operation": getattr(self, "_upload_operation", "upload"),
                          "attempt": attempt + 1, "error_type": type(exc).__name__,
                          "http_status": status, "server_message": detail,
                          "local_checkpoint_retained": True}
                # Keep diagnostics in Kaggle Output even when Save & Run All fails.
                try:
                    with (self.work / "hf_upload_errors.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                except OSError:
                    print("Không ghi được log HF ra đĩa; xem lỗi bên dưới.", flush=True)
                print("HF upload lỗi", attempt + 1, diagnostic, "; bản local còn ở", self.work, flush=True)
                if attempt == 2:
                    raise RuntimeError("HF upload thất bại; giữ Kaggle Output và KHÔNG đánh dấu đã đồng bộ. "
                                       + diagnostic) from None
                time.sleep(5 * (attempt + 1))

    def save(self, files, completed_epoch, extra=None):
        aliases = {}
        if self.compact:
            files, aliases = self._compact_files(files)
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
                    "completed_epoch": int(completed_epoch), "is_complete": int(completed_epoch) == self.max_epoch,
                    "sha256": hashes}
        if self.compact:
            manifest.update(storage_policy="compact-v1", artifact_aliases=aliases)
        validate_progress(manifest, self.folder, self.model, self.config_hash, self.dataset_hash, self.max_epoch)
        atomic_json(stage / "manifest.json", manifest)
        self._upload_operation = "upload_folder"
        print("HF upload_folder bắt đầu:", self.folder, "epoch", completed_epoch,
              "repo=", self.repo_id, "repo_type=", self.repo_type, "remote=", snap,
              "files=", len(files) + 1,
              "upload_MiB=", round(sum(p.stat().st_size for p in stage.iterdir()) / 1024**2, 2),
              "local=", stage, flush=True)
        self._retry(lambda: self.api.upload_folder(folder_path=str(stage), path_in_repo=snap,
                     repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
                     commit_message=f"{self.folder}: snapshot epoch {completed_epoch}"))
        # Verify server snapshot before moving pointer. A failed pointer commit leaves previous version valid.
        remote_manifest = self._download(snap + "/manifest.json")
        if sha256_file(remote_manifest) != sha256_file(stage / "manifest.json"):
            raise RuntimeError("HF upload manifest verification failed")
        pointer = self.work / "latest.json"
        atomic_json(pointer, {"snapshot": snap, "manifest_sha256": sha256_file(stage / "manifest.json")})
        self._upload_operation = "upload_latest_pointer"
        print("HF snapshot manifest verified; updating latest:", self.folder, flush=True)
        self._retry(lambda: self.api.upload_file(path_or_fileobj=str(pointer), path_in_repo=self.folder + "/latest.json",
                    repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
                    commit_message=f"{self.folder}: advance epoch {completed_epoch}"))
        remote_pointer = json.loads(self._download(self.folder + "/latest.json").read_text(encoding="utf-8"))
        if remote_pointer != json.loads(pointer.read_text(encoding="utf-8")):
            raise RuntimeError("HF latest pointer chưa được xác nhận sau upload")
        if self.compact:
            remote_files = self._files()
            if any(snap + "/" + name not in remote_files for name in hashes):
                raise RuntimeError("Uploaded snapshot missing an artifact; retention NOT run")
            try:
                self._prune_compact(manifest)
            except Exception as error:
                # New recovery is durable; cleanup conflicts must not lose it or
                # stop a successful epoch. Retry retention on the next save.
                warning = {"folder": self.folder, "snapshot": snap, "retention": "SKIPPED",
                           "error_type": type(error).__name__,
                           "http_status": getattr(getattr(error, "response", None), "status_code", None)}
                atomic_json(self.work / "retention_warning.json", warning)
                print("WARNING: retention skipped; checkpoint uploaded. Extra snapshots kept:",
                      type(error).__name__, flush=True)
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
        raise RuntimeError("Không xác định duy nhất repo_type của " + HF_REPO_ID + "; kiểm tra token/quyền/repo")
    return api, matches[0]
