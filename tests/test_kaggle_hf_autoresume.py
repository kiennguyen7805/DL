"""CPU-only contract tests. They do not claim Kaggle GPU integration passed."""

import ast
import json
from pathlib import Path

import pytest

from src.kaggle_hf_shared import HFStore, fingerprint, sha256_file, validate_progress


ROOT = Path(__file__).resolve().parents[1]


class FakeHF:
    def __init__(self, remote):
        self.remote = remote
        self.fail_pointer = False

    def list_repo_files(self, **kwargs):
        return [p.relative_to(self.remote).as_posix() for p in self.remote.rglob("*") if p.is_file()]

    def upload_folder(self, folder_path, path_in_repo, **kwargs):
        import shutil
        destination = self.remote / path_in_repo
        shutil.copytree(folder_path, destination)

    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        import shutil
        if self.fail_pointer:
            raise OSError("simulated interruption")
        dest = self.remote / path_in_repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path_or_fileobj, dest)


def make_store(tmp_path, folder="YOLO 11s"):
    api = FakeHF(tmp_path / "remote")
    api.remote.mkdir()
    store = HFStore(api, "fake-secret-never-printed", "dataset", folder, "yolo11s.pt" if folder == "YOLO 11s" else "fasterrcnn_resnet50_fpn_v2",
                    "c" * 64, "d" * 64, tmp_path / "work")
    store._download = lambda remote: api.remote / remote
    store._retry = lambda action: action()
    return store, api


def test_immutable_snapshot_and_resume(tmp_path):
    store, api = make_store(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"full optimizer checkpoint")
    assert store.restore() is None
    first = store.save({"recovery_full.pt": checkpoint}, 1)
    checkpoint.write_bytes(b"next checkpoint")
    second = store.save({"recovery_full.pt": checkpoint}, 2)
    assert first["snapshot"] != second["snapshot"]
    assert len(list((api.remote / store.folder / "snapshots").iterdir())) == 2
    assert list((store.work / "snapshots").iterdir()) == []
    manifest, files = store.restore()
    assert manifest["completed_epoch"] == 2
    assert files["recovery_full.pt"].read_bytes() == b"next checkpoint"
    assert manifest["sha256"]["recovery_full.pt"] == sha256_file(files["recovery_full.pt"])


def test_repeated_large_backups_do_not_accumulate_local_staging(tmp_path):
    store, _ = make_store(tmp_path, folder="Faster R-CNN")
    checkpoint = tmp_path / "checkpoint.pt"
    for epoch in range(1, 6):
        checkpoint.write_bytes(bytes([epoch]) * (2 * 1024 * 1024))
        store.save({"checkpoint.pt": checkpoint}, epoch, {"next_sample": 0})
        assert list((store.work / "snapshots").iterdir()) == []
        assert checkpoint.stat().st_size == 2 * 1024 * 1024
    assert store.restore()[0]["completed_epoch"] == 5


def test_failed_pointer_retains_previous_snapshot(tmp_path, monkeypatch):
    store, api = make_store(tmp_path)
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"first")
    store.save({"recovery_full.pt": ckpt}, 1)
    api.fail_pointer = True
    store._retry = HFStore._retry.__get__(store, HFStore)
    monkeypatch.setattr("src.kaggle_hf_shared.time.sleep", lambda _: None)
    ckpt.write_bytes(b"second")
    with pytest.raises(RuntimeError, match="upload thất bại"):
        store.save({"recovery_full.pt": ckpt}, 2)
    manifest, _ = store.restore()
    assert manifest["completed_epoch"] == 1
    assert len(list((store.work / "snapshots").iterdir())) == 1  # failure backup retained locally


def test_incomplete_remote_and_wrong_hash_fail_closed(tmp_path):
    store, api = make_store(tmp_path)
    folder = api.remote / store.folder
    folder.mkdir()
    (folder / "orphan.txt").write_text("partial")
    with pytest.raises(RuntimeError, match="latest.json"):
        store.restore()
    (folder / "orphan.txt").unlink()
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"checkpoint")
    store.save({"recovery_full.pt": ckpt}, 1)
    manifest, files = store.restore()
    files["recovery_full.pt"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA256"):
        store.restore()


def test_wrong_config_model_epoch_and_complete_rejected(tmp_path):
    store, _ = make_store(tmp_path)
    p = {"schema": 2, "folder": store.folder, "model": store.model,
         "config_sha256": store.config_hash, "dataset_sha256": store.dataset_hash,
         "completed_epoch": 9, "is_complete": False, "sha256": {"checkpoint.pt": "abc"}}
    assert validate_progress(p, store.folder, store.model, store.config_hash, store.dataset_hash) == 9
    for key, wrong in (("folder", "Faster R-CNN"), ("model", "wrong"),
                       ("config_sha256", "wrong"), ("dataset_sha256", "wrong"),
                       ("completed_epoch", 11), ("is_complete", True)):
        bad = dict(p, **{key: wrong})
        with pytest.raises(RuntimeError):
            validate_progress(bad, store.folder, store.model, store.config_hash, store.dataset_hash)


def test_folders_are_isolated(tmp_path):
    e4, api = make_store(tmp_path)
    e6 = HFStore(api, "fake", "dataset", "Faster R-CNN", "fasterrcnn_resnet50_fpn_v2",
                 "c" * 64, "d" * 64, tmp_path / "work_e6")
    e6._download = e4._download
    e6._retry = e4._retry
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"model")
    e4.save({"recovery_full.pt": ckpt}, 1)
    assert e6.restore() is None
    e6.save({"checkpoint.pt": ckpt}, 1, {"next_sample": 0})
    assert e4.restore()[0]["model"] == "yolo11s.pt"
    assert e6.restore()[0]["model"] == "fasterrcnn_resnet50_fpn_v2"


def test_notebooks_json_python_syntax_and_no_test_or_e5():
    for name in ("Kaggle_E4_YOLO11s_HF_AutoResume.ipynb", "Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb"):
        book = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        assert book["nbformat"] == 4
        assert all(c["execution_count"] is None and c["outputs"] == [] for c in book["cells"] if c["cell_type"] == "code")
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if not source.lstrip().startswith("%pip"):
                    compile(source, name, "exec")
                assert "E5_scratch" not in source
                assert "TEST_IMAGES = None" in "".join(book["cells"][3]["source"])
        assert "HF_TOKEN2" in json.dumps(book)


def test_e4_frozen_schedule_and_e6_class_mapping():
    e4 = (ROOT / "src/kaggle_hf_e4.py").read_text(encoding="utf-8")
    e6 = (ROOT / "src/kaggle_hf_e6.py").read_text(encoding="utf-8")
    assert 'frozen["epochs"] != 30' in e4
    assert 'E4_MODEL = "yolo11s.pt"' in e4
    assert 'labels.append(int(c) + 1)' in e6
    assert 'FastRCNNPredictor(features, 26)' in e6
    assert 'torch.Generator().manual_seed(20000 + ep)' in e6
    assert '"loss_sum"' in e6 and '"loss_count"' in e6
    assert "E6_DEADLINE = time.monotonic() + 9.5 * 3600" in e6
    assert "E6_EPOCHS, E6_BATCH, E6_BACKUP_BATCHES = 10, 2, 1800" in e6


def test_e6_resume_progress_validation():
    source = (ROOT / "src/kaggle_hf_e6.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "validate_e6_payload")
    env = {"config_hash": "c", "dataset_hash": "d", "E6_EPOCHS": 10, "E6_BATCH": 2,
           "math": __import__("math"),
           "train_ds": list(range(7)), "RuntimeError": RuntimeError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "validator", "exec"), env)
    payload = {k: object() for k in ("model", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")}
    payload.update(optimizer={"state": {}, "param_groups": [{}]}, scheduler={"last_epoch": 1}, scaler={})
    payload.update(current_epoch=1, completed_epoch=1, next_sample=4, loss_sum=3.0,
                   loss_count=2, metrics_rows=[{}], config_hash="c", dataset_hash="d")
    manifest = {"completed_epoch": 1, "next_sample": 4}
    assert env["validate_e6_payload"](payload, manifest) == (1, 1, 4)
    for key, value in (("next_sample", 3), ("current_epoch", 2), ("config_hash", "bad")):
        with pytest.raises(RuntimeError):
            env["validate_e6_payload"](dict(payload, **{key: value}), manifest)


def test_e4_full_checkpoint_epoch_optimizer_and_class_guard(tmp_path):
    import torch
    from types import SimpleNamespace

    source = (ROOT / "src/kaggle_hf_e4.py").read_text(encoding="utf-8")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "inspect_full_yolo_checkpoint")
    names = ["class" + str(i) for i in range(25)]
    frozen = {k: v for k, v in (("epochs", 30), ("imgsz", 640), ("batch", 8), ("optimizer", "SGD"),
                                  ("lr0", .01), ("lrf", .01), ("warmup_epochs", 3.0),
                                  ("momentum", .937), ("weight_decay", .0005), ("close_mosaic", 10),
                                  ("seed", 0), ("amp", True))}
    env = {"torch": torch, "E4_BUDGET": 10, "frozen": frozen, "class_names": names,
           "RuntimeError": RuntimeError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "e4guard", "exec"), env)
    ckpt = {"epoch": 3, "optimizer": {"state": {}, "param_groups": [{}]}, "scaler": {},
            "ema": SimpleNamespace(nc=25, yaml={"scale": "s"}, names=dict(enumerate(names))),
            "train_args": frozen}
    path = tmp_path / "resume.pt"
    torch.save(ckpt, path)
    assert env["inspect_full_yolo_checkpoint"](path, 4) == 4
    for key, bad in (("optimizer", None), ("epoch", 10),
                     ("ema", SimpleNamespace(nc=26, yaml={"scale": "s"}, names=dict(enumerate(names))))):
        torch.save(dict(ckpt, **{key: bad}), path)
        with pytest.raises(RuntimeError):
            env["inspect_full_yolo_checkpoint"](path, 4)


def test_e6_yolo_to_torchvision_target(tmp_path):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms.functional import to_tensor

    source = (ROOT / "src/kaggle_hf_e6.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RODDataset")
    env = {"Dataset": Dataset, "Path": Path, "Image": Image, "torch": torch,
           "math": __import__("math"), "to_tensor": to_tensor, "RuntimeError": RuntimeError}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "dataset", "exec"), env)
    images, labels = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    Image.new("RGB", (100, 50), "white").save(images / "a.jpg")
    (labels / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n24 0.25 0.5 0.2 0.4\n")
    image, target = env["RODDataset"](images)[0]
    assert image.shape == (3, 50, 100)
    assert target["labels"].tolist() == [1, 25]
    assert target["boxes"].tolist()[0] == pytest.approx([30, 15, 70, 35])
    (labels / "a.txt").write_text("25 0.5 0.5 0.4 0.4\n")
    with pytest.raises(RuntimeError):
        env["RODDataset"](images)[0]
    # Real ROD Final line: 0.7565101563 + 0.4869796875/2 = 1.00000000005.
    (labels / "a.txt").write_text("11 0.5 0.7565101563 1 0.4869796875\n")
    _, target = env["RODDataset"](images)[0]
    assert target["labels"].tolist() == [12]
    assert target["boxes"][0, 3].item() == 50.0
    (labels / "a.txt").write_text("11 0.5 0.76 1 0.50\n")
    with pytest.raises(RuntimeError):
        env["RODDataset"](images)[0]
