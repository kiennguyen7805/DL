"""Standalone E4 notebook training cell (embedded by build_hf_autoresume_notebooks.py)."""

import json
import os
import shutil
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from kaggle_secrets import UserSecretsClient


E4_FOLDER = "YOLO 11s"
E4_MODEL = "yolo11s.pt"
E4_BUDGET = 10
E4_DEADLINE_HOURS = 10.0
E4_OUTPUT = Path("/kaggle/working/rod_e4_yolo11s_hf")
E4_OUTPUT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("E4 cần Kaggle GPU; chưa có CUDA")
if ultralytics.__version__ != "8.4.165":
    raise RuntimeError("E4 cần Ultralytics 8.4.165 để xác minh callback/checkpoint")

# FROZEN_E4_YAML is embedded from configs/e4_yolo11s_pretrained.yaml by the builder.
frozen = yaml.safe_load(FROZEN_E4_YAML)
if frozen["model"] != E4_MODEL or frozen["main_budget_epochs"] != 10 or frozen["epochs"] != 30:
    raise RuntimeError("Cấu hình E4 đã bị thay đổi ngoài thiết kế đã kiểm toán")

train_args = {k: v for k, v in frozen.items() if k not in ("experiment_id", "main_budget_epochs", "task", "mode")}
train_args.update(data=str(working_yaml), project=str(E4_OUTPUT), name="train", exist_ok=True,
                  model=E4_MODEL, device=0)
config_identity = dict(train_args)
config_identity["data"] = "ROD_FINAL_A4_TRAIN_VALID"
config_identity["project"] = "E4_OUTPUT"
config_identity.pop("exist_ok", None)
config_identity["ultralytics_version"] = ultralytics.__version__
config_hash = fingerprint(config_identity)
dataset_hash = sha256_file(source_yaml)
print("E4 config SHA256:", config_hash, "dataset YAML SHA256:", dataset_hash)
atomic_json(E4_OUTPUT / "config.json", config_identity)
atomic_json(E4_OUTPUT / "environment.json", {"torch": torch.__version__,
                                            "ultralytics": ultralytics.__version__,
                                            "gpu": torch.cuda.get_device_name(0)})

hf_token = UserSecretsClient().get_secret(HF_SECRET_NAME)
hf_api, repo_type = connect_hf(hf_token)
store = HFStore(hf_api, hf_token, repo_type, E4_FOLDER, E4_MODEL, config_hash,
                dataset_hash, E4_OUTPUT / "hf_backup")
restored = store.restore()
deadline = time.monotonic() + E4_DEADLINE_HOURS * 3600


def inspect_full_yolo_checkpoint(path, expected_completed=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("optimizer"), dict):
        raise RuntimeError("E4 checkpoint thiếu optimizer; không thể resume")
    if not ckpt["optimizer"].get("param_groups") or "state" not in ckpt["optimizer"]:
        raise RuntimeError("E4 optimizer state không đầy đủ")
    if not isinstance(ckpt.get("scaler"), dict) or ckpt.get("ema") is None:
        raise RuntimeError("E4 checkpoint thiếu AMP scaler hoặc EMA")
    completed = int(ckpt.get("epoch", -1)) + 1
    if not 1 <= completed <= E4_BUDGET:
        raise RuntimeError("E4 checkpoint epoch ngoài ngân sách")
    if expected_completed is not None and completed != expected_completed:
        raise RuntimeError("E4 checkpoint epoch khác HF manifest")
    args = ckpt.get("train_args", {})
    for key in ("epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "warmup_epochs",
                "momentum", "weight_decay", "close_mosaic", "seed", "amp"):
        if args.get(key) != frozen[key]:
            raise RuntimeError("E4 effective train args không khớp cấu hình: " + key)
    ema = ckpt["ema"]
    if getattr(ema, "nc", None) != 25 or getattr(ema, "yaml", {}).get("scale") != "s":
        raise RuntimeError("E4 checkpoint architecture hoặc class count không khớp")
    names = getattr(ema, "names", {})
    ordered = [names[i] if i in names else names[str(i)] for i in range(25)] if isinstance(names, dict) else list(names)
    if ordered != class_names:
        raise RuntimeError("E4 checkpoint class names/order không khớp ROD Final")
    return completed


if restored:
    state, recovered = restored
    if "recovery_full.pt" not in recovered:
        raise RuntimeError("E4 HF snapshot thiếu recovery_full.pt")
    completed = inspect_full_yolo_checkpoint(recovered["recovery_full.pt"], state["completed_epoch"])
    resume_file = E4_OUTPUT / "train" / "weights" / "last.pt"
    resume_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(recovered["recovery_full.pt"], resume_file)
    for name in ("results.csv", "args.yaml"):
        if name not in recovered:
            raise RuntimeError("E4 HF snapshot thiếu " + name)
        shutil.copy2(recovered[name], E4_OUTPUT / "train" / name)
    if "best.pt" in recovered:
        shutil.copy2(recovered["best.pt"], resume_file.parent / "best.pt")
else:
    completed, resume_file = 0, None

if completed == E4_BUDGET:
    print("E4 COMPLETED 10/10 theo manifest và full checkpoint; chỉ hậu kiểm, không train")
else:
    model = YOLO(str(resume_file) if resume_file else E4_MODEL)
    if not resume_file and len(model.names) != 80:
        raise RuntimeError("yolo11s.pt không phải COCO pretrained 80 class")

    def backup_epoch(trainer):
        epoch_done = int(trainer.epoch) + 1
        full = Path(trainer.last)
        inspect_full_yolo_checkpoint(full, epoch_done)
        # Keep a local full-state copy before final_eval strips last.pt.
        local_full = E4_OUTPUT / "recovery_full.pt"
        temp = E4_OUTPUT / "recovery_full.pt.tmp"
        shutil.copy2(full, temp)
        os.replace(temp, local_full)
        files = {"recovery_full.pt": local_full,
                 "last.pt": local_full,
                 "results.csv": Path(trainer.csv),
                 "args.yaml": Path(trainer.save_dir) / "args.yaml",
                 "config.json": E4_OUTPUT / "config.json",
                 "environment.json": E4_OUTPUT / "environment.json"}
        if Path(trainer.best).is_file():
            files["best.pt"] = Path(trainer.best)
        manifest = store.save(files, epoch_done, extra={"architecture": "YOLO11s", "class_count": 25,
                                                       "learning_rate_schedule_epochs": 30})
        atomic_json(E4_OUTPUT / "last_hf_manifest.json", manifest)
        if epoch_done >= E4_BUDGET or time.monotonic() >= deadline:
            trainer.stop = True
            print("E4 dừng an toàn sau epoch", epoch_done, "; HF snapshot đã xác minh")

    model.add_callback("on_model_save", backup_epoch)
    if resume_file:
        # 30-epoch scheduler remains in checkpoint. No restart from epoch 1.
        model.train(resume=True, data=str(working_yaml), device=0)
    else:
        model.train(**{k: v for k, v in train_args.items() if k != "model"})

latest = store.restore()
if latest is None:
    raise RuntimeError("E4 không có HF checkpoint sau phiên này")
final_state, final_files = latest
inspect_full_yolo_checkpoint(final_files["recovery_full.pt"], final_state["completed_epoch"])
print("E4 status:", final_state["completed_epoch"], "/10", "COMPLETE" if final_state["is_complete"] else "CONTINUE")
