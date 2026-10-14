"""Standalone E6 notebook training cell (embedded by build_hf_autoresume_notebooks.py)."""

import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from kaggle_secrets import UserSecretsClient
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


E6_FOLDER = "Faster R-CNN"
E6_MODEL = "fasterrcnn_resnet50_fpn_v2"
E6_WORK = Path("/kaggle/working/rod_e6_faster_rcnn_hf")
E6_WORK.mkdir(parents=True, exist_ok=True)
E6_EPOCHS, E6_BATCH, E6_BACKUP_BATCHES = 10, 2, 1800
E6_DEADLINE = time.monotonic() + 9.5 * 3600  # reserve ~2.5 h for HF upload / Kaggle shutdown

if not torch.cuda.is_available():
    raise RuntimeError("E6 cần Kaggle GPU; chưa có CUDA")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False


class RODDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)
        self.images = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        self.labels_dir = self.images_dir.parent / "labels"
        if not self.images or not self.labels_dir.is_dir():
            raise RuntimeError("ROD images/labels thiếu trong " + str(self.images_dir))
        if any(not (self.labels_dir / (p.stem + ".txt")).is_file() for p in self.images):
            raise RuntimeError("ROD có ảnh thiếu label trong " + str(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        p = self.images[index]
        with Image.open(p) as image:
            image = image.convert("RGB")
            width, height = image.size
            boxes, labels = [], []
            for row in (self.labels_dir / (p.stem + ".txt")).read_text(encoding="utf-8").splitlines():
                if not row.strip():
                    continue
                fields = row.split()
                if len(fields) != 5:
                    raise RuntimeError("YOLO annotation phải có 5 trường: " + str(p))
                values = [float(x) for x in fields]
                c, x, y, bw, bh = values
                left, top, right, bottom = x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2
                # Decimal YOLO labels can round an edge at 1.0 to 1.00000000005.
                # Accept only this tiny serialization error, then clip to the image.
                edge_tolerance = 1e-6
                if not (math.isfinite(c) and c.is_integer() and 0 <= c < 25 and
                        all(math.isfinite(v) for v in values[1:]) and 0 < bw <= 1 and 0 < bh <= 1 and
                        -edge_tolerance <= left < right <= 1 + edge_tolerance and
                        -edge_tolerance <= top < bottom <= 1 + edge_tolerance):
                    raise RuntimeError("YOLO class/bbox ngoài phạm vi: " + str(p))
                clipped = [max(0., left) * width, max(0., top) * height,
                           min(1., right) * width, min(1., bottom) * height]
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    raise RuntimeError("YOLO bbox rỗng sau khi xử lý sai số biên: " + str(p))
                boxes.append(clipped)
                labels.append(int(c) + 1)  # Torchvision background 0; ROD classes 1..25.
            b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target = {"boxes": b, "labels": torch.tensor(labels, dtype=torch.int64),
                      "image_id": torch.tensor([index], dtype=torch.int64),
                      "area": (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]),
                      "iscrowd": torch.zeros(len(labels), dtype=torch.int64)}
            return to_tensor(image), target


def collate(batch):
    return tuple(zip(*batch))


train_ds, val_ds = RODDataset(TRAIN_IMAGES), RODDataset(VAL_IMAGES)
if len(train_ds) != 18151 or len(val_ds) != 3318 or len(class_names) != 25:
    raise RuntimeError("ROD Final A4 split/class count không khớp freeze")

E6_CONFIG = {"model": E6_MODEL, "init": "COCO_V2_DEFAULT", "epochs": E6_EPOCHS,
             "batch": E6_BATCH, "seed": 0, "optimizer": "SGD", "lr": 0.005,
             "momentum": 0.9, "weight_decay": 0.0005, "scheduler": "StepLR(3,0.1)",
             "min_size": 640, "max_size": 960, "classes_with_background": 26,
             "train_count": len(train_ds), "valid_count": len(val_ds),
             "torch": torch.__version__, "torchvision": torchvision.__version__}
config_hash, dataset_hash = fingerprint(E6_CONFIG), sha256_file(source_yaml)
print("E6 config SHA256:", config_hash, "dataset YAML SHA256:", dataset_hash)
atomic_json(E6_WORK / "config.json", E6_CONFIG)
atomic_json(E6_WORK / "environment.json", {"torch": torch.__version__,
                                            "torchvision": torchvision.__version__,
                                            "gpu": torch.cuda.get_device_name(0)})
hf_token = UserSecretsClient().get_secret(HF_SECRET_NAME)
hf_api, repo_type = connect_hf(hf_token)
store = HFStore(hf_api, hf_token, repo_type, E6_FOLDER, E6_MODEL, config_hash,
                dataset_hash, E6_WORK / "hf_backup")
restored = store.restore()

net = fasterrcnn_resnet50_fpn_v2(weights=None if restored else FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                 weights_backbone=None, min_size=640, max_size=960)
features = net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(features, 26)
net.to("cuda")
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scaler = torch.amp.GradScaler("cuda")
current_epoch = completed_epoch = next_sample = 0
loss_sum, loss_count = 0.0, 0
metrics_rows, best_map = [], -1.0


def validate_e6_payload(saved, manifest):
    required = ("model", "optimizer", "scheduler", "scaler", "current_epoch", "completed_epoch",
                "next_sample", "loss_sum", "loss_count", "metrics_rows", "python_rng", "numpy_rng",
                "torch_rng", "cuda_rng", "config_hash", "dataset_hash")
    if any(key not in saved for key in required):
        raise RuntimeError("E6 checkpoint thiếu resume state")
    if saved["config_hash"] != config_hash or saved["dataset_hash"] != dataset_hash:
        raise RuntimeError("E6 checkpoint khác config/dataset")
    ep, done, offset = (saved[k] for k in ("current_epoch", "completed_epoch", "next_sample"))
    if (type(ep) is not int or type(done) is not int or type(offset) is not int or
            not 0 <= done <= E6_EPOCHS or not 0 <= ep <= E6_EPOCHS or
            not 0 <= offset <= len(train_ds) or ep != done or
            (offset and (offset % E6_BATCH != 0 and offset != len(train_ds))) or
            (done == E6_EPOCHS and (ep != E6_EPOCHS or offset != 0))):
        raise RuntimeError("E6 epoch/sample offset không hợp lệ")
    if manifest["completed_epoch"] != done or manifest.get("next_sample") != offset:
        raise RuntimeError("E6 checkpoint và HF manifest không nhất quán")
    if len(saved["metrics_rows"]) != done:
        raise RuntimeError("E6 metrics history thiếu epoch")
    if (not isinstance(saved["optimizer"], dict) or not saved["optimizer"].get("param_groups") or
            not isinstance(saved["scheduler"], dict) or saved["scheduler"].get("last_epoch") != done or
            not isinstance(saved["scaler"], dict)):
        raise RuntimeError("E6 optimizer/scheduler/AMP state không khớp số epoch")
    expected_batches = (offset + E6_BATCH - 1) // E6_BATCH
    if (type(saved["loss_count"]) is not int or saved["loss_count"] != expected_batches or
            not math.isfinite(float(saved["loss_sum"]))):
        raise RuntimeError("E6 mid-epoch loss/batch progress không nhất quán")
    return ep, done, offset


if restored:
    manifest, files = restored
    if "checkpoint.pt" not in files:
        raise RuntimeError("E6 HF snapshot thiếu checkpoint.pt")
    saved = torch.load(files["checkpoint.pt"], map_location="cpu", weights_only=False)
    current_epoch, completed_epoch, next_sample = validate_e6_payload(saved, manifest)
    net.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    metrics_rows, best_map = saved["metrics_rows"], float(saved["best_map"])
    loss_sum, loss_count = float(saved["loss_sum"]), int(saved["loss_count"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if "best_weights.pth" in files:
        import shutil
        shutil.copy2(files["best_weights.pth"], E6_WORK / "best_weights.pth")


def write_metrics():
    path = E6_WORK / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["epoch", "map50", "map5095", "mar100", "loss_mean"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    return path


def persist(ep, offset, done, running_loss, running_count):
    payload = {"model": net.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
               "current_epoch": int(ep), "next_sample": int(offset), "completed_epoch": int(done),
               "loss_sum": float(running_loss), "loss_count": int(running_count),
               "best_map": float(best_map), "metrics_rows": metrics_rows,
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
               "config_hash": config_hash, "dataset_hash": dataset_hash}
    path, temp = E6_WORK / "checkpoint.pt", E6_WORK / "checkpoint.pt.tmp"
    torch.save(payload, temp)
    os.replace(temp, path)
    files = {"checkpoint.pt": path, "metrics.csv": write_metrics(),
             "config.json": E6_WORK / "config.json",
             "environment.json": E6_WORK / "environment.json"}
    if (E6_WORK / "best_weights.pth").is_file():
        files["best_weights.pth"] = E6_WORK / "best_weights.pth"
    latest = store.save(files, done, extra={"next_sample": int(offset), "current_epoch": int(ep),
                                           "class_count": 25, "architecture": E6_MODEL})
    atomic_json(E6_WORK / "last_hf_manifest.json", latest)


@torch.inference_mode()
def evaluate():
    evaluator = MeanAveragePrecision(iou_type="bbox")
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate,
                        generator=torch.Generator().manual_seed(8000))
    net.eval()
    for images, targets in loader:
        predictions = net([image.to("cuda") for image in images])
        evaluator.update([{k: v.detach().cpu() for k, v in pred.items() if k in ("boxes", "labels", "scores")}
                          for pred in predictions],
                         [{k: v.detach().cpu() for k, v in target.items() if k in ("boxes", "labels")}
                          for target in targets])
    values = evaluator.compute()
    return float(values["map_50"]), float(values["map"]), float(values["mar_100"])


if completed_epoch == E6_EPOCHS:
    print("E6 COMPLETED 10/10 theo HF manifest và checkpoint; không train")
else:
    if not restored:
        persist(0, 0, 0, 0.0, 0)
    print("E6 resume state: completed_epoch=", completed_epoch,
          "current_epoch=", current_epoch, "next_sample=", next_sample,
          "HF backup interval (batches)=", E6_BACKUP_BATCHES)
    stop = False
    for ep in range(current_epoch, E6_EPOCHS):
        permutation = np.random.default_rng(10000 + ep).permutation(len(train_ds)).tolist()
        offset = next_sample if ep == current_epoch else 0
        running_loss = loss_sum if ep == current_epoch else 0.0
        running_count = loss_count if ep == current_epoch else 0
        subset = Subset(train_ds, permutation[offset:])
        loader = DataLoader(subset, batch_size=E6_BATCH, shuffle=False, num_workers=2, collate_fn=collate,
                            pin_memory=True, generator=torch.Generator().manual_seed(20000 + ep))
        net.train()
        for images, targets in loader:
            images = [image.to("cuda", non_blocking=True) for image in images]
            targets = [{key: val.to("cuda", non_blocking=True) for key, val in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                parts = net(images, targets)
                loss = sum(parts.values())
            if not torch.isfinite(loss):
                raise RuntimeError("E6 loss NaN/Inf")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            running_count += 1
            offset += len(images)
            if running_count % E6_BACKUP_BATCHES == 0 or time.monotonic() >= E6_DEADLINE:
                persist(ep, offset, ep, running_loss, running_count)
            if time.monotonic() >= E6_DEADLINE:
                print("E6 dừng sau batch đã lưu HF; tiếp tục bằng Edit > Save Version > Save & Run All")
                stop = True
                break
        if stop:
            break
        if offset != len(train_ds):
            raise RuntimeError("E6 epoch chưa xử lý hết TRAIN")
        score50, score, recall = evaluate()
        scheduler.step()
        if score > best_map:
            best_map = score
            torch.save(net.state_dict(), E6_WORK / "best_weights.pth")
        metrics_rows.append({"epoch": ep + 1, "map50": score50, "map5095": score,
                             "mar100": recall, "loss_mean": running_loss / running_count})
        completed_epoch = ep + 1
        persist(completed_epoch, 0, completed_epoch, 0.0, 0)
        print("E6 completed epoch", completed_epoch, "/10; VALID mAP50-95", score)
        if time.monotonic() >= E6_DEADLINE:
            break

latest = store.restore()
if latest is None:
    raise RuntimeError("E6 chưa có HF checkpoint sau phiên này")
latest_manifest, latest_files = latest
latest_payload = torch.load(latest_files["checkpoint.pt"], map_location="cpu", weights_only=False)
validate_e6_payload(latest_payload, latest_manifest)
print("E6 status:", latest_manifest["completed_epoch"], "/10",
      "COMPLETE" if latest_manifest["is_complete"] else "CONTINUE",
      "next_sample", latest_manifest["next_sample"])
