"""Embed audited Python sources into the two standalone Kaggle notebooks."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def cell_source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def update(notebook_path, runner_path, is_e4):
    path = ROOT / notebook_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    shared = cell_source("src/kaggle_hf_shared.py")
    runner = cell_source(runner_path)
    if is_e4:
        frozen = cell_source("configs/e4_yolo11s_pretrained.yaml")
        runner = "FROZEN_E4_YAML = " + repr(frozen) + "\n\n" + runner
    expected_names = yaml.safe_load(cell_source("data/processed/rod_detection_final/data.yaml"))["names"]
    dataset_guard = "\nEXPECTED_ROD_NAMES = " + repr(expected_names) + "\n" + '''
if class_names != EXPECTED_ROD_NAMES or source_cfg.get('nc') != 25:
    raise RuntimeError('ROD Final class ID/order không khớp bản freeze')
for split_name, images_dir, expected_count in [('TRAIN', TRAIN_IMAGES, 18151), ('VALID', VAL_IMAGES, 3318)]:
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    labels_dir = images_dir.parent / 'labels'
    if len(image_files) != expected_count or not labels_dir.is_dir():
        raise RuntimeError(f'{split_name} image/label directory count không khớp ROD Final A4')
    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in labels_dir.glob('*.txt')}
    if image_stems != label_stems:
        raise RuntimeError(f'{split_name} image/label không khớp từng file')
print('ROD Final A4 TRAIN/VALID image-label pairs và 25 class: PASS')
'''
    notebook["cells"][3]["source"] = (
        "".join(notebook["cells"][3]["source"]).split("\nEXPECTED_ROD_NAMES = ")[0] + dataset_guard
    ).splitlines(keepends=True)
    notebook["cells"][4]["source"] = shared.splitlines(keepends=True)
    notebook["cells"][5]["source"] = runner.splitlines(keepends=True)
    notebook["cells"][6]["source"] = [
        "## Xác nhận kết quả\n",
        "Mở `last_hf_manifest.json` trong Kaggle Output và `latest.json` trong đúng HF folder. "
        "Chỉ coi COMPLETE khi `completed_epoch=10`, checksum và hậu kiểm checkpoint PASS. "
        "Nếu upload lỗi, lấy `hf_backup/`, checkpoint và log trong Kaggle Output; "
        "phiên sau không được âm thầm bắt đầu lại epoch 1.\n",
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    update("notebooks/Kaggle_E4_YOLO11s_HF_AutoResume.ipynb", "src/kaggle_hf_e4.py", True)
    update("notebooks/Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb", "src/kaggle_hf_e6.py", False)
