"""Build a standalone Kaggle GPU notebook using the local evaluation pipeline."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundled_sources():
    paths = ["src/kaggle_hf_shared.py", "scripts/download_hf_checkpoints.py",
             "scripts/evaluate_30epoch_outputs.py", "app/evaluator.py"]
    sources = {path: (ROOT / path).read_text(encoding="utf-8") for path in paths}
    # Calibration math is shared exactly with the app; the Kaggle runner does not
    # need the app's local model registry, UI, or historical ten-epoch checkpoints.
    text = (ROOT / "app/calibrate_confidence.py").read_text(encoding="utf-8")
    names = {"Score", "ground_truth", "intersection_over_union", "score_predictions", "choose_threshold"}
    helpers = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            helpers.append(("@dataclass(frozen=True)\n" if node.name == "Score" else "") + ast.get_source_segment(text, node))
    sources["app/calibrate_confidence.py"] = (
        "from __future__ import annotations\nfrom dataclasses import dataclass\nfrom pathlib import Path\n"
        "MATCH_IOU = 0.5\nTHRESHOLDS = [round(i / 100, 2) for i in range(1, 96)]\n\n" + "\n\n".join(helpers))
    for package in ("src", "app", "scripts"):
        sources[package + "/__init__.py"] = ""
    return sources


DISCOVER = '''from pathlib import Path
import json
import zipfile

INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working/evaluation_30epochs')
# Optional: fill these when there are multiple matching inputs.
E3_ZIP = None
E4_ZIP = None
DATASET_ROOT = None
BATCH_SIZE = 4  # Reduce to 2 if GPU memory runs out.

def find_output(experiment, explicit):
    if explicit:
        return Path(explicit)
    zipped = list(INPUT.rglob('results_' + experiment.lower() + '.zip'))
    if len(zipped) == 1:
        return zipped[0]
    if len(zipped) > 1:
        raise RuntimeError('Multiple ZIPs found; set ' + experiment + '_ZIP explicitly')
    # Kaggle often exposes uploaded ZIP contents as already extracted files.
    expected_model = 'yolo11n.pt' if experiment == 'E3' else 'yolo11s.pt'
    matches = []
    for path in INPUT.rglob('last_hf_manifest.json'):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('model') == expected_model and manifest.get('completed_epoch') == 30:
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError('Attach the completed ' + experiment + ' Output ZIP as a Kaggle input dataset; found ' + str(len(matches)) + ' matching extracted outputs')
    path, manifest = matches[0]
    run = path.parent
    snapshot = manifest['snapshot']
    # Repack only this snapshot; no older checkpoints or HF cache needed.
    from scripts.download_hf_checkpoints import safe_filename
    folder = 'YOLO 11n' if experiment == 'E3' else 'YOLO 11s'
    prefix = folder + '/snapshots/'
    if not snapshot.startswith(prefix):
        raise RuntimeError('Invalid snapshot pointer')
    safe_filename(snapshot[len(prefix):])
    staged = Path('/kaggle/working/input_staging')
    staged.mkdir(parents=True, exist_ok=True)
    target = staged / ('results_' + experiment.lower() + '.zip')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.write(path, 'run/last_hf_manifest.json')
        for name in manifest['sha256']:
            safe_filename(name)
            suffix = snapshot + '/' + name
            candidates = [p for p in run.rglob(name) if p.as_posix().endswith('/' + suffix)]
            if len(candidates) != 1:
                raise RuntimeError('Missing or ambiguous snapshot artifact: ' + suffix)
            archive.write(candidates[0], 'run/hf_backup/download/' + suffix)
    return target

if DATASET_ROOT:
    dataset = Path(DATASET_ROOT)
else:
    datasets = [p.parent for p in INPUT.rglob('data.yaml')
                if (p.parent / 'valid/images').is_dir() and (p.parent / 'test/images').is_dir()]
    if len(datasets) != 1:
        raise RuntimeError('Attach the original ROD Final A4 dataset with data.yaml, valid/images, valid/labels, test/images, test/labels. If multiple matches exist, set DATASET_ROOT.')
    dataset = datasets[0]
e3_zip, e4_zip = find_output('E3', E3_ZIP), find_output('E4', E4_ZIP)
print('E3:', e3_zip, '\\nE4:', e4_zip, '\\nDataset:', dataset, '\\nOutput:', OUTPUT, flush=True)
'''


def build():
    def code(source, ident):
        compile(source, ident, "exec")
        return {"cell_type": "code", "id": ident, "metadata": {},
                "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)}

    sources = bundled_sources()
    setup = '''import os, sys, subprocess
os.environ['YOLO_AUTOINSTALL'] = 'false'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                       'ultralytics==8.4.165', 'ultralytics-thop'])
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Enable Kaggle Settings > Accelerator > GPU before running')
print('GPU:', torch.cuda.get_device_name(0), '; torch:', torch.__version__, flush=True)
'''
    bundle = ("from pathlib import Path\nimport sys\n"
              "CODE_ROOT = Path('/kaggle/working/rod_eval_code')\n"
              "SOURCES = " + repr(sources) + "\n"
              "for name, content in SOURCES.items():\n"
              "    path = CODE_ROOT / name\n"
              "    path.parent.mkdir(parents=True, exist_ok=True)\n"
              "    path.write_text(content, encoding='utf-8')\n"
              "sys.path.insert(0, str(CODE_ROOT))\n"
              "print('Standalone evaluation code ready', flush=True)\n")
    run = '''from scripts.evaluate_30epoch_outputs import main
main(['--e3-zip', str(e3_zip), '--e4-zip', str(e4_zip), '--dataset', str(dataset),
      '--output', str(OUTPUT), '--batch', str(BATCH_SIZE), '--device', '0'])
'''
    pack = '''import shutil
archive = shutil.make_archive('/kaggle/working/evaluation_30epochs_results', 'zip', OUTPUT)
print('Download:', archive, flush=True)
print('Summary:', OUTPUT / 'summary_E3_E4.csv', flush=True)
'''
    note = ("# E3/E4 — đánh giá VALID và TEST sau 30 epoch trên Kaggle GPU\n"
            "Gắn Input: Output E3 (results_e3.zip), Output E4 (results_e4.zip) và dataset ROD Final A4 gốc dùng khi train. "
            "Hỗ trợ cả ZIP nguyên và Output đã được Kaggle giải nén. Bật GPU và Internet để cài Ultralytics. Không cần HF token.\n\n"
            "Chạy Save Version > Save & Run All. Notebook kiểm tra checkpoint 30/30 và SHA256, chọn ngưỡng trên 3318 ảnh VALID, "
            "khóa cấu hình rồi đánh giá 1611 ảnh TEST. Không train lại. Có log tiến độ và cache để chạy lại trong cùng runtime; "
            "cache trong /kaggle/working không tự tồn tại qua phiên mới.\n\n"
            "Kết quả: /kaggle/working/evaluation_30epochs/summary_E3_E4.csv và ZIP evaluation_30epochs_results.zip. "
            "Dataset phải giữ data.yaml gốc để kiểm SHA256; đường dẫn Windows trong YAML không ảnh hưởng vì runner đọc trực tiếp thư mục split.\n")
    book = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": [{"cell_type": "markdown", "id": "instructions", "metadata": {}, "source": note.splitlines(keepends=True)},
                      code(setup, "setup"), code(bundle, "bundle"), code(DISCOVER, "inputs"),
                      code(run, "evaluate"), code(pack, "download-results")]}
    path = ROOT / "notebooks/Kaggle_E3_E4_30epochs_Valid_Test.ipynb"
    path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("BUILT:", path)
    return path


if __name__ == "__main__":
    build()
