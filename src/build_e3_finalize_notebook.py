"""Generate the standalone E3 epoch-10 audit notebook from tested source."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ("sha256", "canonical_hash", "normalize_effective_arg",
           "effective_args_mismatches", "safe_extract", "inspect_checkpoint")


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def helper_source():
    text = (ROOT / "src/kaggle_stage_c.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    if {node.name for node in functions} != set(HELPERS):
        raise ValueError("Missing Stage C helper for E3 notebook")
    return "\n".join("".join(lines[node.lineno-1:node.end_lineno]) for node in functions)


def build():
    implementation = (ROOT / "src/e3_finalize_10epochs.py").read_text(encoding="utf-8")
    implementation = implementation.replace(
        "from kaggle_stage_c import (canonical_hash, effective_args_mismatches,\n"
        "                            inspect_checkpoint, safe_extract, sha256)\n", "")
    if "from kaggle_stage_c import" in implementation:
        raise ValueError("E3 notebook still imports external Stage C runner")
    implementation = implementation.split('\nif __name__ == "__main__":\n', 1)[0]
    code = "from __future__ import annotations\nimport os\nimport re\nimport shutil\nimport tempfile\n" \
           "from datetime import datetime, timezone\n" + helper_source() + "\n" + \
           implementation.replace("from __future__ import annotations\n", "", 1)
    cells = [
        cell("markdown", "# E3 YOLO11n COCO Pretrained — epoch 10 finalization\n\n"
             "Read and package the verified epoch-10 result. This notebook does not train, "
             "validate on TEST, or alter ROD Final. The original training schedule was 30 epochs; "
             "the current project budget stops at epoch 10.\n"),
        cell("code", "SOURCE_ZIP = None  # optional exact /kaggle/input/.../results11.zip path\n"
             "OUTPUT = '/kaggle/working/rod_e3_pretrained_epoch010.zip'\n"
             "FROZEN_YAML = None  # optional exact path to e3_pretrained.yaml\n"),
        cell("code", "import importlib.metadata, subprocess, sys\n"
             "try:\n    version = importlib.metadata.version('ultralytics')\n"
             "except importlib.metadata.PackageNotFoundError:\n    version = None\n"
             "if version != '8.4.165':\n"
             "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.4.165'])\n"
             "import torch, yaml\nfrom ultralytics import YOLO\n"
             "print('torch', torch.__version__, 'ultralytics', importlib.metadata.version('ultralytics'))\n"),
        cell("code", code),
        cell("code", "from pathlib import Path\n"
             "input_root = Path('/kaggle/input')\n"
             "matches = [Path(SOURCE_ZIP)] if SOURCE_ZIP else list(input_root.rglob('results11.zip'))\n"
             "if len(matches) != 1 or not matches[0].is_file():\n"
             "    raise FileNotFoundError('Attach exactly one results11.zip or set SOURCE_ZIP')\n"
             "yaml_matches = [Path(FROZEN_YAML)] if FROZEN_YAML else list(input_root.rglob('e3_pretrained.yaml'))\n"
             "if len(yaml_matches) != 1 or not yaml_matches[0].is_file():\n"
             "    raise FileNotFoundError('Attach exactly one e3_pretrained.yaml or set FROZEN_YAML')\n"
             "result = finalize(matches[0], yaml_matches[0], Path(OUTPUT), torch, yaml, YOLO)\n"
             "print({key: value for key, value in result.items() if key != 'manifest'})\n"),
        cell("markdown", "Save a Kaggle Version with outputs, verify the artifact and SHA256 "
             "sidecar in the saved Version, then download a backup. An output under "
             "`/kaggle/working` alone is not durable storage.\n")]
    for i, item in enumerate(cells):
        item["id"] = f"rod-e3-finalize-{i:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks/kaggle_rod_e3_finalize_10epochs.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
