"""Read-only checkpoint complexity probe. Writes one new JSON beside this script.

Run from project root with .venv/Scripts/python.exe. No image or TEST access.
FLOPs are requested for a single RGB 640x640 tensor, batch one, CPU.
THOP counts supported operators and may omit dynamic detection/postprocess work.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from app.model_adapters import CHECKPOINTS, load_adapter  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    rows = []
    for option in CHECKPOINTS:
        row = {"model": option, "input": "RGB 1x3x640x640", "device": "cpu"}
        try:
            adapter = load_adapter(option)
            row["parameters"] = sum(p.numel() for p in adapter.model.model.parameters()) if hasattr(adapter.model, "model") else sum(p.numel() for p in adapter.model.parameters())
            row["checkpoint"] = str(CHECKPOINTS[option][0].relative_to(ROOT)).replace('\\', '/')
            if option.startswith("YOLO"):
                from ultralytics.utils.torch_utils import get_flops

                # The public YOLO.info method prints a summary but returns None.
                estimate = float(get_flops(adapter.model.model, imgsz=640))
                row["gflops"] = estimate if estimate > 0 else None
                row["flops_scope"] = "Ultralytics get_flops/THOP graph estimate; excludes preprocessing/NMS"
                if estimate <= 0:
                    row["flops_warning"] = "get_flops returned zero after an internal profiling failure"
            else:
                from thop import profile

                net = adapter.model.cpu().eval()
                tensor = torch.zeros((3, 640, 640), dtype=torch.float32)
                with torch.inference_mode():
                    macs, thop_params = profile(net, inputs=([tensor],), verbose=False)
                row["thop_macs"] = float(macs)
                row["thop_params"] = float(thop_params)
                row["gflops_2mac"] = float(2 * macs / 1e9)
                row["flops_scope"] = "THOP trace approximation; dynamic ROI/NMS operations may be omitted"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    output = Path(__file__).with_name("complexity_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    output.write_text(json.dumps({"rows": rows, "torch": torch.__version__}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("OUTPUT", output, flush=True)


if __name__ == "__main__":
    main()
