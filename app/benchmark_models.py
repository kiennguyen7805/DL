"""Same-device, batch-one VALID inference benchmark for E3/E4/E6."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torchvision
import ultralytics
from PIL import Image

from app.confidence import CONFIG_PATH, auto_confidence
from app.evaluate_models import FOLDERS, IDS, ROOT
from app.main import run_detection
from app.model_adapters import CHECKPOINTS, DATA_YAML, load_adapter, sha256_file


def benchmark(model_id: str, *, sample_count: int = 30, warmup: int = 3) -> dict:
    if model_id not in IDS or not 1 <= sample_count <= 3318 or warmup < 0:
        raise ValueError("Invalid benchmark configuration")
    paths = sorted((ROOT / "data/processed/rod_detection_final/valid/images").glob("IMG_*"))
    if len(paths) != 3318:
        raise RuntimeError("VALID split incomplete")
    indices = [round(i * (len(paths) - 1) / (sample_count - 1)) for i in range(sample_count)] if sample_count > 1 else [0]
    selected = [paths[i] for i in indices]
    option = IDS[model_id]
    adapter = load_adapter(option)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for path in selected[:warmup]:
        with Image.open(path) as source:
            adapter.predict(source.convert("RGB"), 0.01)
    timings = []
    for path in selected:
        with Image.open(path) as source:
            prediction = adapter.predict(source.convert("RGB"), 0.01)
        if prediction.device != device:
            raise RuntimeError("Model switched device during benchmark")
        timings.append(prediction.inference_time_ms)
    ordered = sorted(timings)
    result = {
        "model": model_id, "split": "valid", "images": len(selected),
        "image_names": [path.name for path in selected],
        "device": device, "batch": 1, "warmup_images": min(warmup, len(selected)),
        "latency_mean_ms": statistics.mean(timings),
        "latency_median_ms": statistics.median(timings),
        "latency_p95_ms": ordered[min(len(ordered)-1, int(0.95 * len(ordered)))],
        "serial_fps": 1000 / statistics.mean(timings),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None),
        "parameters": adapter.parameter_count,
        "checkpoint_bytes": CHECKPOINTS[option][0].stat().st_size,
        "checkpoint_sha256": sha256_file(CHECKPOINTS[option][0]),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "cpu": platform.processor() if device == "cpu" else None,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "timing_scope": "preprocessing, forward and NMS/postprocessing; excludes file I/O and model load",
    }
    output = ROOT / "results" / FOLDERS[model_id] / "metrics" / "common_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def benchmark_server(model_id: str, *, sample_count: int = 30, warmup: int = 3) -> dict:
    """Measure the Gradio callback on loaded PIL inputs, including PNG output.

    This is server processing time, not browser upload/network/render latency.
    """
    if model_id not in IDS or not 1 <= sample_count <= 3318 or warmup < 0:
        raise ValueError("Invalid benchmark configuration")
    paths = sorted((ROOT / "data/processed/rod_detection_final/valid/images").glob("IMG_*"))
    if len(paths) != 3318:
        raise RuntimeError("VALID split incomplete")
    indices = [round(i * (len(paths) - 1) / (sample_count - 1)) for i in range(sample_count)] if sample_count > 1 else [0]
    selected = [paths[i] for i in indices]
    option = IDS[model_id]

    def call(path: Path, output_dir: Path) -> float:
        with Image.open(path) as source:
            started = time.perf_counter()
            annotated, message, downloadable = run_detection(source, option, output_dir=output_dir)
            elapsed_ms = (time.perf_counter() - started) * 1000
        if not isinstance(annotated, Image.Image) or not downloadable or not Path(downloadable).is_file():
            raise RuntimeError(f"Gradio callback failed for {model_id}: {message}")
        if "Thời gian xử lý trên máy chủ" not in message:
            raise RuntimeError("Callback did not report server processing time")
        return elapsed_ms

    with tempfile.TemporaryDirectory(prefix=f"rod_{model_id.lower()}_latency_") as temporary:
        output_dir = Path(temporary)
        first_request_ms = call(selected[0], output_dir)
        for path in selected[:warmup]:
            call(path, output_dir)
        timings = [call(path, output_dir) for path in selected]
    ordered = sorted(timings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setting = auto_confidence(option)
    result = {
        "model": model_id, "split": "valid", "images": len(selected),
        "image_names": [path.name for path in selected],
        "device": device, "batch": 1, "warmup_requests": warmup,
        "first_request_ms": first_request_ms,
        "latency_mean_ms": statistics.mean(timings),
        "latency_median_ms": statistics.median(timings),
        "latency_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "confidence_threshold": setting.threshold,
        "checkpoint_sha256": sha256_file(CHECKPOINTS[option][0]),
        "confidence_config_sha256": sha256_file(CONFIG_PATH),
        "data_yaml_sha256": sha256_file(DATA_YAML),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "cpu": platform.processor() if device == "cpu" else None,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "timing_scope": "Gradio Python callback: input validation, model loading if cold, inference, drawing, PNG save, summary; excludes browser upload, network transfer and frontend rendering",
        "note": "First request is reported separately; mean/p95 use warm model and operational confidence threshold",
    }
    output = ROOT / "results" / FOLDERS[model_id] / "metrics" / "server_latency_benchmark.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*IDS, "all"], default="all")
    parser.add_argument("--images", type=int, default=30)
    parser.add_argument("--scope", choices=["model", "server"], default="model")
    args = parser.parse_args()
    if args.model == "all" and torch.cuda.is_available():
        parser.error("GPU VRAM benchmark: run each --model in a separate process")
    for model_id in (IDS if args.model == "all" else (args.model,)):
        measure = benchmark_server if args.scope == "server" else benchmark
        result = measure(model_id, sample_count=args.images)
        if args.scope == "server":
            print(model_id, result["device"], "server mean", result["latency_mean_ms"],
                  "ms", "p95", result["latency_p95_ms"], "ms")
        else:
            print(model_id, result["device"], result["latency_mean_ms"], "ms", result["serial_fps"], "FPS")


if __name__ == "__main__":
    main()
