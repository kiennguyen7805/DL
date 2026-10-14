"""Vietnamese Gradio interface for E3/E4/E6 ROD image detection."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from app.detector import MODEL_OPTIONS, OUTPUT_DIR, detect, summary


def run_detection(image, model_option: str, *, output_dir: Path = OUTPUT_DIR):
    """Return annotated image, readable summary, and a downloadable PNG."""
    started = time.perf_counter()
    try:
        prediction = detect(image, model_option, output_dir=output_dir)
        message = summary(prediction, model_option)
        download_path = str(prediction.output_path)
        elapsed_ms = (time.perf_counter() - started) * 1000
        message += f"\n\n**Thời gian xử lý trên máy chủ:** {elapsed_ms:.1f} ms"
        return prediction.annotated, message, download_path
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        return None, f"**Không thể xử lý ảnh:** {exc}", None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Phát hiện vật cản ROD") as demo:
        gr.Markdown(
            "# Phát hiện vật cản trong ảnh\n"
            "Tải ảnh lên, chọn mô hình đã fine-tune trên ROD và xem kết quả phát hiện."
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(
                    label="Ảnh đầu vào (JPG, JPEG, PNG)", type="pil",
                    sources=["upload"], height=380,
                )
                model = gr.Dropdown(
                    choices=list(MODEL_OPTIONS), value=MODEL_OPTIONS[0],
                    label="Mô hình",
                )
                submit = gr.Button("Phát hiện vật cản", variant="primary")
            with gr.Column(scale=1):
                annotated = gr.Image(label="Ảnh kết quả", type="pil", height=380)
                report = gr.Markdown("Kết quả sẽ hiển thị tại đây.")
                download = gr.File(label="Tải ảnh kết quả", interactive=False)
        gr.Markdown(
            "Ứng dụng chỉ phát hiện vật thể trên ảnh tải lên. Kết quả không thay thế "
            "công cụ hỗ trợ di chuyển chuyên dụng."
        )
        submit.click(
            fn=run_detection,
            inputs=[source, model],
            outputs=[annotated, report, download],
        )
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(OUTPUT_DIR)],
    )
