"""Vietnamese Gradio interface for E3/E4/E6 ROD image detection."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from app.detector import MODEL_OPTIONS, OUTPUT_DIR, detect, summary


AUTO_MODE = "Tự động theo VALID"
MANUAL_MODE = "Tự chỉnh"


def run_detection(
    image,
    model_option: str,
    threshold_mode: str = AUTO_MODE,
    manual_threshold: float = 0.25,
    high_recall: bool = False,
    filter_far: bool = False,
    zone_colors: bool = True,
    *,
    output_dir: Path = OUTPUT_DIR,
):
    """Return annotated image, readable summary, and a downloadable PNG."""
    started = time.perf_counter()
    try:
        if threshold_mode not in {AUTO_MODE, MANUAL_MODE}:
            raise ValueError("Chế độ ngưỡng không hợp lệ.")
        threshold = manual_threshold if threshold_mode == MANUAL_MODE else None
        prediction = detect(
            image, model_option, output_dir=output_dir,
            threshold=threshold, high_recall=bool(high_recall),
            filter_far=bool(filter_far), assistive_mode=bool(zone_colors),
        )
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
            "Tải ảnh lên, chọn mô hình đã fine-tune trên ROD và xem kết quả phát hiện. "
            "Tuỳ chọn thêm: nhóm phát hiện theo vị trí trong ảnh (**Dưới chân / Trước mặt / Ở xa**)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(
                    label="Ảnh đầu vào (JPG, JPEG, PNG)", type="pil",
                    sources=["upload"], height=380,
                )
                model = gr.Dropdown(
                    choices=[(option, option) for option in MODEL_OPTIONS], value=MODEL_OPTIONS[0],
                    label="Mô hình",
                )
                threshold_mode = gr.Radio(
                    choices=[AUTO_MODE, MANUAL_MODE], value=AUTO_MODE,
                    label="Chế độ confidence",
                )
                manual_threshold = gr.Slider(
                    minimum=0.01, maximum=0.99, value=0.25, step=0.01,
                    label="Ngưỡng confidence tự chỉnh",
                    info="Với vật thể rất nhỏ/xa, thử High recall ở 0,01–0,05; ngưỡng thấp có thể báo nhầm.",
                )
                high_recall = gr.Checkbox(
                    value=True,
                    label="High recall — quét đa vùng",
                    info="Phóng lớn các vùng chồng lấn để tìm vật thể nhỏ/xa; chậm hơn và có thể tăng báo nhầm.",
                )
                with gr.Row():
                    zone_colors = gr.Checkbox(
                        value=True,
                        label="Tô màu theo vùng vị trí",
                        info="Tắt để xem đúng đầu ra mô hình (màu theo lớp).",
                    )
                    filter_far = gr.Checkbox(
                        value=False,
                        label="Ẩn phát hiện vùng Ở XA",
                        info="Chỉ hiển thị vùng Dưới chân và Trước mặt.",
                    )
                gr.Markdown(
                    "Ngưỡng tự động theo VALID: **E3 0,27 · E4 0,24 · E6 0,57**. "
                    "Thanh trượt chỉ có hiệu lực khi chọn **Tự chỉnh**.  "
                )
                submit = gr.Button("Phát hiện vật cản", variant="primary")
            with gr.Column(scale=1):
                annotated = gr.Image(label="Ảnh kết quả", type="pil", height=380)
                report = gr.Markdown("Kết quả sẽ hiển thị tại đây.")
                download = gr.File(label="Tải ảnh kết quả", interactive=False)
        gr.Markdown(
            "### Phân vùng theo vị trí trong ảnh (prototype)\n"
            "- 🔴 **DƯỚI CHÂN:** đáy bbox nằm ở 30% dưới cùng của ảnh, gần giữa theo chiều ngang.\n"
            "- 🟡 **TRƯỚC MẶT:** đáy bbox ở khoảng 40–70% chiều cao, gần giữa; hoặc sát đáy nhưng lệch sang hai bên.\n"
            "- ⚪ **Ở XA:** các phát hiện còn lại (phần trên ảnh hoặc rìa ngoài).\n\n"
            "*Quy tắc phân vùng là heuristic đặt tay, giả định ảnh chụp từ góc nhìn người đi bộ và chưa được "
            "kiểm chứng; nó không ước lượng khoảng cách và không ảnh hưởng tới bbox, confidence hay các chỉ số đánh giá. "
            "Ứng dụng chỉ phát hiện vật thể trên ảnh tải lên, không thay thế công cụ hỗ trợ di chuyển chuyên dụng.*"
        )
        submit.click(
            fn=run_detection,
            inputs=[source, model, threshold_mode, manual_threshold, high_recall, filter_far, zone_colors],
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
