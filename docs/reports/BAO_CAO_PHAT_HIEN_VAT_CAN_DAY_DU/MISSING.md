# Bằng chứng và phép đo còn thiếu

Danh sách này không phải lệnh chạy TEST. Phần TEST chỉ được thực hiện khi giai đoạn đánh giá cuối được cho phép riêng.

| ID | Thiếu gì | Vì sao cần | Cách bổ sung, không dùng TEST để chọn cấu hình |
|---|---|---|---|
| M01 | `results/E6_Faster_RCNN/metrics.csv`, `config.json`, `environment.json`, `last_hf_manifest.json` hoặc snapshot HF/Kaggle tương ứng có checksum. | Xác minh độc lập epoch, best epoch, LR, augmentation, GPU, framework và đường learning curve E6; hiện chỉ có code, checkpoint và VALID evaluator. | Tải đúng artifact run E6, kiểm SHA256/manifest và đối chiếu checkpoint; không tạo log giả. |
| M02 | Kết quả TEST cuối của cả ba mô hình. | Cần ước lượng chất lượng trên split chưa dùng chọn checkpoint/ngưỡng. | Sau khi khóa checkpoint/protocol theo VALID, chạy đúng một lượt evaluator chung bằng cổng `app/evaluate_test.py`; hiện **chưa chạy**. |
| M03 | GPU VRAM, độ trễ/FPS trên thiết bị mục tiêu và độ trễ end-to-end của giao diện. | Benchmark hiện có là CPU cùng 30 ảnh, không bao gồm mạng/browser và VRAM là null. | Đo ba model trên cùng thiết bị, batch/input và quy trình; tách warm/cold, p95, VRAM, thời gian upload/render nếu liên quan. |
| M04 | Thời gian huấn luyện và cấu hình môi trường so sánh được cho cả ba run. | Log E3/E4 có `time`, E6 thiếu metrics/environment rời; khác GPU/phiên có thể làm so sánh thời gian sai. | Khôi phục log thực tế, chuẩn hóa định nghĩa thời gian và ghi thiết bị mỗi run. |
| M05 | Phân tích nguyên nhân lỗi theo ảnh, kích thước vật, che khuất, điều kiện sáng và từng lớp; khoảng tin cậy/lặp seed. | AP/lớp và FN count không tự chứng minh nguyên nhân hoặc độ ổn định thống kê. | Dùng prediction/ảnh VALID có kiểm soát, kiểm tra nhãn gốc; thiết kế tập điều kiện sáng riêng trước khi nói về thiếu sáng. |
| M06 | FLOPs E6 cùng quy tắc đếm chính xác với YOLO. | THOP trace của detector hai giai đoạn có ROI/NMS động và phép quy đổi 2 FLOPs/MAC là xấp xỉ. | Nếu cần so FLOPs định lượng, lập protocol profiler chung và công bố ops bị loại; hiện dùng latency thực đo để bàn về chi phí triển khai. |
| M07 | Đối chiếu toàn bộ nhãn Final và near-duplicate còn nghi vấn bằng con người. | Freeze kỹ thuật không đồng nghĩa mọi annotation chính xác ngữ nghĩa; báo cáo cũ vẫn ghi nhãn chờ xử lý. | Chỉ báo là giới hạn dữ liệu trong bản thảo; nếu sửa phải tạo phiên bản mới và quy trình đánh giá mới, không sửa ngầm Final. |

## Công cụ của phiên điều tra

Đo complexity bằng `.venv/Scripts/python.exe` trên CPU; Python hệ thống thiếu `torch`. Script và hai JSON đầu ra ở `figures/scripts/`. Lượt đầu trả FLOPs YOLO `null` do gọi `.info()` không trả giá trị; bản chạy sau dùng `get_flops` và cho số hợp lệ. Không chạy training, inference trên TEST hoặc benchmark GPU.
