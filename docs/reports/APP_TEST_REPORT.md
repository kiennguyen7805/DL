# Kiểm thử ứng dụng upload ảnh ROD

Đây là báo cáo của phiên ứng dụng ban đầu, giữ lại để truy vết. Phiên hiện hành đã bỏ slider, tích hợp cả E6 và dùng [Auto Confidence](AUTO_CONFIDENCE_REPORT.md); số kiểm thử mới nhất xem phần cập nhật cuối tài liệu.

Ngày: 2026-10-11. Môi trường local Windows, Python 3.12, Gradio 6.30.0, Ultralytics 8.4.165. Chỉ dùng ảnh TRAIN IMG_00001.jpg và ảnh trắng tổng hợp. Không train, không dùng TEST.

## Checkpoint và class

| Mô hình | Nguồn fine-tuned | SHA256 | Load | Class |
|---|---|---|---|---:|
| E3 YOLO11n | results/E3_YOLO11n/weights/best.pt | 41f7943cbab67d1d5920e2852068b1e98d22eda8023f978abcb608a65a2aed02 | PASS | 25 |
| E4 YOLO11s | results/E4_YOLO11s/checkpoints/best.pt | dc09c71052027e9083f014b99144edb65bcdc2ed032ab3591fce0db52b26e2f5 | PASS | 25 |

Ứng dụng đối chiếu ID 0–24 và thứ tự tên class checkpoint với data/processed/rod_detection_final/data.yaml. E3 được giải nén vào cache sau khi SHA256 đúng; bản ZIP gốc không đổi. E4 đọc trực tiếp file .pt. Model được cache theo lựa chọn trong một process.

## Kiểm thử

| Hạng mục | Kết quả | Bằng chứng |
|---|---|---|
| pytest -q tests/test_app.py | **4 PASS, 0 FAIL** | load hai checkpoint, class, CPU inference, bbox/confidence, PNG, callback, ảnh trắng không detection, lỗi đầu vào |
| Inference E3 ảnh TRAIN | PASS | 1 detection, bbox/confidence hợp lệ, PNG 413.160 byte |
| Inference E4 ảnh TRAIN | PASS | 1 detection, bbox/confidence hợp lệ, PNG 413.097 byte |
| CPU fallback | PASS | test ép CUDA unavailable; cả hai chạy trên CPU |
| Gradio UI | PASS | build_app() tạo Blocks; localhost port 7867 trả HTTP 200, sau đó server được đóng |
| Dependency | PASS | pip check: No broken requirements found |
| Ảnh không có detection | PASS | ảnh trắng tổng hợp ở ngưỡng 0,99, trả thông báo rõ ràng |
| Thiếu ảnh/ngưỡng sai/model sai | PASS | callback trả lỗi tiếng Việt, không trả file giả |
| GPU và E6 | NOT VERIFIED | không cần cho app hiện tại; không chạy GPU |

Thời gian trong lần thử thủ công trên một ảnh TRAIN: E3 1.782 giây và E4 0.349 giây cho lệnh predict; đây là lần chạy đơn lẻ gồm hiệu ứng khởi động, **không phải benchmark so sánh mô hình**. Ứng dụng hiển thị thời gian thực tế của từng yêu cầu.

## File tạo/sửa

- app/__init__.py, app/detector.py, app/main.py
- tests/test_app.py
- requirements.txt, .gitignore, README.md
- docs/reports/APP_TEST_REPORT.md

## Giới hạn còn lại

- Tại thời điểm kiểm thử ban đầu, E6 chưa nằm trong dropdown; trạng thái hiện hành được cập nhật dưới đây.
- Chưa chọn một checkpoint triển khai cuối cùng bằng evaluator chung trên VALID; dropdown phục vụ demo hai mô hình đã có.
- Ảnh kết quả lưu ở app/outputs/ để Gradio cung cấp file tải xuống. Ứng dụng hiện đọc trực tiếp E3 best.pt ở `results/E3_YOLO11n/weights/` và không cần cache checkpoint.
- Inference CPU đã kiểm; tốc độ trên máy khác cần đo lại. Chưa chạy một phiên tương tác trình duyệt của người dùng cuối.

## Cập nhật hiện hành 2026-10-12

E6 Faster R-CNN ResNet-50 FPN V2 dùng checkpoint fine-tuned `results/E6_Faster_RCNN/checkpoints/best_weights.zip` (SHA256 `ce459c56afc57efec80c63ab213bab60ecb9a966c78a1396051de6524783f9bd`). Model nạp bằng Torchvision với head 26 lớp; background 0 không đi vào kết quả ROD 0–24. Giao diện hiện cho chọn E3/E4/E6, không có slider, xuất ảnh PNG và danh sách văn bản. Kiểm thử local toàn dự án: **65 tests, 7 subtests PASS; 0 FAIL**. E6 đã hoàn tất 3.318 ảnh VALID và dùng ngưỡng 0,59 theo F2; không chạy TEST. Benchmark cùng CPU và 30 ảnh VALID: E3 76,26 ms, E4 185,83 ms, E6 2.511,30 ms trung bình mỗi ảnh; xem từng JSON trong `results/E3_YOLO11n/metrics/`, `results/E4_YOLO11s/metrics/`, `results/E6_Faster_RCNN/metrics/`. Chưa kiểm thử GUI tương tác thủ công bằng trình duyệt ở phiên này hoặc GPU deployment.

**Đo bổ sung độ trễ ứng dụng:** callback Gradio hiện hiển thị thời gian xử lý trên máy chủ bên cạnh thời gian suy luận. Với cùng 30 ảnh VALID, sau 3 lượt warmup, trung bình/p95 lần lượt là E3 **134,1/209,8 ms**, E4 **232,6/301,2 ms**, E6 **2.535,2/2.692,2 ms** trên CPU local. Lượt đầu nạp model được ghi riêng: E3 975,4 ms, E4 653,9 ms, E6 5.061,3 ms. Bằng chứng và danh sách ảnh nằm trong `server_latency_benchmark.json` của từng thư mục metrics. Thời gian này gồm callback xử lý và lưu PNG, chưa gồm mạng, hàng đợi và render trình duyệt; không gọi là độ trễ đầu-cuối của người dùng từ xa.
