# Auto Confidence cho ứng dụng phát hiện vật cản

Ngày khởi tạo: 2026-10-11; cập nhật E6: 2026-10-12. Phạm vi: hậu xử lý E3/E4/E6 và giao diện Gradio. Không train, không sửa checkpoint/ROD Final, không dùng TEST để chọn ngưỡng.

## Phân biệt ba khái niệm

1. **Confidence dự đoán** là điểm do checkpoint xuất ra cho từng bbox; ứng dụng giữ và hiển thị dưới dạng phần trăm. Đây không phải xác suất đã được hiệu chuẩn.
2. **Threshold** lọc các bbox có confidence thấp hơn một mức cố định. Threshold không làm thay đổi trọng số hoặc điểm confidence của bbox được giữ.
3. **Auto Confidence** tự đọc threshold riêng cho E3/E4/E6 từ [confidence_config.json](../../app/confidence_config.json). Người dùng không chỉnh threshold; ngưỡng không thay đổi theo từng ảnh.

## Nguồn và phép chọn ngưỡng

Mỗi checkpoint fine-tuned chạy **một lượt** trên toàn bộ 3.318 ảnh VALID của ROD Final A4, ở confidence thu thập 0,01, NMS IoU 0,7 và max_det 300. YOLO dùng imgsz 640; E6 dùng quy tắc Torchvision min 640, max 960. Prediction cùng GT YOLO được lưu theo từng mô hình dưới `results/E3_YOLO11n/confidence/`, `results/E4_YOLO11s/confidence/` và `results/E6_Faster_RCNN/confidence/`, kèm metadata và SHA256. Sau đó quét 95 ngưỡng từ 0,01 đến 0,95 trên prediction đã lưu, **không chạy lại mạng**.

Một prediction là TP khi cùng class và khớp một GT chưa dùng ở IoU ≥ 0,5. Prediction thừa/sai class là FP; GT chưa khớp là FN. Chọn micro F2 cao nhất, hòa điểm chọn Precision rồi ngưỡng cao hơn. F2 ưu tiên Recall nhưng vẫn phạt FP. Đây là Precision/Recall tại ngưỡng theo phép đánh giá trên; không thay thế mAP Ultralytics và không phải một kiểm tra độc lập ngoài VALID.

| Model | Threshold chọn từ VALID | Precision | Recall | F2 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| E3 YOLO11n | **0,22** | 0,6455 | 0,7733 | 0,7439 | 4.114 | 2.259 | 1.206 |
| E4 YOLO11s | **0,19** | 0,6502 | 0,8075 | 0,7703 | 4.296 | 2.311 | 1.024 |
| E6 Faster R-CNN | **0,59** | 0,7483 | 0,8453 | 0,8239 | 4.497 | 1.513 | 823 |

Ba model đều dùng 5.320 GT object trong VALID. Ngưỡng/metric và checksum nằm trong [cấu hình](../../app/confidence_config.json), [sweep E3](../../results/E3_YOLO11n/confidence/E3_threshold_sweep.csv), [sweep E4](../../results/E4_YOLO11s/confidence/E4_threshold_sweep.csv) và [sweep E6](../../results/E6_Faster_RCNN/confidence/E6_threshold_sweep.csv). Cache prediction 3.318 ảnh/model đã được kiểm đủ và không trùng ID.

### Khảo sát từng class

Đã khảo sát **cả 25 class** trên cùng prediction VALID ở [bảng E3](../../results/E3_YOLO11n/confidence/E3_per_class_exploration.csv) và [bảng E4](../../results/E4_YOLO11s/confidence/E4_per_class_exploration.csv). Mọi dòng đều ghi applied=NO. Ví dụ Stairs có 336 GT VALID: ngưỡng F2 riêng khám phá là E3 0,15 và E4 0,16. Guard rail có 128 GT: E3 0,34, E4 0,16. Chưa áp dụng class threshold vì cần đánh giá độ ổn định và quy tắc thống nhất trước khi thêm nhiều ngưỡng; ứng dụng hiện dùng một ngưỡng theo model.

## Ảnh cầu thang người dùng nêu

Ảnh đó thuộc TEST và chưa có đường dẫn/file trong workspace nên **chưa thể chạy kiểm tra trực tiếp**. Chỉ suy ra từ điểm do người dùng cung cấp: E3 Stairs 0,168 và Guard rail 0,170 đều thấp hơn ngưỡng VALID 0,22 nên sẽ bị lọc; E4 Stairs 0,919 cao hơn ngưỡng VALID 0,19 nên sẽ hiển thị nếu prediction vẫn như lần thử. Không hạ ngưỡng theo ảnh TEST. Đây là giới hạn quan trọng của E3 và là lý do không được coi thông báo không phát hiện là môi trường an toàn.

## Fallback và giao diện

Nếu một model chưa có calibration hợp lệ, cấu hình khai báo fallback 0,05 và ứng dụng hiện dòng “Tự động (chưa hiệu chuẩn trên VALID)”; giá trị null không được chuyển vào inference. Hiện E3, E4 và E6 đã có kết quả VALID nên fallback **không được sử dụng**. Slider/ô nhập confidence đã bị xóa. Giao diện chỉ cần tải ảnh, chọn model và bấm “Phát hiện vật cản”; kết quả có bbox, nhãn, phần trăm confidence, tổng số vật thể, danh sách văn bản đọc được bằng screen reader và file ảnh tải xuống. Khi không có detection, ứng dụng ghi “Không phát hiện vật cản thuộc các lớp mô hình hỗ trợ.”.

## Kiểm thử local

- tests/test_app.py cùng tests/test_app_confidence.py: **13 PASS, 0 FAIL**. Gồm hai checkpoint fine-tuned, threshold theo model, đầy đủ 3.318 prediction VALID/model và tái lập ngưỡng, F2/ghép một-một, no-detection, nhiều object, confidence hiển thị, NMS, callback và không có Slider.
- Ảnh TRAIN IMG_00091.jpg nhiều vật thể: E3 trả 5 detection, E4 trả 7; đây là kiểm thử hiển thị, không phải kết luận độ chính xác.
- Gradio bản không slider phục vụ localhost HTTP 200, server thử nghiệm đã đóng.
- Không thực hiện GPU benchmark hoặc đánh giá TEST.

## File thay đổi

- app/confidence.py, app/confidence_config.json, app/calibrate_confidence.py
- app/detector.py, app/main.py
- tests/test_app.py, tests/test_app_confidence.py
- README.md, báo cáo này và prediction/sweep/per-class/metadata tại results/E3_YOLO11n/confidence/ and results/E4_YOLO11s/confidence/

**Cập nhật hiện hành:** E6 đã được thêm vào giao diện bằng checkpoint best weights đã fine-tune. Toàn bộ VALID E6 và ngưỡng 0,59 đã hoàn tất; evaluator chung và benchmark nằm trong `results/E6_Faster_RCNN/metrics/`. Lượt regression toàn dự án sau tích hợp: 65 tests và 7 subtests PASS. Các tệp log huấn luyện E6 rời `metrics.csv`, `config.json`, `environment.json`, `last_hf_manifest.json` vẫn thiếu ở local; kết quả VALID mới không thay thế lịch sử training đó. TEST chưa chạy.
