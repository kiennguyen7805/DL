# EXPERIMENT STATUS

Ngày đối chiếu: 2026-10-11. Chỉ trạng thái có file/log/checksum mới được ghi `COMPLETED` hoặc `VERIFIED`. E6 có checkpoint local ghi 10 epoch; chưa đối chiếu đầy đủ artifact/manifest Kaggle hoặc HF.

## Định nghĩa

| Trạng thái | Nghĩa |
|---|---|
| `NOT_STARTED` | chưa có implementation/run phù hợp |
| `PREPARED` | code/config/test local đã có; chưa có main run hoàn tất |
| `RUNNING` | một phần main run đã hoàn tất hoặc đang tiếp tục |
| `COMPLETED` | run đạt target nhưng chưa audit đầy đủ artifact |
| `FAILED` | run/gate thất bại; artifact có thể vẫn phục hồi được |
| `VERIFIED` | kết quả/checkpoint đã audit bằng file, log và checksum |

## Dataset và hạ tầng

| Hạng mục | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| ROD Final A4 | `VERIFIED` | 18.151/3.318/1.611 ảnh; 38.467 object; 25 lớp; `results/lam_sach_du_lieu/final/a4_report.md` |
| Kaggle packaging | `VERIFIED` | dataset được Smoke Test nhận đúng; Final local không sửa |
| Stage B Smoke E3 | `VERIFIED` | 1 epoch/5% TRAIN/VALID, reload PASS; không dùng TEST |
| E3 finalization độc lập | `VERIFIED` local | `src/e3_finalize_10epochs.py` và `notebooks/kaggle_rod_e3_finalize_10epochs.ipynb`; artifact riêng, không gọi training/E5 |
| Stage C runner/notebook cũ | lịch sử | flow 30 epoch ghép E3/E5; không dùng cho kế hoạch 10 epoch |

## Thí nghiệm

| ID | Mô hình | Vai trò | Trạng thái | Tiến độ và bằng chứng | Việc để chuyển trạng thái |
|---|---|---|---|---|---|
| E3 | YOLO11n COCO Pretrained | core | `VERIFIED` local | 10/10 epoch thực chạy; P=0,84610, R=0,73533, mAP50=0,81492, mAP50-95=0,60605 trên VALID; artifact/checkpoint audit local PASS | hậu kiểm Save Version nếu dùng Kaggle; đánh giá chung sau E4/E6 |
| E4 | YOLO11s COCO Pretrained | core | `COMPLETED`; checkpoint `VERIFIED` local | 10/10 epoch theo người dùng; `results/E4_YOLO11s/checkpoints/best.pt` nạp được với 25 class; output `results/E4_YOLO11s/checkpoints/` được giữ nguyên | hậu kiểm đầy đủ metrics, config, hashes và artifact rồi chuẩn hóa đánh giá |
| E6 | Faster R-CNN ResNet-50 FPN V2 COCO Pretrained | core | `COMPLETED` theo checkpoint local; artifact chưa `VERIFIED` | `results/E6_Faster_RCNN/checkpoints/checkpoint.zip` ghi `completed_epoch=10`, `next_sample=0`, có optimizer/scheduler; `best_weights.zip` nạp strict vào Faster R-CNN V2 26 lớp PASS, SHA256 đã ghi trong hướng dẫn cấu trúc | bổ sung `metrics.csv`, `config.json`, `environment.json`, `last_hf_manifest.json`; đối chiếu manifest, checksum và metrics trước khi ghi `VERIFIED` |

E5 YOLO11n Scratch đã **loại khỏi phạm vi hiện tại**. Smoke E5 từng PASS, nhưng đây chỉ là lịch sử; không huấn luyện, không đánh giá và không đưa vào bảng so sánh. Giữ nguyên code, YAML, checkpoint và log cũ để truy vết.

## E3 checkpoint Stage 1

| Kiểm tra | Trạng thái | Bằng chứng |
|---|---|---|
| `results.csv` epoch 1–10 liên tục | `VERIFIED` | recovery report/audit |
| `resume_epoch010.pt` load CPU | `VERIFIED` | raw epoch 9, optimizer 255 state, EMA/scaler có |
| `recovery_full.pt` | `VERIFIED` | trùng bit với resume checkpoint |
| `best.pt`, `last.pt` | `VERIFIED` local | nạp được cho inference; đã strip optimizer, không dùng để resume |
| ZIP E3 10 epoch độc lập | `VERIFIED` local | CRC, SHA256, manifest và metric/checkpoint có thể đọc lại; xem báo cáo hậu kiểm mới |
| ZIP E3 partial | `VERIFIED` | 56.690.259 byte, CRC PASS, SHA256 `2f6f72f6...a3fc10` |
| lỗi hậu kiểm `device` | `FAILED` ở run gốc | chuỗi `'0'` so với số `0`, sau checkpoint/validation; không ảnh hưởng trọng số |
| bản sửa `device` | `VERIFIED` local | `0` và `'0'` bằng nhau; `cpu` và `0` vẫn bị báo lệch; regression test |
| resume bắt đầu epoch 11 | không bắt buộc | chưa có log GPU thật; chỉ xem xét khi mở rộng huấn luyện được phê duyệt |

E3 đã được huấn luyện **10 epoch trong lịch gốc 30 epoch** (`args.yaml` và `train_args.epochs=30`). Quyết định ngân sách 10 epoch là quyết định hiện tại, không sửa lịch sử, không tuyên bố scheduler được thiết kế lại cho 10 epoch. Huấn luyện mở rộng chỉ được đề xuất từ bằng chứng VALID và phê duyệt riêng, kèm thay đổi protocol cho cả ba mô hình.

## Đánh giá và ứng dụng hiện hành

| Thành phần | Trạng thái | Gate |
|---|---|---|
| evaluator chung ba model | `NOT_STARTED` | cùng schema/metric, unit tests; dùng VALID để chọn |
| benchmark latency/FPS/VRAM | `NOT_STARTED` | cùng máy/ảnh/batch/precision |
| đánh giá TEST cuối | `NOT_STARTED` | chỉ sau khi chốt checkpoint và lựa chọn bằng VALID |
| upload ảnh → detection → bbox/class/confidence | `COMPLETED` với E3/E4; E6 chưa tích hợp | app Gradio đã kiểm thử local, 24 test app/HF logic PASS trong đợt audit cấu trúc; E6 cần loader Torchvision riêng |
| report/slides | `NOT_STARTED` | số liệu truy vết, không claim chưa đo |

Webcam, khoảng cách, hướng, cảnh báo/TTS, GPS và database đã ra khỏi phạm vi sản phẩm hiện tại; chỉ còn trong tài liệu lịch sử.

## Quyết định tiếp theo

1. Bảo toàn artifact E3; hậu kiểm toàn bộ output E4, không train lại.
2. Hậu kiểm E6 bằng metrics/config/manifest rời để nâng từ `COMPLETED` theo checkpoint lên `VERIFIED`; không train lại.
3. Dựng evaluator/benchmark chung, chọn mô hình bằng VALID; chỉ sau khi khóa quyết định mới đánh giá TEST.
4. Giữ app upload ảnh E3/E4 đã có; tích hợp E6 sau khi có artifact đầy đủ, rồi kiểm thử và chuẩn bị demo.

