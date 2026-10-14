# Phát hiện vật cản trong ảnh hướng tới hỗ trợ di chuyển cho người khiếm thị

Dự án so sánh ba mô hình phát hiện vật cản trên ROD Final A4 và xây dựng ứng dụng **upload ảnh → phát hiện vật cản → hiển thị bounding box, tên lớp và confidence**. Phạm vi hiện tại không gồm webcam, ước lượng khoảng cách, cảnh báo giọng nói, GPS hay database.

## Dữ liệu và thí nghiệm

ROD Final A4 tại [`data/processed/rod_detection_final/`](data/processed/rod_detection_final/) đã freeze: 18.151 ảnh TRAIN, 3.318 ảnh VALID, 1.611 ảnh TEST; tổng 23.080 ảnh, 25 lớp. Không chỉnh dữ liệu đã freeze. TEST chỉ dùng để đánh giá cuối cùng sau khi chọn mô hình bằng VALID.

| ID | Mô hình | Trạng thái có bằng chứng |
|---|---|---|
| E3 | YOLO11n COCO Pretrained | 10/10 epoch; artifact và checkpoint đã kiểm tra local |
| E4 | YOLO11s COCO Pretrained | 10/10 epoch trên Kaggle; checkpoint `best.pt` đã nạp local; hậu kiểm toàn bộ artifact còn tiếp tục |
| E6 | Faster R-CNN ResNet-50 FPN V2 COCO Pretrained | Người dùng ghi nhận 10/10 epoch; best weights local nạp đúng kiến trúc 26 lớp; thiếu log training rời để hậu kiểm độc lập toàn bộ epoch |

E5 Scratch đã rời phạm vi nghiên cứu; code, báo cáo và cấu hình cũ được giữ để truy vết, không thuộc kế hoạch thực thi.

## Điểm vào hiện hành

- E3: [`notebooks/kaggle_rod_e3_finalize_10epochs.ipynb`](notebooks/kaggle_rod_e3_finalize_10epochs.ipynb)
- E4: [`notebooks/Kaggle_E4_YOLO11s_HF_AutoResume.ipynb`](notebooks/Kaggle_E4_YOLO11s_HF_AutoResume.ipynb)
- E6: [`notebooks/Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb`](notebooks/Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb)
- Hướng dẫn Kaggle: [`docs/kaggle/HUONG_DAN.md`](docs/kaggle/HUONG_DAN.md)
- Trạng thái và bước tiếp: [`docs/plans/EXPERIMENT_STATUS.md`](docs/plans/EXPERIMENT_STATUS.md), [`docs/plans/MASTER_RESEARCH_ROADMAP.md`](docs/plans/MASTER_RESEARCH_ROADMAP.md)

## Checkpoint và ứng dụng

- E3: [`results/E3_YOLO11n/weights/best.pt`](results/E3_YOLO11n/weights/best.pt); full recovery ở cùng thư mục. ZIP Kaggle gốc và artifact hậu kiểm ở [`archive/e3_kaggle_evidence/`](archive/e3_kaggle_evidence/).
- E4: [`results/E4_YOLO11s/checkpoints/best.pt`](results/E4_YOLO11s/checkpoints/best.pt); metrics ở [`results/E4_YOLO11s/metrics/results.txt`](results/E4_YOLO11s/metrics/results.txt), cấu hình run ở [`results/E4_YOLO11s/logs/config.txt`](results/E4_YOLO11s/logs/config.txt). Full recovery và `last.zip` giữ trong `checkpoints/`.
- E6: [`results/E6_Faster_RCNN/checkpoints/best_weights.zip`](results/E6_Faster_RCNN/checkpoints/best_weights.zip) và [`checkpoint.zip`](results/E6_Faster_RCNN/checkpoints/checkpoint.zip) là checkpoint PyTorch (đuôi `.zip`), không phải archive cần giải nén. Metrics/config/manifest huấn luyện rời của E6 vẫn thiếu ở local; adapter Torchvision đã nạp được best weights. Ngưỡng vận hành 0,59 đã được chọn bằng F2 trên toàn bộ VALID, với bằng chứng tại `results/E6_Faster_RCNN/confidence/`.
- [`app/`](app/) có ứng dụng Gradio upload ảnh với E3, E4 hoặc E6. Việc chọn **một mô hình triển khai cuối** cho báo cáo nghiên cứu vẫn phải dựa trên đánh giá VALID chung và benchmark cùng phần cứng; dropdown demo hiện cho phép xem cả ba.

## Chạy ứng dụng trên Windows

Trong PowerShell:

```powershell
Set-Location -LiteralPath 'D:\Downloads\kỳ 7\dl\btl'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main
```

Mở `http://127.0.0.1:7860`, tải ảnh JPG/JPEG/PNG, chọn YOLO11n, YOLO11s hoặc Faster R-CNN và bấm **Phát hiện vật cản**. Ứng dụng tự lấy ngưỡng theo mô hình từ [Auto Confidence](app/confidence_config.json); người dùng không phải chỉnh threshold. Ứng dụng hiển thị ảnh có bbox, tên lớp, confidence, số lượng, thời gian suy luận, **thời gian xử lý yêu cầu trên máy chủ** và file PNG để tải. Thời gian xử lý máy chủ gồm kiểm ảnh, suy luận, vẽ bbox, lưu PNG và tạo văn bản; không gồm upload/truyền tải hoặc trình duyệt render. Nhấn `Ctrl+C` ở PowerShell để dừng. Nếu máy không có CUDA, ứng dụng tự dùng CPU.

Ứng dụng đọc trực tiếp checkpoint fine-tuned E3, E4 và E6 ở các đường dẫn trên, đều kiểm SHA256. Adapter E6 dùng Faster R-CNN V2 với 26 lớp gồm background và ánh xạ về ROD 0–24. Ảnh kết quả ở `app/outputs/`. [Báo cáo kiểm thử ứng dụng](docs/reports/APP_TEST_REPORT.md).

**Auto Confidence:** phần trăm trên bbox là confidence do mô hình dự đoán; ngưỡng là mức lọc prediction; chế độ tự động đọc ngưỡng riêng theo model từ [cấu hình](app/confidence_config.json). E3/E4/E6 có ngưỡng vận hành lần lượt 0,22/0,19/0,59, được chọn trên toàn bộ VALID bằng F2. Ngưỡng không thay đổi theo từng ảnh và không dùng TEST. Confidence hiển thị không phải xác suất đã hiệu chuẩn.

Xem [báo cáo Auto Confidence](docs/reports/AUTO_CONFIDENCE_REPORT.md) để biết ngưỡng, Precision/Recall và giới hạn của ảnh cầu thang người dùng nêu.

## Đánh giá thống nhất

Ba mô hình được đánh giá trên cùng VALID bằng `app/evaluator.py`: ghép bbox đúng lớp tại IoU 0,5; AP nội suy 101 điểm tại IoU 0,50–0,95 và tối đa 100 detection/ảnh. Precision/Recall tại ngưỡng vận hành dùng tối đa 300 detection/ảnh, cùng giới hạn lúc chọn ngưỡng bằng VALID. Đây là protocol COCO-style cho ROD, không phải kết quả pycocotools chính thức. AP lấy từ nhiều score threshold. Các lệnh PowerShell:

```powershell
.\.venv\Scripts\python.exe -m app.calibrate_e6
.\.venv\Scripts\python.exe -m app.evaluate_models --model all
.\.venv\Scripts\python.exe -m app.benchmark_models --model all --images 30
.\.venv\Scripts\python.exe -m app.benchmark_models --scope server --model all --images 30
.\.venv\Scripts\python.exe -m app.evaluate_models --freeze
```

Lượt `calibrate_e6` lưu cache sau từng ảnh và tiếp tục được nếu bị ngắt; chỉ dùng 3.318 ảnh VALID. Benchmark đo tuần tự batch 1 trên cùng thiết bị; số FPS CPU không đại diện cho Kaggle GPU. Benchmark `--scope server` đo callback Gradio trên 30 ảnh VALID sau warmup, lưu riêng lượt đầu và thời gian trung bình/p95 tại `results/E*/metrics/server_latency_benchmark.json`. Nó chưa đo độ trễ mạng hoặc trình duyệt. Ba VALID đã PASS và `--freeze` đã tạo [evaluation_lock.json](app/evaluation_lock.json); lệnh này chỉ chạy một lần, lần chạy lại sẽ từ chối ghi đè. Lệnh TEST ở `app/evaluate_test.py` chỉ được chạy sau khi cho phép đánh giá cuối, với `--confirm-locked-test`; **chưa chạy TEST** trong giai đoạn chuẩn bị này. Báo cáo đồ án dạng LaTeX: [BAO_CAO_PHAT_HIEN_VAT_CAN.tex](docs/reports/BAO_CAO_PHAT_HIEN_VAT_CAN.tex).

## Cấu trúc

`configs/` cấu hình hiện hành; `notebooks/` giữ ba notebook E3/E4/E6; `src/` logic và công cụ còn cần để tái lập; `app/` ứng dụng; `docs/` kế hoạch hiện hành, hướng dẫn và báo cáo; `results/` bằng chứng và kết quả; `artifacts/` gói dữ liệu; `tests/` kiểm thử. ZIP nguồn E3 nằm trong `archive/`. Workflow Scratch, Stage C cũ, Smoke Test và inventory cũ đã được sao lưu ngoài project; xem [báo cáo cleanup cuối](docs/reports/FINAL_PROJECT_CLEANUP.md).

```text
results/
├─ lam_sach_du_lieu/  Báo cáo, ảnh review, manifest V2/V3/V4/Final và audit Day 1/2
├─ E3_YOLO11n/        weights/, metrics/, logs/ và calibration VALID
├─ E4_YOLO11s/        checkpoints/, metrics/, logs/ và calibration VALID
└─ E6_Faster_RCNN/    Checkpoint E6; metrics/config/manifest rời còn thiếu
```

`results/day2/` đã được đổi tên trực tiếp thành `results/lam_sach_du_lieu/`; các manifest CSV/JSON gốc ghi đường dẫn cũ được giữ nguyên để bảo toàn bằng chứng. Code đọc ảnh review chấp nhận cả đường dẫn Windows/Unix cũ. Snapshot môi trường Day 1 ở `archive/day1_environment/environment.json`, có ZIP backup và manifest SHA256 tại `archive/day1_environment_backup.zip`. Nguồn E3 `results11.zip`, các artifact và tài liệu Smoke Test nằm ở `archive/e3_kaggle_evidence/`; thư mục `results11/` giải nén đã được loại sau khi đối chiếu SHA256.

## Source archive

`src/` currently contains 14 Python files required for the app tests, E3/E4/E6 notebook generation, E3 audit, ROD Final build/verification, and reproducible dataset statistics. The 25 historical ROD V2/V3/V4, review, leakage, and environment scripts were verified and removed from the active source tree. Their original paths and SHA256 hashes are recorded in `../btl_src_history_20261011.zip` (outside the project). Restore a historical script only after checking that ZIP manifest. `src/01_inspect_rod.py` remains because the current dataset quality test calls it.
