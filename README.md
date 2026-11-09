# Phát hiện vật cản trong ảnh hướng tới hỗ trợ di chuyển cho người khiếm thị

Dự án so sánh ba mô hình phát hiện vật cản trên ROD Final A4 và xây dựng ứng dụng **upload ảnh → phát hiện vật cản → hiển thị bounding box, tên lớp và confidence**. Phạm vi hiện tại không gồm webcam, ước lượng khoảng cách, cảnh báo giọng nói, GPS hay database. Demo có thêm bước phân vùng theo vị trí bbox (prototype, chưa kiểm chứng; xem phần ứng dụng).

## Dữ liệu và thí nghiệm

ROD Final A4 tại [`data/processed/rod_detection_final/`](data/processed/rod_detection_final/) đã freeze: 18.151 ảnh TRAIN, 3.318 ảnh VALID, 1.611 ảnh TEST; tổng 23.080 ảnh, 25 lớp. Không chỉnh dữ liệu đã freeze. TEST chỉ dùng để đánh giá cuối cùng sau khi chọn mô hình bằng VALID.

| ID | Mô hình | Trạng thái có bằng chứng |
|---|---|---|
| E3 | YOLO11n COCO Pretrained | 30/30 epoch; checkpoint và đánh giá VALID/TEST đã xác minh local |
| E4 | YOLO11s COCO Pretrained | 30/30 epoch; checkpoint và đánh giá VALID/TEST đã xác minh local |
| E6 | Faster R-CNN ResNet-50 FPN V2 COCO Pretrained | 30/30 epoch; best checkpoint mới đã đánh giá đủ VALID/TEST, ngưỡng 0,60 |

E5 Scratch đã rời phạm vi nghiên cứu; code, báo cáo và cấu hình cũ được giữ để truy vết, không thuộc kế hoạch thực thi.

## Điểm vào hiện hành

**Kết quả cập nhật:** E3/E4/E6 hoàn tất các run 30 epoch, đã đánh giá đủ VALID/TEST. E6 mới nhập từ `results_valid_e6_30.zip`, ngưỡng VALID 0,60. Bảng hiện hành và bằng chứng tại [`results/30epochs/`](results/30epochs/).

- E3: [`notebooks/Kaggle_E3_30epochs_AutoResume.ipynb`](notebooks/Kaggle_E3_30epochs_AutoResume.ipynb)
- E4: [`notebooks/Kaggle_E4_30epochs_AutoResume.ipynb`](notebooks/Kaggle_E4_30epochs_AutoResume.ipynb)
- E6: [`notebooks/Kaggle_E6_30epochs_AutoResume.ipynb`](notebooks/Kaggle_E6_30epochs_AutoResume.ipynb)
- E3/E4 VALID/TEST: [`notebooks/Kaggle_E3_E4_30epochs_Valid_Test.ipynb`](notebooks/Kaggle_E3_E4_30epochs_Valid_Test.ipynb).
- E6 VALID/TEST trên Kaggle: [`Kaggle_E6_30epochs_Valid_Test.ipynb`](notebooks/Kaggle_E6_30epochs_Valid_Test.ipynb).
- Hướng dẫn Kaggle: [`docs/kaggle/HUONG_DAN.md`](docs/kaggle/HUONG_DAN.md)
- Trạng thái và bước tiếp: [`docs/plans/EXPERIMENT_STATUS.md`](docs/plans/EXPERIMENT_STATUS.md), [`docs/plans/MASTER_RESEARCH_ROADMAP.md`](docs/plans/MASTER_RESEARCH_ROADMAP.md)

## Checkpoint và ứng dụng

- E3: [`results/30epochs/E3/checkpoint/best.pt`](results/30epochs/E3/checkpoint/best.pt), SHA256 `d88b24f0…c33786`; ngưỡng VALID 0,27.
- E4: [`results/30epochs/E4/checkpoint/best.pt`](results/30epochs/E4/checkpoint/best.pt), SHA256 `a5c32e2f…ffe1ce`; ngưỡng VALID 0,24.
- E6: [`results/30epochs/E6/checkpoint/best_weights.pth`](results/30epochs/E6/checkpoint/best_weights.pth), SHA256 `a75beb6e3d8629bf9066bc8a03a0b44de93a720aa3d7b3663f0641b732ae8a8f`; run hoàn tất 30/30, ngưỡng VALID 0,60.
- E6 VALID/TEST: ngưỡng **0,60**, VALID mAP50–95 0,660507, TEST mAP50–95 0,677107. Bảng ba mô hình: [`summary_E3_E4_E6.csv`](results/30epochs/summary_E3_E4_E6.csv).
- Gói input Kaggle mới: [`results_e6_30epochs_eval.zip`](results_zips/results_e6_30epochs_eval.zip). Kết quả đánh giá mới nhất: `results_zips/results_valid_e6_30.zip`.
- Các ZIP kết quả nguồn được gom trong [`results_zips/`](results_zips/); kết quả đã nhập cho hệ thống vẫn ở [`results/30epochs/`](results/30epochs/).
- [`app/`](app/) có ứng dụng Gradio upload ảnh với E3, E4 hoặc E6. Việc chọn **một mô hình triển khai cuối** cho báo cáo nghiên cứu vẫn phải dựa trên đánh giá VALID chung và benchmark cùng phần cứng; dropdown demo hiện cho phép xem cả ba.

## Chạy ứng dụng trên Windows

Trên máy đã chuẩn bị dataset và checkpoint, trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -m app.main
```

Mở `http://127.0.0.1:7860`, tải ảnh JPG/JPEG/PNG, chọn mô hình và bấm **Phát hiện vật cản**. Chế độ mặc định dùng ngưỡng đã chọn trên VALID; chọn **Tự chỉnh** để dùng thanh confidence 0,01–0,99. Tùy chọn **High recall — quét đa vùng** phóng lớn các vùng chồng lấn để tìm vật thể nhỏ/xa rồi khử box trùng; chế độ này chậm hơn và có thể tăng báo nhầm. Ứng dụng hiển thị ảnh có bbox, tên lớp, confidence, số lượng, ngưỡng và kiểu quét thực dùng. Nhấn `Ctrl+C` ở PowerShell để dừng.

Ứng dụng đọc trực tiếp checkpoint fine-tuned E3, E4 và E6 ở các đường dẫn trên, đều kiểm SHA256. Adapter E6 dùng Faster R-CNN V2 với 26 lớp gồm background và ánh xạ về ROD 0–24. Ảnh kết quả ở `app/outputs/`.

**Phân vùng theo vị trí (prototype):** sau khi mô hình dự đoán, mỗi bbox được gán vào **Dưới chân / Trước mặt / Ở xa** theo vị trí đáy và tâm ngang của bbox trong ảnh (ngưỡng 70%/40% chiều cao, khai báo tại `ZONE_*` trong `app/detector.py`). Đây là heuristic đặt tay, giả định ảnh chụp từ góc nhìn người đi bộ và **chưa được kiểm chứng**: nó không ước lượng khoảng cách, không xác định vật thể có thực sự cản lối đi, và không thay đổi bbox, confidence hay bất kỳ số liệu VALID/TEST nào của E3/E4/E6. Giao diện có tuỳ chọn tắt tô màu theo vùng (xem đúng đầu ra mô hình) và ẩn phát hiện vùng Ở xa. Kiểm chứng quy tắc này (đánh giá theo vùng, hoặc gán nhãn thủ công “có cản lối đi”) thuộc hướng phát triển.

**Auto Confidence:** phần trăm trên bbox là confidence do mô hình dự đoán; ngưỡng là mức lọc prediction; chế độ tự động đọc ngưỡng riêng theo model từ [cấu hình](app/confidence_config.json). E3/E4/E6 có ngưỡng VALID lần lượt 0,27/0,24/0,60. Ngưỡng E6 được chọn trên VALID của checkpoint hiện hành. Ngưỡng không thay đổi theo từng ảnh và không dùng TEST. Confidence hiển thị không phải xác suất đã hiệu chuẩn.

Cấu hình ngưỡng nằm trong `app/confidence_config.json`; bằng chứng hiệu chuẩn và đánh giá E6 hiện hành ở `results/30epochs/E6/`. `results/E6_Faster_RCNN/confidence/` là bằng chứng của checkpoint E6 cũ.

## Đánh giá thống nhất

Ba mô hình được đánh giá trên cùng VALID bằng `app/evaluator.py`: ghép bbox đúng lớp tại IoU 0,5; AP nội suy 101 điểm tại IoU 0,50–0,95 và tối đa 100 detection/ảnh. Precision/Recall tại ngưỡng vận hành dùng tối đa 300 detection/ảnh, cùng giới hạn lúc chọn ngưỡng bằng VALID. Đây là protocol COCO-style cho ROD, không phải kết quả pycocotools chính thức. AP lấy từ nhiều score threshold. Các lệnh PowerShell:

```powershell
.\.venv\Scripts\python.exe -m app.calibrate_e6
.\.venv\Scripts\python.exe -m app.evaluate_models --model all
.\.venv\Scripts\python.exe -m app.benchmark_models --model all --images 30
.\.venv\Scripts\python.exe -m app.benchmark_models --scope server --model all --images 30
.\.venv\Scripts\python.exe -m app.evaluate_models --freeze
```

Lượt `calibrate_e6` chỉ dùng 3.318 ảnh VALID. Benchmark đo tuần tự batch 1 trên cùng thiết bị; số FPS CPU không đại diện cho Kaggle GPU. Benchmark `--scope server` đo callback Gradio, chưa đo độ trễ mạng hoặc trình duyệt. Các lệnh trên là quy trình đánh giá local; bộ kết quả hiện hành được nhập từ Kaggle, có khóa VALID riêng cho từng mô hình. [evaluation_lock.json](app/evaluation_lock.json) là chỉ mục local của các khóa gốc. Xem [bảng tổng hợp](results/30epochs/summary_E3_E4_E6.csv) và `results/30epochs/{E3,E4,E6}/test/` cho kết quả TEST hiện hành. Báo cáo: [NguyenTranKien.pdf](NguyenTranKien.pdf). Bản Word: [ACL_Paper.docx](ACL_Paper.docx). Slide: [Slide_BaoCao.pptx](Slide_BaoCao.pptx).

## Cấu trúc

`configs/` cấu hình hiện hành; `notebooks/` giữ ba notebook E3/E4/E6; `src/` logic và công cụ còn cần để tái lập; `app/` ứng dụng; `docs/` kế hoạch hiện hành và hướng dẫn; `results/` bằng chứng và kết quả; `artifacts/` gói dữ liệu; `tests/` kiểm thử. ZIP nguồn E3 nằm trong `archive/`. Workflow Scratch, Stage C cũ, Smoke Test và inventory cũ đã được sao lưu ngoài project.

```text
results/
├─ 30epochs/          Kết quả hiện hành E3/E4/E6; GitHub lưu metrics và cache dự đoán
├─ lam_sach_du_lieu/  Báo cáo, ảnh review, manifest V2/V3/V4/Final và audit Day 1/2
├─ E3_YOLO11n/        weights/, metrics/, logs/ và calibration VALID
├─ E4_YOLO11s/        checkpoints/, metrics/, logs/ và calibration VALID
└─ E6_Faster_RCNN/    Artifact E6 cũ, chỉ giữ local để truy vết
```

`results/day2/` đã được đổi tên trực tiếp thành `results/lam_sach_du_lieu/`; các manifest CSV/JSON gốc ghi đường dẫn cũ được giữ nguyên để bảo toàn bằng chứng. Code đọc ảnh review chấp nhận cả đường dẫn Windows/Unix cũ. Snapshot môi trường Day 1 ở `archive/day1_environment/environment.json`, có ZIP backup và manifest SHA256 tại `archive/day1_environment_backup.zip`. Nguồn E3 `results11.zip`, các artifact và tài liệu Smoke Test nằm ở `archive/e3_kaggle_evidence/`; thư mục `results11/` giải nén đã được loại sau khi đối chiếu SHA256.

## Source archive

`src/` currently contains 14 Python files required for the app tests, E3/E4/E6 notebook generation, E3 audit, ROD Final build/verification, and reproducible dataset statistics. The 25 historical ROD V2/V3/V4, review, leakage, and environment scripts were verified and removed from the active source tree. Their original paths and SHA256 hashes are recorded in `../btl_src_history_20261011.zip` (outside the project). Restore a historical script only after checking that ZIP manifest. `src/01_inspect_rod.py` remains because the current dataset quality test calls it.
