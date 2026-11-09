# Đánh giá E3/E4 sau 30 epoch trên máy local

Để chạy bằng GPU Kaggle, import `notebooks/Kaggle_E3_E4_30epochs_Valid_Test.ipynb`, gắn hai Output E3/E4 và dataset ROD Final A4 gốc vào Input, bật Accelerator GPU và Internet rồi Save & Run All. Notebook có sẵn code, không cần upload repo hoặc HF token. Kết quả ở `evaluation_30epochs_results.zip` trong Output. Internet chỉ cần để cài thư viện; bước inference đọc Input local của Kaggle.

Giữ `results_e3.zip`, `results_e4.zip` trong thư mục `results_zips/` của dự án. Mở PowerShell và chạy:

```powershell
& 'D:\Downloads\kỳ 7\dl\btl\scripts\run_30epoch_valid_test.ps1'
```

Launcher tự tìm Python ở `.venv`, Python portable đã chuẩn bị tại `tmp/python312_runtime/python.exe`, rồi Python trên PATH. Có thể chỉ định môi trường khác:

```powershell
& '.\scripts\run_30epoch_valid_test.ps1' -PythonPath 'C:\path\python.exe' -Device 0 -BatchSize 4
```

Môi trường cần PyTorch, torchvision, ultralytics **8.4.165**, numpy, Pillow và PyYAML. PyTorch hiện có trong `.venv/Lib/site-packages` là bản **CPU**; launcher vẫn chạy được bằng CPU. Muốn dùng RTX 3050 cần Python có PyTorch CUDA; chỉ `-Device 0` không tự cài CUDA. Không tải model hoặc gọi Hugging Face trong bước đánh giá.

Chạy từng model hoặc giảm batch nếu thiếu RAM/VRAM:

```powershell
& '.\scripts\run_30epoch_valid_test.ps1' -Models E3 -BatchSize 2
& '.\scripts\run_30epoch_valid_test.ps1' -Models E4 -BatchSize 2
```

Quy trình: kiểm tra manifest COMPLETE 30/30 và checksum artifact trong ZIP, lấy best checkpoint đúng snapshot (hỗ trợ alias của compact upload), kiểm tra kiến trúc và thứ tự 25 lớp; quét **3318 ảnh VALID**, chọn confidence tối đa micro-F2 trên VALID, lưu evaluation lock; sau đó quét **1611 ảnh TEST** bằng ngưỡng đã khóa. Các model được yêu cầu đều hoàn tất VALID trước khi bắt đầu TEST. Log có số ảnh, phần trăm, elapsed và ETA.

Kết quả ở `results/30epochs/E3/` và `results/30epochs/E4/`:

- `checkpoint/`: checkpoint, manifest, audit checksum và metric training.
- `valid/`: predictions, metadata, threshold sweep, metrics và per-class CSV.
- `evaluation_lock.json`: checkpoint, ngưỡng và bằng chứng VALID được khóa trước TEST.
- `test/`: predictions, metadata, metrics và per-class CSV.
- `results/30epochs/summary_E3_E4.csv` và `.json`: bảng tổng hợp sau khi chạy cả hai; chạy riêng có tên tương ứng `summary_E3` hoặc `summary_E4`.

Metric dùng evaluator chung của dự án: AP nội suy 101 điểm, IoU .50:.05:.95, tối đa 100 detection/ảnh cho AP; P/R/F2 tại ngưỡng đã chọn với tối đa 300 detection/ảnh. Đây là metric chung của dự án, có thể khác số Ultralytics in lúc train.

Nếu bị gián đoạn, chạy lại đúng lệnh; script tiếp tục prediction cache đã lưu và kiểm tra checkpoint/config trước khi dùng cache. Nếu thay checkpoint, phiên bản thư viện hoặc thiết bị, dùng thư mục output khác qua CLI Python thay vì trộn cache. File kết quả 10 epoch và cấu hình app hiện tại không bị ghi đè.
