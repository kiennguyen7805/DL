# Đánh giá VALID/TEST cho E6 đã hoàn tất 30 epoch

Notebook: `notebooks/Kaggle_E6_30epochs_Valid_Test.ipynb`.
Code Python tương ứng: `scripts/evaluate_e6_30epochs_valid_test_kaggle.py`.

## Chạy trên Kaggle

1. Import notebook trên vào Kaggle; bật **Accelerator > GPU** (T4 hoặc P100).
2. Gắn dataset **ROD Final** gốc có `data.yaml`, 3.318 ảnh VALID và 1.611 ảnh TEST, kèm labels.
3. Upload `results_e6_30epochs_eval.zip` thành một Kaggle Dataset và gắn vào Input. Đây là gói best weights mới, manifest, cấu hình và log lấy từ `results_e6_282930.zip`. Có thể dùng ZIP gốc thay thế. Notebook hỗ trợ cả ZIP còn nguyên hoặc được Kaggle giải nén.
4. Chọn **Save Version > Save & Run All**. Không cần HF token hoặc tải model pretrained.
5. Tải `/kaggle/working/evaluation_E6_30epochs_complete_results.zip` trong Output sau khi chạy xong.

Nếu Input chứa nhiều bản dữ liệu, đặt `DATASET_ROOT` trong cell code tới thư mục có `data.yaml`. Nếu chứa nhiều ZIP cùng tên, đặt `E6_ZIP` tới đúng file. Nếu hết VRAM, đổi `BATCH_SIZE = 1`. Chạy lại cùng cell trên cùng thư mục output sẽ tiếp tục prediction cache nếu metadata khớp. Muốn tiếp tục ở session mới, cần mang thư mục output cũ trở lại đúng đường dẫn ghi được trước khi chạy. Nếu đổi thiết bị, phiên bản thư viện hoặc batch, dùng thư mục `OUTPUT_ROOT` mới.

Môi trường Kaggle cần bộ PyTorch/torchvision tương thích CUDA, Pillow và PyYAML; notebook dùng thư viện có sẵn, không cài đè PyTorch. Nếu báo CUDA không khả dụng, kiểm tra Accelerator.

## Checkpoint và quy trình

Chỉ chấp nhận manifest `completed_epoch=30`, `is_complete=true`, 25 lớp; best weights phải có SHA256:

```text
a75beb6e3d8629bf9066bc8a03a0b44de93a720aa3d7b3663f0641b732ae8a8f
```

Đây là **best checkpoint sau khi hoàn tất run 30 epoch**, có log mAP50–95 cao nhất tại epoch 30. Không chạy lại huấn luyện. Notebook không đánh giá riêng từng snapshot 28/29/30. Gói mới có best weights cũ tại snapshot 28, manifest 29 và best weights mới tại 30; không có đủ trọng số riêng cho cả ba epoch để làm phép so sánh đó.

Notebook kiểm tra checksum trọng số và fingerprint `data.yaml`, quét đủ VALID, chọn lại ngưỡng tối đa micro-F2 (hòa thì ưu tiên Precision, sau đó ngưỡng cao hơn), ghi `evaluation_lock.json`, rồi mới quét TEST với ngưỡng đã khóa. Không dùng cố định ngưỡng 0,57 của checkpoint cũ. Dùng cùng evaluator COCO-style nội bộ với lần đánh giá trước: AP nội suy 101 điểm, IoU 0,50:0,05:0,95, AP maxDet=100, P/R/F2 maxDet=300, IoU vận hành 0,50.

Kết quả lưu riêng trong `evaluation_E6_30epochs_complete/`, gồm:

- `E6/checkpoint/`: best weights, manifest, log training (nếu input là ZIP).
- `E6/valid/`: predictions, metadata, threshold sweep, metrics và bảng theo lớp.
- `E6/evaluation_lock.json`: ngưỡng cùng hash checkpoint và bằng chứng VALID.
- `E6/test/`: predictions, metadata, metrics và bảng theo lớp gắn hash khóa.
- `summary_E6.csv` và `.json`: hai dòng tổng hợp VALID/TEST.

Sau khi tải kết quả, dùng hai dòng mới này để thay số E6 trong báo cáo, đồng thời cập nhật ngưỡng và bảng theo lớp. Các chỉ số E3/E4 giữ nguyên. Đã nhập kết quả mới từ `results_valid_e6_30.zip`: ngưỡng VALID 0,60, TEST mAP50–95 0,677107; bằng chứng ở `results/30epochs/E6/`.
