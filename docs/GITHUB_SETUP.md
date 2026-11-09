# Cài đặt từ GitHub

Repo chứa code demo Gradio, notebook huấn luyện/đánh giá và kết quả VALID/TEST
của E3, E4, E6. Dataset và checkpoint được lưu riêng, không có trong Git.

## Cài môi trường

Dùng Python 3.12, chạy ở thư mục gốc repo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu cần GPU, cài bản PyTorch/torchvision phù hợp với CUDA của máy.

## Chuẩn bị dữ liệu và checkpoint

Khôi phục nguyên bộ ROD Final A4 vào `data/processed/rod_detection_final/`,
gồm `data.yaml` và các thư mục `train/`, `valid/`, `test/` với ảnh và nhãn.
Demo cần `data.yaml` để đọc đúng tên và thứ tự 25 lớp; chạy đánh giá cần cả ảnh
và nhãn. Giữ nguyên file YAML gốc của bộ dữ liệu đã khóa khi đối chiếu provenance.

Đặt đúng checkpoint fine-tuned vào các vị trí sau. App kiểm tra SHA256 và từ chối
checkpoint khác, kể cả khi tên file giống nhau:

| Mô hình | Đường dẫn | SHA256 |
|---|---|---|
| E3 | `results/30epochs/E3/checkpoint/best.pt` | `d88b24f020a75464d2df2197b8fefeb16be2ebb735f361d19ba569b0dfc33786` |
| E4 | `results/30epochs/E4/checkpoint/best.pt` | `a5c32e2fc8be4eb59dd80b67d76ac8abfc4a5ab13437814962ea2c463dffe1ce` |
| E6 | `results/30epochs/E6/checkpoint/best_weights.pth` | `a75beb6e3d8629bf9066bc8a03a0b44de93a720aa3d7b3663f0641b732ae8a8f` |

Dataset và weights cần lấy từ artifact gốc của đề tài. Không có cơ chế tải tự
động cả ba checkpoint hiện hành chỉ bằng cách clone repo. Script
`scripts/download_hf_checkpoints.py` phục vụ snapshot E3/E4, không thay thế
việc khôi phục và kiểm tra ba artifact đúng SHA256 ở bảng trên.

## Chạy demo

```powershell
.\.venv\Scripts\python.exe -m app.main
```

Mở `http://127.0.0.1:7860`. Ngưỡng confidence mặc định được chọn trên VALID:
E3 = 0,27; E4 = 0,24; E6 = 0,60.

Nhóm vị trí Dưới chân / Trước mặt / Ở xa là heuristic chưa kiểm chứng. Code
hiện tại chưa triển khai vùng tam giác, chưa đo khoảng cách hoặc xác định
vật thể thật sự chắn lối đi.

## Xem kết quả mà không chạy lại mô hình

- `results/30epochs/summary_E3_E4_E6.csv`: bảng tổng hợp VALID và TEST.
- `results/30epochs/{E3,E4,E6}/{valid,test}/metrics.json`: chỉ số chi tiết.
- `per_class.csv`: kết quả từng lớp; `predictions.jsonl`: cache dự đoán và nhãn.
- `evaluation_lock.json`: khóa VALID gốc của từng mô hình.

Các file đánh giá được giữ nguyên byte để kiểm tra SHA256. Chạy lệnh sau để
đối chiếu báo cáo TEST và provenance, không thực hiện suy luận hoặc huấn luyện:

```powershell
.\.venv\Scripts\python.exe scripts/validate_final_test_results.py
```

Notebook run 30 epoch và đánh giá nằm trong `notebooks/`; hướng dẫn ở
`docs/kaggle/TRAIN_30_EPOCHS.md` và `docs/kaggle/E6_30EPOCHS_VALID_TEST.md`.
Khi cần Hugging Face, dùng biến môi trường hoặc Kaggle Secrets cho `HF_TOKEN`,
không ghi token vào notebook và không commit file `.env`.

## Tài liệu hiện tại

- Báo cáo PDF: `NguyenTranKien.pdf`.
- Bản Word: `ACL_Paper.docx`.
- Slide: `Slide_BaoCao.pptx`, bản PDF `Slide_BaoCao.pdf`.

Các tài liệu là phiên bản người dùng đang lưu; lần đồng bộ GitHub này không
biên tập lại nội dung PDF/Word/slide.
