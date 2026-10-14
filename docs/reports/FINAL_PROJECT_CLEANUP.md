# Dọn cấu trúc dự án lần cuối — 2026-10-11

## Kết quả

Phạm vi hiện hành gồm E3 YOLO11n, E4 YOLO11s, E6 Faster R-CNN và ứng dụng upload ảnh E3/E4. Không có Git repository tại thư mục này (`git status`: `not a git repository`). Không sửa ROD Final A4, notebook E3/E4/E6 chính thức, trọng số, class mapping, thuật toán inference hoặc confidence calibration. Không train và không đánh giá TEST.

| Hạng mục | Kết quả |
|---|---|
| File được xét trong phạm vi metadata/code/tài liệu/kết quả, bỏ qua `.venv`, cache và ảnh/label dataset | 2.218 trước dọn; **2.186 sau dọn** |
| File lịch sử/bản trùng loại khỏi cây làm việc sau khi có backup | **31** (30 trong backup chính, 1 script inventory trong backup riêng) |
| File tạm/placeholder được bỏ thêm | helper backup dùng một lần, 4 `.gitkeep`; cache Python/pytest tái tạo được |
| Dung lượng cây làm việc giảm trong cùng phạm vi | **Khoảng 70,9 MB** (đo trong cùng phạm vi trước/sau) |
| ZIP backup ngoài project | `D:\Downloads\kỳ 7\dl\btl_historical_backup_20261011_v2.zip` — 30 file, 69.056.423 byte, SHA256 `cab8cf99a4bfc5f229bf3acd89c25ec45de54c518b4f3b30398356a5f2d5a8ac`; CRC/manifest và SHA256 từng file PASS |
| Backup script inventory cũ | `D:\Downloads\kỳ 7\dl\btl_inventory_script_backup_20261011.zip` — 1 file, SHA256 `e803fd99c60a3be873a16ccfc5d99e1e28bd3cdb078350a976b4c591fb8440b0`; CRC/manifest PASS |

Các file Scratch/Stage C cũ rời cây làm việc: `configs/e5_scratch.yaml`, notebook E3/E5, notebook Smoke Test, builder/reporter Stage C, script recovery gắn E5, test Stage B/Stage C lịch sử. Notebook E4 ARMED cũ và builder/test cũ cũng được đưa vào backup; notebook E4 HF AutoResume chính thức vẫn ở `notebooks/`. Kế hoạch 5 ngày, prompt và hướng dẫn Kaggle cũ, inventory CSV nhiều nghìn dòng và báo cáo cleanup cũ đã được backup rồi loại khỏi cây hiện hành. `src/kaggle_stage_c.py` **được giữ** vì `src/e3_finalize_10epochs.py` và builder notebook E3 còn dùng hàm của nó; các công cụ dựng lại ROD Final cũng được giữ để bảo toàn khả năng tái lập dataset.

Một bản kế hoạch chi tiết cũ 04 được giữ ở `archive/plans/04_KE_HOACH_CHI_TIET_TUNG_NGAY.md` (SHA256 `cb0f883f2f6ba66191668d6c5b2cd88d87bfea1bb2d7ed58675ffdd5bb6c5644`) để ba liên kết lịch sử trong `results/lam_sach_du_lieu/audit_day1_day2.md` tiếp tục hoạt động. Bản này cũng nằm trong ZIP backup ngoài project.

`results11/` và `results/E3_YOLO11n/kaggle_stage_b/`, `kaggle_stage_c/` đã được xử lý ở đợt dọn liền trước: 34/34 file giải nén khớp SHA256 với ZIP Kaggle nguồn, các ZIP qua CRC, bản canonical đặt trong `weights/`, `metrics/`, `logs/`; ZIP nguồn và artifact còn nguyên trong `archive/e3_kaggle_evidence/`. Không còn các thư mục cũ trong `results/`. `results/lam_sach_du_lieu/` cùng mọi bằng chứng làm sạch giữ nguyên.

## Checkpoint, metrics và giới hạn

| Mô hình | Inference checkpoint | Recovery / metrics |
|---|---|---|
| E3 | `results/E3_YOLO11n/weights/best.pt` — SHA256 `41f7943cbab67d1d5920e2852068b1e98d22eda8023f978abcb608a65a2aed02` | `weights/recovery_full.pt`, `metrics/results.csv`, `metrics/plots/`, `logs/args.yaml` và manifest |
| E4 | `results/E4_YOLO11s/checkpoints/best.pt` — SHA256 `dc09c71052027e9083f014b99144edb65bcdc2ed032ab3591fce0db52b26e2f5` | `checkpoints/recovery_full.zip`, `checkpoints/last.zip`, `metrics/results.txt` (10 epoch), `logs/config.txt` |
| E6 | `results/E6_Faster_RCNN/checkpoints/best_weights.zip` — SHA256 `ce459c56afc57efec80c63ab213bab60ecb9a966c78a1396051de6524783f9bd` | `checkpoints/checkpoint.zip`; `metrics.csv`, `config.json`, `environment.json`, `last_hf_manifest.json` rời **MISSING** tại local |

E4 `best.zip` trùng SHA256 với `best.pt`; bản ZIP gốc đã nằm trong backup ngoài project trước khi xóa bản làm việc. Hai file E4 `results.txt` và `config.txt` được **di chuyển**, giữ nguyên byte. E6 checkpoint vẫn là định dạng PyTorch dù mang đuôi `.zip`; không giải nén. E6 best weights đã được nạp strict vào `fasterrcnn_resnet50_fpn_v2` với head 26 lớp: **404 tensor, PASS**. App hiện chỉ hỗ trợ E3/E4; không giả định E6 đã được tích hợp. Các file E6 metadata rời còn thiếu nên chưa hậu kiểm toàn bộ artifact E6. Báo cáo đồ án chính thức `docs/BAO_CAO.md` chưa có; không tạo nội dung kết quả khoa học chưa được đo.

## Cây kết quả cuối

```text
results/
├─ lam_sach_du_lieu/ (analysis, conversion_audit, final, smoke, v2, v3, v4, verify)
├─ E3_YOLO11n/ (confidence, weights, metrics, logs)
├─ E4_YOLO11s/ (confidence, checkpoints, metrics, logs)
└─ E6_Faster_RCNN/ (checkpoints)
```

Không tạo `models/` trống hoặc bản sao checkpoint lớn. `results/E3_YOLO11n/confidence/` và `results/E4_YOLO11s/confidence/` không thay đổi. Ba notebook chính thức giữ tên file gốc để bảo toàn liên kết và khả năng truy vết, không đổi fingerprint.

## Kiểm thử

- Unit tests hiện hành sau khi bỏ test chỉ phục vụ workflow lịch sử: **58 passed, 7 subtests passed, 0 failed**.
- E6 strict model loading với 26 lớp: **PASS**.
- E3/E4 Gradio inference bằng checkpoint fine-tuned thật: được bao phủ trong unit tests, **PASS**.
- ZIP backup ngoài project: CRC, manifest, SHA256 từng file **PASS**. E4 best duplicate: SHA256 trùng `best.pt` **PASS**.
- Python syntax **52 file PASS**; JSON/cell Python của **3 notebook chính thức PASS** (bỏ qua `%pip` magic khi phân tích cú pháp); YAML E3/E4/ROD Final và JSON confidence **PASS**. Import/inference E3/E4 nằm trong unit tests. Markdown README/docs/results: **1.761 link, 0 hỏng**.

Chạy ứng dụng trên Windows: từ thư mục project, dùng `.\.venv\Scripts\python.exe -m app.main`. Dùng ba notebook chính thức tại `notebooks/kaggle_rod_e3_finalize_10epochs.ipynb`, `notebooks/Kaggle_E4_YOLO11s_HF_AutoResume.ipynb` và `notebooks/Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb` khi cần tái kiểm tra thực nghiệm.
