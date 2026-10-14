# Chạy E4 và E6 tự lưu, tự resume trên Kaggle

Hai notebook độc lập trên ROD Final A4: TRAIN 18.151 ảnh, VALID 3.318 ảnh, 25 lớp. Không train/VALID trên TEST; không chạy E3 hoặc E5.

## Trạng thái kiểm chứng

Local test đã PASS cho notebook JSON/Python syntax, tách folder HF, snapshot, SHA256, config/dataset hash, điều kiện COMPLETE, chuyển YOLO class 0..24 sang Torchvision foreground 1..25. E6 đã thực sự hoàn thành 2 epoch và VALID trên Kaggle GPU, rồi phiên lỗi vì hết dung lượng đĩa; resume ở phiên kế tiếp vẫn chưa được xác minh thực tế. Repo `iraprp/deeplearning` không truy cập được qua API công khai từ máy local; notebook kiểm tra repo_type bằng HF_TOKEN2 lúc chạy và dừng nếu không xác định được. Notebook không tạo repo.

## Cấu hình hai tài khoản

| Tài khoản | Notebook import | Folder HF | Kaggle Output |
|---|---|---|---|
| Tài khoản chạy E4 | `notebooks/Kaggle_E4_YOLO11s_HF_AutoResume.ipynb` | `YOLO 11s/` | `/kaggle/working/rod_e4_yolo11s_hf/` |
| Tài khoản chạy E6 | `notebooks/Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb` | `Faster R-CNN/` | `/kaggle/working/rod_e6_faster_rcnn_hf/` |

Trên **mỗi tài khoản**: Add Input dataset `nguyenkin/deeplearningg` chứa `rod_detection_final`; Accelerator = GPU; Internet = On; tạo Kaggle Secret tên `HF_TOKEN2` có quyền đọc/ghi repo đích và cấp quyền sử dụng Secret cho notebook. Không dán giá trị token vào cell, output hoặc tài liệu. Không cần upload module Python khác; mã đã nhúng vào notebook.

## Lần đầu và các lần tiếp theo

1. Import notebook tương ứng, chọn **Save Version → Save & Run All**. Trước training, notebook xác minh 18.151/3.318 cặp ảnh/nhãn, đúng thứ tự 25 class, đường dẫn TRAIN/VALID, config SHA256, GPU và loại repo HF. Folder HF phải trống nếu chưa có run.
2. E4 khởi tạo `yolo11s.pt` COCO; E6 khởi tạo Faster R-CNN ResNet-50 FPN V2 COCO rồi thay head thành 26 lớp. Nếu folder có file nhưng thiếu `latest.json`, notebook dừng để kiểm tra, không bắt đầu lại từ epoch 1.
3. Phiên tiếp theo mở **Edit → Save Version → Save & Run All** trên cùng notebook, dataset và HF Secret. Notebook đọc `latest.json`, tải snapshot đúng folder, kiểm tra manifest/checksum, rồi tiếp tục. Không cần sửa stage hay nhập path.
4. Xem `last_hf_manifest.json` trong Kaggle Output và `latest.json` trên HF. Chỉ kết luận xong khi `completed_epoch=10`, `is_complete=true` và hậu kiểm checkpoint PASS. Save Version thành công đơn thuần chưa đủ.

## Checkpoint và giới hạn thời gian

- **E4:** Ultralytics 8.4.165, batch 8, imgsz 640, SGD và hyperparameter theo `configs/e4_yolo11s_pretrained.yaml`. Mỗi epoch callback sao lưu `last.pt` đầy đủ **trước** khi Ultralytics strip optimizer. Checkpoint được kiểm tra optimizer, AMP scaler, EMA, epoch, kiến trúc s, 25 class và effective args. Upload `recovery_full.pt`, `last.pt`, `best.pt` nếu có, `results.csv`, `args.yaml`, `config.json`, `environment.json`, `manifest.json`. Dừng tại epoch 10 hoặc sau một epoch hoàn chỉnh nếu đã dùng khoảng 10 giờ. E4 chạy 10 epoch đầu của lịch LR **30 epoch**, phù hợp lịch E3 đã thực sự chạy.
- **E6:** batch 2; Torchvision label 1..25, background 0. Lưu `checkpoint.pt` mỗi **1.800 minibatch**, cuối epoch và khi gần 9,5 giờ. Checkpoint gồm model, optimizer, scheduler, AMP scaler, RNG, epoch, offset mẫu, tổng/count loss, metrics. Cùng snapshot có `metrics.csv`, `best_weights.pth` nếu có, `config.json`, `environment.json`, `manifest.json`. Resume giữa epoch tái tạo cùng thứ tự ảnh từ seed, giữ số batch và lịch LR trong điều kiện single GPU. Không tuyên bố tái lập bitwise giữa phần cứng hoặc phiên bản thư viện khác.
- Mỗi snapshot dùng UUID mới trong `<folder>/snapshots/`; `latest.json` chỉ đổi sau khi snapshot được upload và manifest được kiểm tra. Hai folder tách riêng, không ghi đè checkpoint nhau. Bản staging tạm trong Kaggle Output được dọn **chỉ sau khi** HF snapshot và pointer đã xác minh; checkpoint gốc local vẫn giữ. Snapshot cũ trên HF không bị xóa tự động.
- Nếu upload HF lỗi, notebook dừng, giữ checkpoint/log trong Kaggle Output và **không** đánh dấu đồng bộ. Tải Kaggle Output về ngay. `/kaggle/working` phiên cũ không tự có lại ở phiên mới; phiên mới chỉ resume từ snapshot đã có trên HF. Nếu phần tiến độ chỉ còn trong Output, cần cứu thủ công, không tự train lại các epoch đó.

## Theo dõi lần chạy thật

Theo dõi GPU/VRAM, thời gian epoch đầu, mAP VALID và snapshot trên HF. Đây là bài kiểm tra tích hợp chưa chạy tại local. Nếu OOM, timeout hoặc HF từ chối upload, giữ Output và lỗi; không tự đổi batch, LR, kích thước ảnh, lịch hoặc thuật toán khi resume. Chạy trên hai tài khoản theo quota và điều khoản Kaggle áp dụng; không cần hai phiên GPU đồng thời.

## E6 sau lỗi IMG_04016 (2026-10-11)

Notebook E6 đã được cập nhật để clip sai số làm tròn cực nhỏ ở biên bbox. Hãy import lại file E6 mới từ thư mục notebooks/ vào Kaggle rồi chọn Save Version → Save & Run All. Không sửa ROD Final hoặc cấu hình training. Nếu HF đã có checkpoint epoch 0 hợp lệ, notebook sẽ tải và xác minh nó; nếu không có checkpoint, notebook khởi tạo COCO pretrained. Lỗi ở phiên trước xảy ra trước khi hoàn thành epoch 1; các minibatch chưa backup không thể tự khôi phục. Không cần upload lại notebook E4.

## E6 sau lỗi hết dung lượng ở epoch 3 (2026-10-11)

Log Kaggle đã xác nhận epoch 1 và 2 hoàn thành, rồi Papermill báo `No space left on device`. Notebook E6 mới đã dọn staging local sau khi HF snapshot được xác minh và giảm chu kỳ backup còn 1.800 minibatch. **Import lại notebook E6 mới** từ đường dẫn ở bảng trên vào tài khoản E6 rồi chọn Save Version → Save & Run All. Không sửa HF repo, không xóa checkpoint HF và không sửa ROD Final. Notebook sẽ đọc `Faster R-CNN/latest.json`, xác minh checkpoint và tiếp tục từ epoch/offset mới nhất đã lưu. Nếu pointer hoặc checkpoint không hợp lệ, notebook dừng thay vì train lại từ đầu; gửi log lỗi để xử lý. E4 không cần thay đổi vì phiên này là lỗi E6.

