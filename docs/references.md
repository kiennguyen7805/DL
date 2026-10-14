# TÀI LIỆU THAM KHẢO KHOA HỌC VÀ KỸ THUẬT — ROD / DETECTION / CẢNH BÁO

> File dùng chung cho bốn kế hoạch tại `docs/plans/`. Phạm vi hiện hành là E3/E4/E6 COCO Pretrained, cùng ROD Final 25 lớp; E5 Scratch chỉ còn là lịch sử. **Các paper cung cấp phương pháp và bối cảnh**, không cung cấp “hyperparameter tối ưu” cho ROD. Kết quả ROD phải đến từ các run thật.

## 1. Paper khoa học (peer-reviewed)

### [R1] Bergstra & Bengio (2012) — Random Search for Hyper-Parameter Optimization

- **Tác giả:** James Bergstra, Yoshua Bengio.
- **Công bố:** *Journal of Machine Learning Research* (JMLR), 13(10):281–305, 2012.
- **Link nhà xuất bản:** https://jmlr.org/papers/v13/bergstra12a.html
- **Nội dung liên quan:** Random Search là một chiến lược khảo sát tham số có cơ sở thực nghiệm/lý thuyết; dùng làm lựa chọn khả thi khi không gian tham số lớn và ngân sách bị giới hạn.
- **Áp dụng cho project:** Có thể chọn random search với seed cố định và phân phối khai báo trước. Nếu thử hết lưới ứng viên, phải gọi **grid search nhỏ**, không gọi Random Search. Paper **không đưa ra LR/imgsz tối ưu cho ROD**.

### [R2] Li et al. (2018) — Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization

- **Tác giả:** Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar.
- **Công bố:** *Journal of Machine Learning Research* (JMLR), 18(185):1–52, 2018.
- **Link nhà xuất bản:** https://jmlr.org/papers/v18/16-558.html
- **Nội dung liên quan:** Phân bổ thích nghi ngân sách cho nhiều trial, dừng sớm các cấu hình kém hứa hẹn; khác việc chạy mọi candidate với pilot epochs cố định.
- **Áp dụng cho project:** Cơ sở tham khảo cho quản lý ngân sách GPU và phương pháp multi-fidelity; **không** được ghi đã chạy Hyperband nếu implementation chỉ chạy pilot đồng đều hay grid/random search thông thường.

### [R3] He, Girshick & Dollár (2019) — Rethinking ImageNet Pre-Training

- **Tác giả:** Kaiming He, Ross Girshick, Piotr Dollár.
- **Công bố:** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019.
- **Link hội nghị:** https://openaccess.thecvf.com/content_ICCV_2019/html/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.html
- **PDF hội nghị:** https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf
- **Nội dung liên quan:** So sánh detector khởi tạo ngẫu nhiên với pretrained trong thí nghiệm của tác giả; pretraining có thể giúp hội tụ sớm nhưng training-from-scratch có thể theo kịp khi huấn luyện đủ dài trong những điều kiện họ khảo sát.
- **Vai trò hiện tại:** nguồn của câu hỏi Pretrained vs Scratch trong kế hoạch lịch sử; không dùng để biện minh cho một run E5 mới. Paper dùng ImageNet/detector khác; ba mô hình hiện hành dùng COCO Pretrained trên ROD.

## 2. Tài liệu kỹ thuật (KHÔNG phải paper peer-reviewed)

### [D1] Ultralytics — Model Training

- Link: https://docs.ultralytics.com/modes/train/
- Áp dụng: fine-tune YOLO11n/YOLO11s từ `.pt`, training arguments, checkpoints/resume và logging. Nhánh `.yaml` scratch chỉ thuộc lịch sử. Kiểm tra cú pháp với phiên bản Ultralytics đã cài và **ghi version thực tế**.

### [D2] Ultralytics — Hyperparameter Tuning

- Link: https://docs.ultralytics.com/guides/hyperparameter-tuning/
- Áp dụng: search space, cách tổ chức, lưu kết quả, lưu ý tuning ngắn. Built-in `.tune()` dùng evolutionary/genetic algorithm theo phiên bản hiện tại; **không tự gọi `.tune()` là Random Search hoặc Hyperband** nếu không có implementation tương ứng.

### [D3] ROD Dataset (Kaggle) / Dataset Card (Hugging Face)

- Kaggle: https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset
- Dataset card: https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO
- Áp dụng: nguồn dataset, class/annotation mô tả; mọi số lượng, license, split và tình trạng chuyển đổi phải kiểm tra độc lập trên bản sử dụng trong project.

## 3. Quy tắc trích dẫn trong báo cáo

1. **Phương pháp HPO:** [R1], nếu thực sự sử dụng ngẫu nhiên; [R2] như cơ sở quản lý ngân sách, chỉ ghi đã dùng Hyperband khi code có Hyperband thật; [D2] để nêu kỹ thuật Ultralytics.
2. **Ba kiến trúc hiện hành:** [R4], [R5], [D4], [D5] để giải thích YOLO/Faster R-CNN/FPN. [R3] chỉ tham khảo khi mô tả lịch sử E5, không phải giả thuyết hiện hành.
3. **Nguồn dữ liệu:** [D3] và thống kê `results/day1`, `results/lam_sach_du_lieu` do script thực tế sinh.
4. **Kết quả:** chỉ dẫn tới `results/hparam/*.csv`, `results/runs/*`, `results/metrics/*.csv` sau khi train/verify thật; chưa có ghi **CHƯA ĐO**.

## 4. Nguồn bổ sung cho phạm vi ba mô hình

### [R4] Ren et al. (2015) — Faster R-CNN

- **Tác giả:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
- **Tên:** *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*.
- **Link:** https://arxiv.org/abs/1506.01497
- **Áp dụng:** cơ sở của detector hai giai đoạn và Region Proposal Network. Số tốc độ/accuracy trong paper không phải kết quả trên ROD hoặc phần cứng project.

### [R5] Lin et al. (2017) — Feature Pyramid Networks

- **Tác giả:** Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie.
- **Công bố:** CVPR 2017.
- **Link:** https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html
- **Áp dụng:** cơ sở cho biểu diễn đa tỉ lệ trong Faster R-CNN ResNet-50 FPN V2 và giả thuyết về vật thể nhiều kích thước. Không khẳng định FPN sẽ tốt hơn YOLO trên ROD trước khi đo.

### [D4] Ultralytics YOLO11

- **Link:** https://docs.ultralytics.com/models/yolo11/
- **Áp dụng:** tên model `yolo11n.pt`, `yolo11s.pt`, task detection và workflow Ultralytics. Bảng COCO chỉ là thông tin nhà cung cấp, không dùng như metric ROD.

### [D5] Torchvision Faster R-CNN ResNet-50 FPN V2

- **Link:** https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
- **Áp dụng:** builder, COCO weights, transforms và giới hạn API detection. Phải pin version thực tế vì tài liệu cảnh báo module detection có thể thay đổi compatibility.

## 5. Nguồn cho khoảng cách và cảnh báo thời gian thực

### [D6] OpenCV Camera Calibration

- **Link:** https://docs.opencv.org/doc/doxygen/html/dc/dbb/tutorial_py_calibration.html
- **Áp dụng:** camera matrix, distortion và quy trình calibration. Project phải lưu camera/resolution cụ thể và đo reprojection error.

### [R6] Qin & Li (2022) — MonoGround

- **Tên:** *MonoGround: Detecting Monocular 3D Objects From the Ground*.
- **Công bố:** CVPR 2022.
- **Link:** https://openaccess.thecvf.com/content/CVPR2022/html/Qin_MonoGround_Detecting_Monocular_3D_Objects_From_the_Ground_CVPR_2022_paper.html
- **Áp dụng:** bằng chứng rằng monocular depth là bài toán ill-posed và ground-plane prior cung cấp ràng buộc hình học. Project không tuyên bố triển khai MonoGround nếu chỉ dùng calibration/homography.

### [R7] Zhang et al. (2021) — ByteTrack

- **Tên:** *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*.
- **Link:** https://arxiv.org/abs/2110.06864
- **Áp dụng:** tài liệu tham khảo cho tracking-by-detection và giữ ID qua frame nhằm điều tiết cảnh báo. Chỉ trích dẫn là implementation nếu project thực sự dùng ByteTrack; state machine/cooldown tự viết phải mô tả đúng là quy tắc project.

## 6. Quy tắc sử dụng nguồn mới

1. R4/R5 giải thích kiến trúc; không nhập metric paper vào bảng kết quả ROD.
2. D4/D5 là tài liệu kỹ thuật, phải ghi version package thực tế.
3. D6/R6 hỗ trợ giả định hình học; mọi sai số khoảng cách phải đến từ tập đo thật của camera demo.
4. R7 chỉ hỗ trợ lựa chọn tracker; chất lượng cảnh báo phải được kiểm thử bằng kịch bản project.
