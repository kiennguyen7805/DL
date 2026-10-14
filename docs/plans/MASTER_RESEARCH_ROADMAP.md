# Master research roadmap — phạm vi hiện hành

**Đề tài:** Phát hiện vật cản trong ảnh hướng tới hỗ trợ di chuyển cho người khiếm thị.

**Sản phẩm:** Upload ảnh → mô hình phát hiện vật cản → hiển thị bounding box, tên lớp và confidence. Webcam, khoảng cách, TTS, GPS và database không thuộc phạm vi hiện tại. Bản roadmap trước quyết định này được lưu tại [`archive/plans/MASTER_RESEARCH_ROADMAP_before_upload_image_scope.md`](../../archive/plans/MASTER_RESEARCH_ROADMAP_before_upload_image_scope.md); các kế hoạch 01–04 giữ làm lịch sử, không phải danh sách việc cần chạy.

## Dữ liệu và nguyên tắc

ROD Final A4 đã freeze tại `data/processed/rod_detection_final/`: TRAIN 18.151, VALID 3.318, TEST 1.611 ảnh, tổng 23.080 ảnh và 25 class. Không làm sạch lại hoặc đổi split. TRAIN dùng để fine-tune, VALID dùng để theo dõi/chọn checkpoint và mô hình, TEST chỉ dùng đánh giá cuối cùng sau khi khóa quyết định. Cần ghi nhận noise nhãn và near-duplicate chưa được chứng minh đã giải quyết hoàn toàn.

## Ba thí nghiệm chính

| ID | Mô hình | Ngân sách | Trạng thái được quan sát |
|---|---|---:|---|
| E3 | YOLO11n COCO Pretrained | 10 epoch | 10/10, artifact/checkpoint đã audit local |
| E4 | YOLO11s COCO Pretrained | 10 epoch | 10/10 trên Kaggle, `best.pt` đã nạp local; cần hậu kiểm đủ artifact |
| E6 | Faster R-CNN ResNet-50 FPN V2 COCO Pretrained | 10 epoch | checkpoint local ghi 10/10; best weights nạp strict PASS; thiếu metrics/config/manifest rời để xác minh artifact đầy đủ |

E5 Scratch chỉ còn là lịch sử, không có bước huấn luyện hay đánh giá trong roadmap này. E3 đã chạy 10 epoch đầu của **lịch learning rate gốc 30 epoch**; không ghi thành lịch thiết kế cho 10 epoch. So sánh kiến trúc phải công bố optimizer, schedule, batch, phiên bản framework, phần cứng và thời gian train; không quy toàn bộ khác biệt kết quả cho kiến trúc.

## Đường đi có điều kiện hoàn thành

| Bước | Việc cần làm | Đầu ra | PASS khi |
|---|---|---|---|
| 1 | Bảo toàn E3, hậu kiểm E4 và E6; không dùng chung checkpoint | checkpoint, manifest, SHA256, metrics và log riêng mỗi mô hình | artifact đọc lại được, đủ epoch, config và dataset đúng |
| 2 | Chuẩn hóa evaluator ba mô hình trên VALID | metric tổng/per-class, precision, recall, mAP50, mAP50-95, FP/FN | cùng schema, IoU và quy tắc tính; 25 class đủ |
| 3 | Benchmark inference cùng phần cứng và ảnh | FPS, latency, VRAM, params, protocol | đo thật và lưu điều kiện đo |
| 4 | Chọn mô hình triển khai bằng VALID, chú ý Recall và FN | quyết định model/checkpoint/ngưỡng | lựa chọn và giới hạn có bằng chứng, chưa xem TEST để đổi lựa chọn |
| 5 | Đánh giá TEST một lần sau khi khóa quyết định | metric cuối và error analysis | không điều chỉnh lại model/ngưỡng theo TEST |
| 6 | Duy trì ứng dụng upload ảnh E3/E4 đã có; tích hợp E6 sau hậu kiểm | giao diện và mã inference | upload ảnh, bbox, tên lớp, confidence hoạt động với checkpoint đã chọn |
| 7 | Kiểm thử, viết báo cáo và chuẩn bị bảo vệ | báo cáo, hình minh họa, demo | mọi số liệu truy vết được; không tuyên bố chức năng chưa làm |

Chỉ một phiên Kaggle GPU hoạt động tại một thời điểm. Notebook E3/E4/E6 độc lập; E4/E6 lưu checkpoint riêng. Kaggle Save Version và backup ngoài runtime là điều kiện lưu kết quả bền vững. E6 đã có checkpoint ghi 10 epoch, nhưng chưa đạt mức VERIFIED cho toàn bộ artifact khi metrics/config/manifest rời chưa được đối chiếu.

## Tài liệu trạng thái

- [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)
- [Báo cáo freeze ROD Final](../../results/lam_sach_du_lieu/final/a4_report.md)
- [Hướng dẫn Kaggle hiện hành](../kaggle/HUONG_DAN.md)
- [Tài liệu tham khảo](../references.md)
