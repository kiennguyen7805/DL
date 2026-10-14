# Lời thuyết trình bảo vệ đồ án

## Slide 1 — Trang bìa

Giới thiệu ngắn đề tài và ba mô hình. Nhấn mạnh đây là nghiên cứu phát hiện vật cản trong ảnh tĩnh, hướng tới một thành phần hỗ trợ nhận thức cho người khiếm thị.

## Slide 2 — Đặt vấn đề

Một cảnh đô thị có nhiều vật thể với kích thước và mức che khuất khác nhau. Bài toán không chỉ cần phát hiện đúng mà còn phải hạn chế bỏ sót và cân nhắc thời gian xử lý. Không khẳng định hệ thống thay thế gậy dò đường hay bảo đảm an toàn.

## Slide 3 — Mục tiêu và phạm vi

Trình bày bốn mục tiêu: dữ liệu, fine-tuning, so sánh và demo. Chuyển ý: trước khi chọn mô hình, cần xây dựng bộ dữ liệu có split rõ ràng và kiểm soát leakage.

## Slide 4 — Dữ liệu

Nêu 23.080 ảnh, 38.467 bbox và 25 lớp. TEST gồm 1.611 ảnh được khóa và chưa dùng để chọn mô hình. Ảnh mẫu cho thấy dữ liệu có cảnh phức tạp và nhiều đối tượng.

## Slide 5 — Làm sạch

Giải thích leakage khiến mô hình có thể gặp lại ảnh gần như giống nhau ở VALID/TEST và làm kết quả lạc quan giả tạo. Sau làm sạch đã loại 178 ảnh khỏi TRAIN/VALID và không còn nhóm trùng chính xác giữa các split. Các đề xuất sửa nhãn chưa áp dụng được công khai như giới hạn.

## Slide 6 — Quy trình

TRAIN dùng tối ưu trọng số; VALID dùng chọn checkpoint và ngưỡng. Sau khi khóa cấu hình, ba mô hình được chấm một lần trên TEST bằng cùng evaluator.

## Slide 7 — Object Detection

Phân biệt classification chỉ nói ảnh chứa gì, còn detection trả thêm vị trí. Giới thiệu nhanh one-stage và two-stage để chuẩn bị cho hai slide kiến trúc.

## Slide 8 — YOLO11

Backbone trích xuất đặc trưng bằng Conv, C3k2, SPPF, C2PSA. Neck hợp nhất đặc trưng từ nhiều độ phân giải. Head dự đoán trên P3/P4/P5, sau đó NMS loại các bbox trùng lặp. Đây là sơ đồ khái quát đúng các khối của cấu hình YOLO11, không phải sơ đồ YOLOv5/YOLOv8 đổi tên.

## Slide 9 — YOLO11n và YOLO11s

Hai mô hình dùng cùng ý tưởng kiến trúc nhưng YOLO11s rộng/sâu hơn và có số tham số lớn hơn. Trên TEST, E4 cải thiện mAP50–95 khoảng 2,59 điểm phần trăm so với E3 nhưng độ trễ CPU tăng khoảng 2,44 lần.

## Slide 10 — Faster R-CNN

ResNet-50 và FPN tạo đặc trưng đa tỉ lệ. RPN đề xuất vùng có khả năng chứa vật thể. RoI Align lấy đặc trưng từng vùng, rồi RoI Head phân loại và tinh chỉnh bbox. Đây là biến thể `fasterrcnn_resnet50_fpn_v2`, không phải ResNet-50 thuần.

## Slide 11 — Lý do chọn mô hình

YOLO11n và YOLO11s trả lời câu hỏi về quy mô trong cùng một họ. Faster R-CNN tạo đối chứng khác phương pháp phát hiện. Không đặt giả thuyết rằng mô hình lớn hơn chắc chắn tốt hơn; kết luận dựa trên kết quả đo.

## Slide 12 — Cấu hình huấn luyện

Cả ba dùng COCO pretrained và 10 epoch. E3/E4 gần nhau về pipeline; E6 khác batch, image transform và scheduler. Vì vậy đây là so sánh hệ thống theo cấu hình đã chạy, không phải thí nghiệm cô lập tuyệt đối từng biến. Mười epoch chưa chứng minh hội tụ tối ưu.

## Slide 13 — Đánh giá

Recall quan trọng vì FN là vật cản có thật bị bỏ sót. Ngưỡng vận hành được chọn theo F2 trên VALID. AP được tính ở nhiều ngưỡng confidence; evaluator là giao thức COCO-style tùy biến, không phải kết quả pycocotools chính thức.

## Slide 14 — Kết quả TEST

E6 dẫn đầu cả Precision, Recall, mAP50 và mAP50–95 trên TEST. So với E4, E6 tăng khoảng 9,08 điểm phần trăm Precision, 7,65 điểm Recall và 9,75 điểm mAP50–95. Các ngưỡng 0,22; 0,19; 0,59 đã được chọn từ VALID trước khi mở TEST.

## Slide 15 — Hiệu quả tính toán

Benchmark dùng cùng CPU, 30 ảnh VALID, batch 1, warmup 3. E3 đạt khoảng 13,11 FPS, E4 5,38 FPS và E6 0,398 FPS. E6 có chất lượng cao nhất nhưng chậm hơn đáng kể; lựa chọn cuối phụ thuộc mục tiêu triển khai và phần cứng.

## Slide 16 — Demo

Người dùng tải ảnh, chọn một trong ba mô hình, chạy nhận diện và nhận ảnh PNG chứa bbox, lớp và confidence. Ứng dụng còn hiển thị số lượng và thời gian xử lý. Ảnh trên slide là kết quả thực tế đã lưu, không phải mockup.

## Slide 17 — Kết quả và giới hạn

Phân biệt rõ phần đã làm với phần đề xuất. Hạn chế quan trọng gồm số epoch, một lần chạy, thiếu log huấn luyện rời đầy đủ của E6, TEST chưa chạy và chưa đánh giá trên thiết bị mục tiêu hay điều kiện sử dụng thực tế.

## Slide 18 — Kết luận

E3 phù hợp khi ưu tiên tốc độ CPU; E4 là điểm trung gian; E6 đạt chất lượng TEST cao nhất nhưng tốn tài nguyên nhất. Lựa chọn cuối cùng phụ thuộc mục tiêu triển khai và phần cứng.

## Slide 19 — Phụ lục lớp

Dùng khi hội đồng hỏi bộ dữ liệu gồm những vật cản nào. Lưu ý một số lớp là thành phần cảnh như Road/Building/Tree, phản ánh ontology gốc của ROD.

## Slide 20 — Phụ lục khối YOLO11

Giải thích C3k2 là khối CSP hiệu quả, SPPF mở rộng trường tiếp nhận, C2PSA bổ sung attention không gian. Tránh trình bày các chi tiết nội bộ không kiểm chứng từ source.

## Slide 21 — Phụ lục Faster R-CNN

Giải thích FPN giúp vật thể nhiều kích thước, RPN tạo proposal bằng objectness và bbox regression, RoI Align tránh sai số lượng tử hóa khi lấy đặc trưng vùng.

## Slide 22 — Phụ lục metric

Dùng ví dụ TP/FP/FN để trả lời vì sao tăng Recall có thể kéo theo FP. mAP50–95 khắt khe hơn mAP50 vì yêu cầu bbox chính xác ở nhiều ngưỡng IoU.

## Slide 23 — Phụ lục nguồn

Nguồn số liệu chính: `docs/reports/BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU/FACTS.md`, manifest ROD Final A4, các tệp metrics/benchmark trong `results/`, `app/evaluator.py` và `app/evaluation_lock.json`. Nguồn kiến trúc: Ultralytics documentation/source; Ren et al. (Faster R-CNN); He et al. (ResNet); Lin et al. (FPN); Torchvision documentation.

