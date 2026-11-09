# Kết quả mới E3/E4/E6 — E6 hoàn tất 30 epoch và đã đánh giá đủ VALID/TEST

Cập nhật ngày 28/10/2026. Nguồn E6 mới: `results_valid_e6_30.zip`; nguồn E3/E4 giữ nguyên `results_danhgiae3e4.zip`. Nội dung này thay hướng dẫn cũ chưa có đánh giá E6 mới. Các tài liệu Word/PDF/PPTX đi kèm là nguồn để chỉnh trên web, chưa được sửa nội dung trực tiếp trong lần đóng gói này.

## Bảng VALID/TEST phải dùng

| Mô hình | Split | Epoch run | Ảnh | Ngưỡng | mAP50 | mAP50–95 | Precision | Recall | F2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 — YOLO11n | VALID | 30 | 3318 | 0.27 | 0.874701 | 0.698278 | 0.771960 | 0.850752 | 0.833732 |
| E3 — YOLO11n | TEST | 30 | 1611 | 0.27 | 0.887897 | 0.716404 | 0.729181 | 0.870433 | 0.837968 |
| E4 — YOLO11s | VALID | 30 | 3318 | 0.24 | 0.885642 | 0.719386 | 0.808310 | 0.870301 | 0.857153 |
| E4 — YOLO11s | TEST | 30 | 1611 | 0.24 | 0.898635 | 0.735530 | 0.819368 | 0.880629 | 0.867654 |
| E6 — Faster R-CNN ResNet-50 FPN V2 | VALID | 30 | 3318 | 0.60 | 0.870263 | 0.660507 | 0.750919 | 0.844925 | 0.824287 |
| E6 — Faster R-CNN ResNet-50 FPN V2 | TEST | 30 | 1611 | 0.60 | 0.877393 | 0.677107 | 0.755195 | 0.849193 | 0.828567 |

VALID: 3.318 ảnh, 5.320 đối tượng. TEST: 1.611 ảnh, 2.354 đối tượng. E6 VALID: TP=4.495, FP=1.491, FN=825. E6 TEST: TP=1.999, FP=648, FN=355. Ngưỡng E6 **0,60** được chọn bằng micro-F2 trên VALID trước TEST; không dùng lại ngưỡng 0,57 của checkpoint trước.

## Checkpoint và kiểm chứng

- E6: Faster R-CNN ResNet-50 FPN V2, 25 lớp ROD và một background, run 30/30, `is_complete=true`. Best SHA256 `a75beb6e3d8629bf9066bc8a03a0b44de93a720aa3d7b3663f0641b732ae8a8f`.
- Đã xác minh SHA256 trọng số, predictions, metrics VALID, threshold sweep và hash khóa được TEST tham chiếu; thứ tự/tổng số ảnh khớp dataset Final local. Đã tính lại các metric từ prediction cache bằng evaluator trong app và khớp kết quả được cung cấp.
- Mỗi mô hình có khóa VALID riêng tại `metrics/<model>/evaluation_lock.json`. Chỉ mục khóa trong app được tạo sau khi nhập kết quả, không phải một lần khóa chung trước TEST.
- AP nội suy 101 điểm, IoU 0,50:0,05:0,95; class-aware matching; AP maxDet=100, P/R/F2 maxDet=300, IoU vận hành 0,50. Gọi là COCO-style nội bộ, không phải pycocotools chính thức.

## Kết quả huấn luyện E6

| Epoch | mAP50 trong log | mAP50–95 trong log | mAR@100 | Loss |
|---|---:|---:|---:|---:|
| 28 | 0.866035 | 0.647005 | 0.717775 | 0.061214 |
| 29 | 0.865957 | 0.648472 | 0.717774 | 0.061124 |
| 30 | 0.866528 | 0.652066 | 0.720636 | 0.061149 |

Log đủ 30 epoch nằm ở `metrics/E6/training/metrics.csv`. Epoch 30 có mAP50–95 cao nhất log: 0,6520662308, vượt mốc best cũ ở epoch 15 (0,6518420577). Không trộn đường cong log training với số tính lại bằng evaluator chung. mAR@100 không thay thế Recall ở ngưỡng 0,60.

## Nhận xét có thể chèn vào báo cáo

> Ba mô hình E3, E4 và E6 đều đã hoàn tất các run 30 epoch và được đánh giá bằng cùng protocol trên 3.318 ảnh VALID và 1.611 ảnh TEST. Với ngưỡng được chọn riêng trên VALID, E4 đạt chất lượng TEST cao nhất ở cả mAP50, mAP50–95, Precision, Recall và F2. E6 sử dụng ngưỡng 0,60, đạt mAP50 0,877393; mAP50–95 0,677107; Precision 0,755195; Recall 0,849193 và F2 0,828567. So với E3, E6 có Precision cao hơn nhưng các chỉ số mAP, Recall và F2 thấp hơn. Do đó, kết quả hiện có không ủng hộ kết luận E6 tốt nhất tổng thể.

E4 so với E6 trên TEST: Precision +6.4173, Recall +3.1436, F2 +3.9088, mAP50 +2.1242, mAP50–95 +5.8423 **điểm phần trăm**; E4 có ít hơn 74 FN và 191 FP. AP50–95 theo lớp dẫn đầu: E3 7/25, E4 17/25, E6 1/25; nếu có hòa điểm thì mỗi mô hình đồng hạng được tính một lần.

## E6 theo lớp trên TEST — checkpoint mới, ngưỡng 0,60

| Lớp | GT | AP50 | AP50–95 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Bike | 48 | 0.941756 | 0.807771 | 45 | 9 | 3 |
| Building | 54 | 0.898382 | 0.724033 | 50 | 14 | 4 |
| Car | 247 | 0.828113 | 0.604089 | 193 | 58 | 54 |
| Person | 320 | 0.829833 | 0.563730 | 233 | 49 | 87 |
| Stairs | 222 | 0.816959 | 0.605368 | 195 | 145 | 27 |
| Traffic sign | 104 | 0.819359 | 0.527111 | 89 | 55 | 15 |
| Electrical Pole | 42 | 0.888090 | 0.642809 | 38 | 8 | 4 |
| Road | 37 | 0.745220 | 0.483172 | 31 | 17 | 6 |
| Motorcycle | 38 | 0.752850 | 0.503392 | 26 | 8 | 12 |
| Dustbin | 45 | 0.654331 | 0.515663 | 29 | 8 | 16 |
| Dog | 30 | 0.996806 | 0.777042 | 30 | 4 | 0 |
| Manhole | 242 | 0.944469 | 0.828499 | 229 | 34 | 13 |
| Tree | 150 | 0.825200 | 0.584700 | 117 | 22 | 33 |
| Guard rail | 69 | 0.888910 | 0.615519 | 62 | 15 | 7 |
| Pedestrian crosswalk | 189 | 0.832246 | 0.581025 | 157 | 67 | 32 |
| Truck | 43 | 0.867964 | 0.647914 | 37 | 12 | 6 |
| Bus | 55 | 0.842076 | 0.605643 | 45 | 16 | 10 |
| Bench | 28 | 0.960396 | 0.793746 | 27 | 13 | 1 |
| Traffic Cone | 92 | 0.942154 | 0.681969 | 88 | 20 | 4 |
| Fire hydrant | 25 | 0.998477 | 0.876473 | 24 | 0 | 1 |
| Teraffic Barrel | 19 | 1.000000 | 0.971400 | 19 | 5 | 0 |
| Plant Pot | 34 | 0.906154 | 0.730302 | 30 | 7 | 4 |
| Electrical Box | 34 | 0.972242 | 0.855981 | 33 | 9 | 1 |
| Chair | 151 | 0.934764 | 0.707267 | 141 | 41 | 10 |
| Bicycle Rack | 36 | 0.848079 | 0.693060 | 31 | 12 | 5 |

## Các phần cần sửa

| Phần | Việc cần sửa |
|---|---|
| Tóm tắt/Abstract | Dùng kết quả mới, ghi E6 đã hoàn tất training và đủ VALID/TEST. |
| Thiết lập thí nghiệm | Run 30 epoch cho cả ba; phân biệt best checkpoint với last checkpoint, khóa riêng từng mô hình. |
| Bảng kết quả/biểu đồ | Thay số E6 cũ, giữ số E3/E4, ngưỡng 0,27/0,24/0,60. |
| Phân tích theo lớp | Dùng CSV mới, cập nhật số lớp dẫn đầu và TP/FP/FN. |
| Training E6 | Log đủ 30 epoch, best ở epoch 30, thêm các mốc 28–30. |
| Demo/ứng dụng | E6 nạp trọng số mới, ngưỡng tự động 0,60. |
| So sánh/kết luận/giới hạn | E4 dẫn đầu tổng thể; giữ giới hạn benchmark cũ và prototype phân vùng. |
| Mục lục/danh sách hình bảng | Cập nhật theo trang thực tế sau khi sửa. |

## Prompt dán lên web

Hãy cập nhật báo cáo nguồn Paper_ACL_E6_30Epoch_BanMoi_27102026.docx và PDF trong gói này bằng toàn bộ kết quả đã xác minh trong KET_QUA_MOI_E3_E4_E6.md. Đây là bản cập nhật mới sau khi đã chạy đủ VALID và TEST cho best checkpoint E6 của run hoàn tất 30 epoch; thay thế các ghi chú cũ “E6 chưa đánh giá lại”, “E6 chỉ 10 epoch” hoặc “E6 dừng ở 28/30”. E3/E4/E6 đều đã hoàn tất run 30 epoch; bảng so sánh hiện dùng các best checkpoint được chọn trong các run đó. Không tự khẳng định best checkpoint của cả ba đều xuất hiện ở epoch 30: riêng log E6 cho biết best tại epoch 30.

Dùng đúng bảng sáu dòng VALID/TEST trong file Markdown và CSV. Ngưỡng E3/E4/E6 lần lượt là 0,27/0,24/0,60. Mỗi ngưỡng được chọn bằng micro-F2 trên VALID rồi khóa trước TEST, với khóa riêng cho từng mô hình. Không nói có một khóa chung được tạo trước ba lần TEST: app/evaluation_lock.json chỉ là chỉ mục local nhập sau đánh giá, tham chiếu các khóa gốc. Thay cả ngưỡng E6 0,57/0,59 lẫn mọi số E6 cũ trong phần kết quả hiện hành. Nếu nhắc lại số cũ để so sánh lịch sử, phải gắn rõ nhãn checkpoint cũ.

Cập nhật nhất quán Abstract/tóm tắt, thiết lập thí nghiệm, kết quả VALID/TEST, bảng theo lớp, biểu đồ, phần ứng dụng/demo, thảo luận, giới hạn và kết luận. Bổ sung bảng training epoch 28/29/30, dùng CSV đủ 30 epoch để vẽ đường cong. Phân biệt mAP50–95 trong log E6 (best 0,652066 tại epoch 30) với evaluator chung sau training (VALID 0,660507; TEST 0,677107). mAR@100 trong log không phải Recall tại ngưỡng vận hành 0,60. Không dùng TEST để chọn lại checkpoint, threshold hoặc giải thích rằng ngưỡng được tối ưu trên TEST.

Kết luận cho bảng mới: E4 cao nhất cả năm chỉ số TEST tổng hợp. E6 cao hơn E3 về Precision TEST, nhưng thấp hơn E3 ở mAP50, mAP50–95, Recall và F2. Đánh giá cải thiện theo đúng số: hoàn tất thêm epoch không có nghĩa mọi chỉ số TEST đều tăng. Nếu phân tích theo lớp, dùng CSV so sánh mới; không giữ số lớp dẫn đầu cũ. Đây là evaluator COCO-style nội bộ, không phải pycocotools chính thức. Không tuyên bố kết quả CPU/latency cũ đã đo lại cho checkpoint mới; giữ thì ghi baseline cũ.

Giữ bố cục, trích dẫn và phương pháp không liên quan. Giữ hoặc cập nhật mục lục và kê khai AI 75% theo mẫu đã có; cập nhật số trang thực tế. Bổ sung giới hạn: heuristic phân vùng bbox chỉ là prototype, không suy ra khoảng cách hoặc độ an toàn thực tế. Nếu chỉnh slide baocao_final.pptx đi kèm, cập nhật cả biểu đồ, kết luận và speaker notes theo cùng số liệu; giữ theme/tỷ lệ. Không tạo code Gradio. Không bịa số liệu. Trả báo cáo DOCX và PDF mới, kèm danh sách các mục đã thay. Render và kiểm tra toàn bộ trang để sửa tràn bảng, đè chữ, lỗi tiếng Việt và số trang trước khi xuất.
