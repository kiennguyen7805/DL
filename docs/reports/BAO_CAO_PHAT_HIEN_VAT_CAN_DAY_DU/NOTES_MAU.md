# NOTES MẪU — phân tích DOCX tham chiếu

Nguồn: `docs/reports/12423TN_12423018_NguyenTranKien (2).docx`. Tệp đã được đọc toàn bộ bằng `python-docx`, đối chiếu trực tiếp các phần OOXML (`document.xml`, `styles.xml`, `settings.xml`, header/footer, relationships và media), xuất PDF bằng Microsoft Word và kiểm tra trực quan đủ 45 trang. DOCX gốc không bị sửa.

## Định dạng và hệ thống trang

- Khổ A4 dọc 595,3 × 841,9 pt, một section; bìa dùng `titlePg`, không có trang ngang.
- Lề XML: trên 56,7 pt (2 cm), phải 71 pt (2,5 cm), dưới 17,3 pt (0,61 cm), trái 85,05 pt (3 cm); header 22,7 pt, footer 39,7 pt.
- Bìa có khung chữ nhật, tên Bộ/Trường, logo, loại đồ án, tên đề tài, ngành/chuyên ngành, sinh viên–giảng viên và địa điểm/năm. Bìa không hiện header/footer.
- Trang 2–45 có header giữa, chữ đậm nhỏ và đường kẻ mảnh; footer là số trang giữa. Trường `PAGE` tồn tại trong `footer1.xml`.
- Style mặc định ghi Calibri 12 pt, nhưng thân bài chủ yếu dùng định dạng trực tiếp khoảng 13 pt, giãn dòng 1,5, canh đều, thụt đầu dòng khoảng 1,25 cm. Tiêu đề chương khoảng 14 pt đậm, in hoa, căn giữa; mục/tiểu mục khoảng 13 pt đậm, căn trái.
- Caption khoảng 11 pt, căn giữa, số theo chương dạng “Hình 3.1” và “Bảng 3.1”. Hình thường chiếm 60–90% vùng chữ. Bảng có đường kẻ đầy đủ và hàng tiêu đề rõ.
- Mục lục Word dùng `TOC \\o "1-3"`; có danh mục thuật ngữ, bảng và hình. Trường `SEQ` dùng cho caption, `PAGEREF` cho số trang. Trích dẫn trong thân bài theo số trong ngoặc vuông.
- Không có phụ lục thực tế. Báo cáo mới bổ sung phụ lục cấu hình, taxonomy và tái lập vì dự án có bằng chứng kỹ thuật cần truy vết.

## Cấu trúc nội dung và chuẩn độ sâu

Word tính 45 trang và 10.415 từ. Phân bổ: bìa 1; nhận xét 1; cam đoan 1; cảm ơn 1; mục lục 2; thuật ngữ 1; danh mục bảng 1; danh mục hình 1; Chương 1 trang 10–12; Chương 2 trang 13–20; Chương 3 trang 21–36; Chương 4 trang 37–41; tài liệu tham khảo trang 42–45.

| Mục mẫu | Trang | Số từ ước lượng | Cách trình bày |
|---|---:|---:|---|
| 1.1 Lý do chọn đề tài | 10–11 | 517 | Bối cảnh → vấn đề → khoảng trống → lựa chọn đề tài; có nguồn. |
| 1.2 Mục tiêu | 11 | 152 | Đoạn định hướng và danh sách mục tiêu cụ thể. |
| 1.3 Giới hạn/phạm vi | 11–12 | 152 | Tách đối tượng và phạm vi; nói rõ phần không làm. |
| 1.4–1.5 Nội dung/phương pháp | 12 | 149 | Pipeline và quy trình nghiên cứu ngắn gọn. |
| Chương 2 | 13–20 | khoảng 2.540 | Khái niệm → kỹ thuật → thuật toán; văn xuôi, công thức, ví dụ và liên hệ đề tài. |
| 3.1 Thu thập dữ liệu | 21–22 | 393 | Nguồn, schema, trường dữ liệu và giới hạn. |
| 3.2 Phân tích/tiền xử lý | 22–26 | 858 | Hình phân bố, số mô tả, rồi giải thích từng quyết định làm sạch. |
| 3.3 Xây dựng mô hình | 26–34 | khoảng 1.877 | Dataset/DataLoader, tuning, baseline, kiến trúc, huấn luyện và metric. |
| 3.4 Phân tích kết quả | 34–36 | 577 | Kết quả, so sánh, nguyên nhân, hạn chế và kết chương. |
| Chương 4 | 37–41 | khoảng 920 | Ảnh giao diện, luồng nghiệp vụ, kiểm thử và giới hạn triển khai. |
| Tài liệu tham khảo | 42–45 | 875 | 33 mục gồm paper, sách, tài liệu kỹ thuật và nguồn dữ liệu. |

Mẫu viết theo nhịp “định nghĩa → cơ chế → ví dụ → vai trò trong đề tài”. Công thức làm rõ mô hình/metric, không thay thế diễn giải. Chương thực nghiệm có mật độ hình/bảng cao hơn chương lý thuyết. Mỗi bảng kết quả có đoạn phân tích ngay sau; kết chương không đưa kết quả mới.

## Ánh xạ sang báo cáo mới

- Chương 1: bối cảnh, mục tiêu, đối tượng, phạm vi và phương pháp.
- Chương 2: phát hiện đối tượng, bbox/IoU, Precision–Recall–AP, YOLO11, Faster R-CNN, ResNet và FPN.
- Chương 3: ROD 25 lớp, chuỗi làm sạch, cấu hình E3/E4/E6, checkpoint, evaluator chung, VALID, phân tích lớp và benchmark.
- Chương 4: ứng dụng Gradio, adapter, ngưỡng tự động, kiểm thử và giới hạn an toàn.
- Mọi khẳng định thực nghiệm phải có dòng trong `FACTS.md`; phần thiếu bằng chứng ghi `[CHƯA CÓ]`, đặc biệt TEST và log huấn luyện E6 rời.
