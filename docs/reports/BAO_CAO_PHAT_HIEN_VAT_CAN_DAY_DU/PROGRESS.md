# Tiến độ báo cáo đầy đủ

## Đầu phiên 2026-10-12

- `PROGRESS.md` chưa tồn tại tại đường dẫn đầu ra; tạo mới để ghi tiến độ theo yêu cầu.
- Phạm vi chỉ là báo cáo LaTeX mới, không sửa báo cáo `.tex` cũ, dataset, checkpoint, mã huấn luyện hay TEST.

## Đã xong trong phiên 2026-10-12

- Đọc cấu hình, ứng dụng, mã evaluator, báo cáo freeze, kết quả VALID và benchmark CPU; kiểm tra nguồn khoa học gốc.
- Lập `FACTS.md` với đường dẫn nguồn cho mọi số thực nghiệm được dùng.
- Tạo bản thảo `BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU.tex`; giữ nguyên file `.tex` cũ.
- Kiểm tra 3 tệp VALID đều là `FULL`, cùng 3.318 ảnh, 5.320 GT và cùng các giá trị protocol (khác thứ tự khóa JSON). Checkpoint và data hashes trong kết quả cũng khớp lock.
- Kiểm tra log E3/E4 đều có 10 hàng epoch, hàng cuối là 10; xác minh tỷ lệ độ trễ từ JSON.
- Kiểm tra mọi ID bằng chứng trong bản thảo tồn tại ở `FACTS.md`; các môi trường LaTeX được dùng có số `begin/end` bằng nhau. Mọi đường dẫn nguồn có trong `FACTS.md` đều tồn tại sau khi sửa một biểu thức wildcard thành các đường dẫn cụ thể.

## Còn dở / bị chặn

- Chưa có kết quả đánh giá TEST cuối; không chạy TEST trong công việc này.
- Chưa kiểm chứng độc lập toàn bộ log/manifest huấn luyện E6 tại máy local.
- Máy hiện không tìm thấy `xelatex`, `lualatex`, `tectonic` hoặc `pdflatex` trên PATH. Chưa biên dịch PDF; cú pháp LaTeX mới chỉ được kiểm tra tĩnh.
- Git CLI báo thư mục hiện tại không phải repository Git; không dùng Git status làm bằng chứng thay đổi trong phiên này.

## Quyết định

- Dùng đúng kết quả evaluator chung trên VALID; không trộn với metric xuất từ framework riêng.
- Mô tả benchmark CPU là đo trên mẫu ảnh tĩnh, không chứng minh real-time.
- Không chạy đo lại, huấn luyện hay đánh giá TEST; tất cả thay đổi chỉ nằm trong thư mục báo cáo mới.

## Phiên điều tra 2026-10-12 — chỉ FACTS, không viết LaTeX

- Đã đọc README, configs, src, app, notebook E3/E4/E6, kết quả dataset/metrics/logs, `docs/references.md` và bản thảo `.tex` cũ theo phạm vi cần xác minh. Không chỉnh sửa bản thảo `.tex` nào.
- Đã chuyển `FACTS.md` sang bảng 5 cột theo yêu cầu, gồm các mục A–I và 24 dòng bằng chứng. Đếm trực tiếp tên ảnh Final: 18.151 TRAIN, 3.318 VALID, 1.611 TEST; không đọc ảnh/nhãn TEST.
- Đã đối chiếu E3/E4 CSV có 10 epoch, trong lịch 30; `close_mosaic` chưa kích hoạt. Đã kiểm tra tiêu chí best trong code Ultralytics local và E6 code.
- Đã chạy script CPU mới `figures/scripts/measure_complexity.py` bằng `.venv` để đo tham số và ước tính FLOPs/MACs từ ba checkpoint. JSON đầu ở `figures/scripts/complexity_20261012T084622Z.json` có YOLO FLOPs null do API `.info()` không trả giá trị; đã sửa script và chạy lại, kết quả dùng ở `figures/scripts/complexity_20261012T084711Z.json`. Không sửa checkpoint/dataset.
- Đã tạo `CONFLICTS.md` và `MISSING.md`. Đã kiểm tra các file bằng chứng V2/V3/V4/Final tồn tại; xác nhận bốn file log E6 rời vẫn thiếu. Script đo qua kiểm tra cú pháp AST.
- Còn thiếu: TEST cuối (không chạy trong phiên này), môi trường/log huấn luyện E6 đầy đủ, GPU VRAM/end-to-end và phân tích lỗi ảnh có duyệt. Không dùng FLOPs THOP E6 để xếp hạng trực tiếp với YOLO.
- Quyết định: giữ nguyên các LaTeX, không cập nhật nội dung báo cáo cho đến khi người dùng yêu cầu viết tiếp.

## Phiên định dạng khung từ DOCX — không biên dịch

- Đã đọc DOCX mẫu bằng `python-docx` và OOXML (`word/document.xml`, `styles.xml`, `sectPr`, `header1.xml`, `footer1.xml`); chỉ lấy format, không lấy nội dung môn học. Các quan sát và lựa chọn ghi trong `NOTES_FORMAT.md`.
- Đã tạo `main.tex`, `preamble.tex`, `cover.tex`, `frontmatter.tex`, `chapters/` với các file chỉ có comment, `refs.bib` rỗng và `build.ps1`. Không sửa bản thảo LaTeX cũ.
- Đã xác nhận Windows có file font Times New Roman; preamble ưu tiên Times New Roman và fallback TeX Gyre Termes. Chưa kiểm chứng XeLaTeX nhận font vì không biên dịch.
- Trước khi người dùng giới hạn lại nhiệm vụ, lệnh cài MiKTeX đã hoàn tất. Sau giới hạn mới, agent không chạy trình biên dịch hay build script.
- Đã chạy `python docs/reports/BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU/lint_static.py`: **STATIC LINT PASS; checked 12 required files; TeX not compiled**. Lượt lint đầu lỗi mã hóa console cp1252; đã điều chỉnh thông báo và chạy lại thành công.
- Còn dở: người dùng biên dịch trên Prism, xem PDF/layout thật, kiểm tra lề dưới hẹp của mẫu và log thiếu font/file. Không tuyên bố PDF đã có hoặc layout đạt.

## Phiên chuyển báo cáo sang cấu trúc paper 2026-10-13

- Đã giữ nguyên báo cáo theo chương và tạo paper mới tại `PAPER_PHAT_HIEN_VAT_CAN.tex`.
- Đã chuyển tài liệu tham khảo sang BibTeX số trong `refs.bib`; không thêm kết quả thực nghiệm chưa có bằng chứng.
- TEST E4/E6 đang chạy ở tiến trình riêng. Paper không đọc số tạm thời và vẫn ghi TEST chưa chốt.
- Không biên dịch LaTeX. Chỉ chạy kiểm tra tĩnh; người dùng biên dịch trên Prism.

## Cập nhật TEST và xuất Word 2026-10-13

- Đã xác minh ba kết quả `FINAL_LOCKED`: E3/E4/E6 cùng 1.611 ảnh và 2.354 GT.
- Đã cập nhật paper với bảng TEST tổng, TP/FP/FN, mAP và AP50–95 của 25 lớp; VALID được giữ làm kết quả lựa chọn checkpoint/ngưỡng.
- Đã tạo `BAO_CAO_PHAT_HIEN_VAT_CAN_PAPER.docx` bằng OpenXML từ DOCX mẫu để người dùng chỉnh sửa trong Word.
- Gói DOCX và XML đã kiểm tra tĩnh: 5 bảng, không còn token LaTeX thô. Không biên dịch LaTeX và chưa render trực quan DOCX vì Word COM nền bị treo, Python renderer không khả dụng trong phiên công cụ.
