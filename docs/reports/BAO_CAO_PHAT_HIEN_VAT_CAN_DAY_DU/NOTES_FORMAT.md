# Quyết định định dạng từ DOCX mẫu

Nguồn chỉ dùng để quan sát **format**: `docs/reports/12423TN_12423018_NguyenTranKien (2).docx`. Đã đọc bằng `python-docx` và trực tiếp `word/document.xml`, `word/styles.xml`, `word/header1.xml`, `word/footer1.xml` (gồm `sectPr`, trường TOC/PAGE/SEQ). Không sao chép nội dung môn học, hình/logo, tác giả, giáo viên hoặc danh mục tài liệu của mẫu.

| Thuộc tính | Quan sát trong DOCX | Quyết định trong khung XeLaTeX |
|---|---|---|
| Trang | A4 21 × 29,7 cm, một section; lề trên 2 cm, dưới 0,61 cm, trái 3 cm, phải 2,5 cm (`sectPr/pgSz`, `pgMar`). | Giữ đúng số lề trong `preamble.tex`. Lề dưới 0,61 cm rất hẹp và footer cách mép 1,4 cm; cần kiểm tra trực quan khi **người dùng** biên dịch trên Prism, chưa xác nhận không chồng lấp. |
| Font | `styles.xml` mặc định Calibri 12 pt, nhưng nhiều đoạn thân bài có direct formatting 13 pt; heading có direct formatting khác định nghĩa style. | Theo yêu cầu hiện tại, dùng **Times New Roman** nếu có (`C:\Windows\Fonts\times.ttf` đã kiểm tra), nếu không dùng **TeX Gyre Termes**. Thân bài 13 pt, dòng 19,5 pt. Không dùng Babel/Polyglossia/inputenc tiếng Việt. |
| Giãn dòng / đoạn | Nhiều đoạn thân bài trực tiếp 1,5; thụt đầu dòng khoảng 1,25 cm; canh đều. | 13/19,5 pt, `\parindent` 1,25 cm; Word có default spacing khác nhưng direct formatting thân bài được ưu tiên. Canh đều mặc định LaTeX. |
| Bìa | Trường/đơn vị ở trên cỡ 14, tên loại báo cáo cỡ 18, tên đề tài cỡ 14, metadata cỡ 13, thành phố/năm dưới cùng; trang đầu khác header/footer. | Bố cục giữa trang tương ứng. Chỉ đề tài hiện tại được điền; trường, người viết, GVHD, lớp, nơi/năm để placeholder. Không lấy logo của báo cáo môn khác. |
| Số trang | `sectPr/titlePg` bật; footer có trường `PAGE`, `pgNumType start=1`, header/footer đầu trang khác trang thường. | Bìa không hiện số; trang sau số Ả Rập ở giữa footer. Trạng thái số trang thực sau bìa cần người dùng kiểm tra trong PDF; chưa kiểm chứng bằng compile. |
| Header/footer | Header thường có tiêu đề báo cáo cũ; footer có trường PAGE. Header cách mép 0,8 cm, footer 1,4 cm. | Thay chữ header bằng tên đề tài hiện tại dạng rút gọn; footer số trang giữa. Không sao chép tiêu đề cũ. |
| Tiêu đề | Heading 1 thực tế cỡ 14 đậm, căn giữa, dạng `CHƯƠNG N: ...`; Heading 2/3 cỡ 13 đậm. Style base ghi kích thước lớn hơn nhưng direct formatting của chương thực tế là 14. | Dùng `titlesec`: chương 14 đậm giữa, mục/tiểu mục 13 đậm; đánh số chương và mục tự động. |
| Mục lục | Trường Word `TOC \\o "1-3"`; sau bìa và các phần đầu. | `\tableofcontents`, cấp chương/mục/tiểu mục; build hai lượt để cập nhật số trang. |
| Danh mục | Mẫu có danh mục các bảng trước danh mục hình ảnh, dùng trường Word TOC/SEQ; có danh mục thuật ngữ. | `\listoftables` rồi `\listoffigures`; danh mục thuật ngữ chưa có mục vì khung rỗng, bổ sung khi có thuật ngữ thực. |
| Caption | Đoạn Caption thực tế 11 pt, giữa; dạng `Hình 3. 1:` và `Bảng 3. 1:`. | Caption cỡ `small` (trên lớp 12 pt tương ứng 11 pt), căn giữa, nhãn `Hình/Bảng`, đánh số theo chương `3.1`. Bỏ khoảng trắng bất thường giữa dấu chấm và số để tránh số hiệu không chuẩn; ghi rõ đây là chuẩn hóa. |
| Trích dẫn | Thân bài dùng chỉ số trong ngoặc vuông, ví dụ `[3, 4, 5]`; cuối tài liệu có danh mục `[1] ...`, thứ tự số. | Dành cấu trúc cho trích dẫn số `[n]` theo thứ tự xuất hiện. `refs.bib` **rỗng** theo yêu cầu nên khung chưa có mục trích dẫn hay nguồn giả; sẽ nối BibTeX khi viết nội dung. |
| Phụ lục | Không thấy Heading phụ lục thực tế sau tài liệu tham khảo. | Chuẩn bị `\appendix` và `chapters/phu_luc.tex` rỗng; số chữ cái theo thông lệ khoa học khi có phụ lục thật. |

## Giới hạn kiểm tra

Người dùng chỉ định không biên dịch trong phiên này. Các file mới chỉ được kiểm tra tĩnh; **không có nhận định PDF đã tạo, font đã được XeLaTeX nhận, hoặc layout đạt**. Trước khi người dùng đổi yêu cầu, lệnh cài MiKTeX đã hoàn tất; không chạy XeLaTeX sau đó. `build.ps1` là hướng dẫn chạy do người dùng thực hiện trên Prism, chưa được agent chạy.

## Quyết định chuyển sang định dạng paper

- `main.tex` trỏ tới `PAPER_PHAT_HIEN_VAT_CAN.tex`; báo cáo theo chương cũ vẫn được giữ nguyên để truy vết.
- Dùng lớp `article`, A4, hai cột, 10 pt; bố cục gồm abstract, introduction, related work, data, method, experiments, results, application, limitations và conclusion.
- `paper_preamble.tex` dùng `newtxtext/newtxmath` với pdfLaTeX và TeX Gyre Termes với XeLaTeX/LuaLaTeX để tương thích Prism.
- Trích dẫn số dùng `natbib`, `plainnat` và `refs.bib`; không dùng biber, shell-escape, minted hoặc đường dẫn Windows.
- TEST đang chạy được giữ ở trạng thái chưa chốt; VALID không bị đổi nhãn thành TEST.
