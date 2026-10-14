# ROD V2 — Đánh giá các contact sheet Bike/Motorcycle (đề xuất, chưa được duyệt)

Nguồn: `contact_sheets.zip` do người dùng tải lên, gồm 8 sheet MOTORCYCLE, 4 UNCLEAR, 1 bicycle_sample. Đây là kết luận dựa trên bản contact sheet (ảnh thu nhỏ), không thay thế xem file gốc và bbox. Không có quyền sửa V2 và không ghi APPROVE thay người dùng.

| Hướng xử lý | Số object |
|---|---:|
| RELABEL_MOTORCYCLE | 153 |
| KEEP_BIKE | 30 |
| REVIEW_EBIKE_TAXONOMY | 24 |
| REMOVE_OR_REANNOTATE_MIRROR_BBOX | 8 |
| FIX_BBOX_AND_CLASS | 6 |
| FIX_BBOX_MULTIPLE_BICYCLES | 2 |
| REVIEW_RENDER_AND_BBOX | 1 |
| **Tổng** | **224** |

**Nhóm lỗi bbox:**

- `FIX_BBOX_AND_CLASS`: R020, R021, R027, R028, R205, R206
- `REMOVE_OR_REANNOTATE_MIRROR_BBOX`: R154, R158, R159, R162, R173, R314, R315, R368
- `FIX_BBOX_MULTIPLE_BICYCLES`: R268, R330
- `REVIEW_RENDER_AND_BBOX`: R279

**Nhóm đã phân loại BICYCLE mẫu:** 20/20 nhìn giống xe đạp, không thấy xe máy rõ ràng trong sheet.

**Lưu ý quan trọng:** Xe đạp trợ lực điện với bàn đạp chưa có quy tắc nhất quán giữa Bike và Motorcycle; không tự relabel các trường hợp REVIEW_EBIKE_TAXONOMY. Những nhóm FIX_BBOX cần xem trực tiếp ảnh gốc và annotation. Không đánh dấu các đề xuất này là APPROVED.
