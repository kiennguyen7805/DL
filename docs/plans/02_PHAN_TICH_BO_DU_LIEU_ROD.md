# 02 — PHÂN TÍCH BỘ DỮ LIỆU ROD
## Real-Time Obstacle Detection Dataset — nguồn, cấu trúc, phân tích, quyết định 25 class vs subset, và điều kiện trước training

> **CẬP NHẬT DATASET 2026-10-10 — NGUỒN HIỆN HÀNH.**  
> Các phần V1/converter phía dưới là hồ sơ lịch sử. Mọi training mới phải dùng duy nhất `data/processed/rod_detection_final/` (ROD Final A4), không chạy lại converter, không đổi split và không sửa annotation.

## 0. ROD Final A4 đã freeze

| Split | Ảnh | Label | Object | Vai trò |
|---|---:|---:|---:|---|
| TRAIN | 18.151 | 18.151 | 30.793 | cập nhật trọng số |
| VALID | 3.318 | 3.318 | 5.320 | chọn checkpoint/ngưỡng và quyết định kỹ thuật đã đăng ký |
| TEST | 1.611 | 1.611 | 2.354 | đánh giá cuối |
| **Tổng** | **23.080** | **23.080** | **38.467** | **25 lớp** |

Bằng chứng: `results/lam_sach_du_lieu/final/a4_report.md`, `verify_report.json`, `class_counts_v4_final.csv`, `quarantined_images.csv` và `data/processed/rod_detection_final/final_build.json`.

Các gate đã PASS:

- 25/25 class còn hiện diện, thứ tự ID giữ nguyên;
- 0 ảnh thiếu label, 0 label thiếu ảnh;
- 0 lỗi cú pháp, class ID hoặc hình học bbox;
- 0 exact duplicate xuyên split;
- TEST không đổi so với V4, 3.222 file ảnh/label đã đối chiếu checksum;
- 178 cặp ảnh/label TRAIN/VALID đã cách ly theo manifest; không xóa TEST.

Giới hạn phải đưa vào báo cáo:

- còn 271 nhóm exact duplicate trong cùng split;
- 18 bbox và các đề xuất Bike/Motorcycle chưa được phê duyệt không được áp dụng;
- một số bbox nhỏ/trùng trong TRAIN được giữ như noise nhãn;
- near-duplicate không được xem là đã loại hết chỉ vì exact hash bằng 0;
- bất đồng clipping 1.328 so với 190 thuộc audit converter lịch sử, không được diễn giải là 1.328 lỗi còn tồn tại trong Final.

## 0.1. Taxonomy cố định

```text
0 Bike                 1 Building             2 Car
3 Person               4 Stairs               5 Traffic sign
6 Electrical Pole      7 Road                 8 Motorcycle
9 Dustbin             10 Dog                 11 Manhole
12 Tree               13 Guard rail          14 Pedestrian crosswalk
15 Truck              16 Bus                 17 Bench
18 Traffic Cone       19 Fire hydrant        20 Teraffic Barrel
21 Plant Pot          22 Electrical Box      23 Chair
24 Bicycle Rack
```

Giữ nguyên chính tả `Teraffic Barrel` vì đây là tên class đã freeze. Không remap hoặc tạo subset cho nghiên cứu hiện hành.

## 0.2. Quy tắc dùng dữ liệu cho ba mô hình

- Chỉ E3 YOLO11n, E4 YOLO11s và E6 Faster R-CNN ResNet-50 FPN V2, đều COCO Pretrained, được fine-tune trong phạm vi hiện hành. E5 chỉ còn là lịch sử.
- Cả ba phải đọc cùng ảnh và cùng bbox từ ROD Final.
- Adapter Faster R-CNN chỉ chuyển biểu diễn YOLO normalized `(xc,yc,w,h)` thành pixel `(x1,y1,x2,y2)` trong dataloader; không ghi ngược vào dataset.
- Mapping class của Torchvision cần dành index 0 cho background và ánh xạ ROD `0..24` thành model `1..25`; khi đánh giá phải map ngược và có unit test round-trip.
- `data.yaml` Kaggle được tạo trong `/kaggle/working`; không sửa YAML frozen trong input.
- Không dùng TEST để chọn augmentation, hyperparameter, checkpoint, confidence/NMS hay logic distance/warning.

## 0.3. Kiểm tra trước mỗi run

```text
[ ] final_build.json: version=ROD_FINAL_A4, frozen=true, verified=true
[ ] count đúng 18.151 / 3.318 / 1.611
[ ] 25 tên class và thứ tự ID đúng
[ ] fingerprint dataset/config ghi vào run manifest
[ ] train loader chỉ đọc TRAIN; validation chỉ đọc VALID
[ ] TEST path không xuất hiện trong training args
[ ] output nằm ngoài dataset read-only
```

## 0.4. Chuẩn đánh giá chung

Mỗi detector phải xuất prediction về một schema chung: `image_id,class_id,confidence,x1,y1,x2,y2`. Một evaluator chung tính Precision, Recall, F1 ở ngưỡng chọn trên VALID, mAP50, mAP50-95, AP/Recall từng lớp và FN. Với Faster R-CNN, không dùng metric riêng khác chuẩn rồi so trực tiếp với Ultralytics mà không có evaluator chung.

Dataset không có ground-truth khoảng cách. Đánh giá distance phải dùng một bộ calibration/evaluation riêng do dự án đo, tách khỏi ROD và không thay đổi split ROD.

---

## PHẦN LỊCH SỬ V1/V2 BÊN DƯỚI

Giữ để truy vết converter và quá trình làm sạch. Các đường dẫn `rod_detection/`, quyết định subset và checklist “chưa verify” không còn là lệnh thực thi hiện hành.

> `data/processed/rod_detection/` **đã được tạo** bằng `src/prepare_rod_detection.py`. Số trong `conversion_summary.json` là thống kê do converter ghi, **chưa phải kết quả xác minh độc lập**. Số cuối cùng đưa vào báo cáo chỉ lấy sau khi inspect, verify và kiểm tra trực quan trên dataset thật.

---

# 1. Nguồn dữ liệu

```text
Kaggle:       https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset
Tên Kaggle:   abtinzandi/obstacle-detection-dataset
Bản public:   https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO
```

Việc cần làm khi bắt đầu (ghi vào `results/day1/dataset_source.md`):

- ghi lại URL, ngày tải, phiên bản/thời điểm cập nhật hiển thị trên trang;
- ghi lại **license** và thông tin tác giả hiển thị trên trang (không tự đoán);
- ghi lại dung lượng file tải về.

Bản Hugging Face chỉ để tham khảo/đối chiếu. Nếu có khác biệt giữa hai nguồn thì ghi nhận và **chọn một nguồn duy nhất** cho toàn bộ thí nghiệm (mặc định: bản Kaggle vì training chạy trên Kaggle).

---

# 2. Mục đích và mức độ phù hợp

Dataset phục vụ phát hiện vật cản trong môi trường ngoài trời/đường phố, có annotation YOLO TXT hỗn hợp (bounding box và polygon), nhiều class, và đã chia sẵn split. Phù hợp vì:

- là bài toán Object Detection đúng với đề tài;
- annotation raw là YOLO TXT hỗn hợp; converter **đã chuyển** polygon thành bounding box trong bản processed, nay cần verify kết quả trước training;
- có nhiều class liên quan tới di chuyển (người, phương tiện, cột điện, cầu thang, nắp cống, lan can, cọc...);
- có `data.yaml` và split sẵn, giảm rủi ro sai lệch.

Các điểm **cần cảnh giác** (phải kiểm tra bằng dữ liệu, không giả định):

| Vấn đề | Cách kiểm tra |
|---|---|
| Class như `Road`, `Building` có bbox rất lớn, có thể chi phối mAP tổng và không hẳn là "vật cản" | thống kê area ratio theo class, đọc AP per class |
| `Bike`, `Bicycle`, `Motorcycle`, `Bicycle Rack` dễ nhầm lẫn/định nghĩa chồng | vẽ ảnh mẫu từng class, đọc nhãn thực tế |
| Dữ liệu "real-time" có thể là frame liên tiếp từ video → ảnh gần trùng giữa train/valid/test (rủi ro rò rỉ) | hash MD5 tìm ảnh trùng hoàn toàn giữa split; xem thủ công mẫu ảnh gần giống; ghi vào báo cáo như giới hạn nếu không loại trừ được |
| Class hiếm quá ít mẫu | `class_counts.csv` |
| Ảnh không nhãn (background) | thống kê empty label |

---

# 3. Cấu trúc dữ liệu hiện có

```text
data/raw/archive/ROD-Dataset/dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Cấu trúc thực tế có thể khác (thư mục lồng nhau, tên khác). Vì vậy script **không hard-code** đường dẫn: phải tìm `data.yaml` rồi suy ra gốc dataset.

Annotation YOLO TXT gốc, mỗi dòng một object:

```text
class_id x_center y_center width height     (bounding box, tọa độ normalize 0–1)
class_id x1 y1 x2 y2 ... xN yN              (polygon, tọa độ normalize 0–1)
```

Theo `data/processed/rod_detection/conversion_summary.json`, converter đã ghi 35.057 box và 5.138 polygon trong tổng 40.195 object, đã chuyển polygon thành box bao ngoài. Bản detection hiện ở `data/processed/rod_detection/`; **kiểm tra lại** các nhãn và không chạy lại converter vào thư mục output này.

Cùng file summary ghi `boxes_clipped` train/valid/test lần lượt **1.053/203/72** (tổng 1.328); `src/prepare_rod_detection.py` clip các box nguồn tràn biên trong tolerance nhỏ. Đây là **log của converter, chưa xác minh** clip đúng trên ảnh và chưa kiểm đếm độc lập. Visual review nên lấy mẫu cả các ảnh có nhãn từng bị clip và polygon đã đổi thành bbox. **Phải kiểm tra lý do và mức clip** (sai số số học nhỏ hay nhãn sai lớn); nếu nhãn bị sửa đáng kể thì không tự động coi là sạch.

---

# 4. Thông tin từ YAML/summary đã có, cần kiểm đếm độc lập

| Hạng mục | Mô tả ban đầu | Giá trị thực tế |
|---|---|---|
| Số ảnh | khoảng 24.326 | Converter ghi 24.326 (train 19.186, valid 3.511, test 1.629); **cần kiểm đếm độc lập** |
| Số object | khoảng 40.195 | Converter ghi 40.195 = 35.057 box + 5.138 polygon; **cần verify** |
| Số class | 25 | `data/processed/rod_detection/data.yaml` khai báo `nc: 25`; cần đối chiếu label ID |
| Split | train/valid/test | Có ba thư mục và YAML tương ứng; ảnh–nhãn ghép cặp và overlap **cần xác minh** |
| Format | YOLO TXT | Converter tạo bản detection 5 trường; **cần xác minh từng nhãn** |
| `data.yaml` | có | Có file thật, nhưng `path` đang là đường dẫn Windows local, phải tạo YAML portable trên Kaggle |

Nếu số thực tế khác mô tả: dùng số thực tế và ghi chú sự khác biệt.

---

# 5. Danh sách class theo `data/processed/rod_detection/data.yaml`

```text
Bike, Building, Car, Person, Stairs, Traffic sign, Electrical Pole, Road,
Motorcycle, Dustbin, Dog, Manhole, Tree, Guard rail, Pedestrian crosswalk,
Truck, Bus, Bench, Traffic Cone, Fire hydrant, Teraffic Barrel, Plant Pot,
Electrical Box, Chair, Bicycle Rack
```

**Quy tắc bắt buộc:**

- **không** giả định class ID theo thứ tự trên;
- `names` trong `data.yaml` thực tế là **nguồn duy nhất** của class mapping;
- script xuất `results/day1/class_mapping.csv` (`class_id, class_name`) trực tiếp từ `data.yaml`;
- kiểm tra tên trong `data.yaml` có khớp danh sách mô tả không (khác biệt về chữ hoa/thường, tên, số lượng đều phải ghi lại);
- nếu `names` ở dạng list hay dict đều xử lý được.

---

# 6. Split

Dataset đã có `train / valid / test` → **giữ nguyên, không tự chia lại** (trừ khi phát hiện lỗi nghiêm trọng như overlap; khi đó ghi lý do và dừng để quyết định).

| Split | Vai trò |
|---|---|
| train | cập nhật trọng số |
| valid | validation trong lúc train, chọn `best.pt`, tuning, chọn ngưỡng confidence |
| test | **chỉ đánh giá cuối** |

Test **không** được dùng để: chọn model, tune hyperparameter, chọn confidence threshold, chọn checkpoint, hay quyết định giữ/bỏ class. Mọi quyết định về subset (nếu có) chỉ dựa trên **train** (và valid nếu cần), tuyệt đối không nhìn kết quả test.

Kiểm tra bắt buộc: không trùng tên file giữa các split, và không trùng nội dung ảnh (hash) giữa các split.

---

# 7. `data.yaml`

Cần đọc và ghi lại:

```text
path (nếu có)
train / val (hoặc valid) / test
nc (nếu có)
names
```

Lưu ý phổ biến: `data.yaml` gốc thường chứa đường dẫn tương đối hoặc đường dẫn không hợp lệ trên máy khác. Trên Kaggle `/kaggle/input` là read-only. Cách xử lý:

- dùng `data/processed/rod_detection/data.yaml` đã có sau khi verify; trên Kaggle tạo **bản YAML portable riêng** trong `/kaggle/working` trỏ tới dataset đã upload/Add Input, không ghi đè bản local;
- không sửa dataset gốc;
- Ultralytics ghi cache label (`labels.cache`) cạnh thư mục labels; nếu thư mục read-only, nó cảnh báo và tiếp tục nhưng chậm hơn. Cách gọn: copy dataset sang `/kaggle/working` nếu dung lượng cho phép (kiểm tra `du -sh` trước), hoặc chấp nhận cảnh báo.

---

# 8. Script phân tích: `src/01_inspect_rod.py`

Input: đường dẫn dataset (hoặc tự tìm `data.yaml` dưới thư mục cho trước).
Output: `results/day1/` và `results/lam_sach_du_lieu/analysis/`.

## 8.1. Thống kê cần có

```text
total images / train images / valid images / test images
total label files
total objects
objects per class (theo split và tổng)
images per class (theo split và tổng)
objects per image (phân bố: 0, 1, 2, 3-5, 6+ ...)
class imbalance (tỷ lệ max/min, phần trăm class top-3)
bbox width / height (mean, median, percentile)
bbox area ratio = (w*h) (chuẩn hóa) — phân bố theo class
empty label (file nhãn rỗng)
missing label (ảnh không có file nhãn)
missing image (file nhãn không có ảnh)
invalid annotation
unknown class ID
ảnh trùng giữa split (hash)
kích thước ảnh (w×h) — phân bố
```

## 8.2. File output

```text
archive/day1_environment/environment.json
results/day1/dataset_source.md
results/day1/class_mapping.csv
results/day1/dataset_summary.json
results/lam_sach_du_lieu/analysis/class_counts.csv
results/lam_sach_du_lieu/analysis/objects_per_image.csv
results/lam_sach_du_lieu/analysis/bbox_stats.csv
results/lam_sach_du_lieu/analysis/split_overlap.json
results/lam_sach_du_lieu/analysis/fig_class_distribution.png
results/lam_sach_du_lieu/analysis/fig_images_per_class.png
results/lam_sach_du_lieu/analysis/fig_objects_per_image.png
results/lam_sach_du_lieu/analysis/fig_bbox_area_ratio.png
```

`class_counts.csv` cột gợi ý:

```text
class_id, class_name, train_objects, valid_objects, test_objects,
train_images, valid_images, test_images, median_area_ratio
```

## 8.3. Kích thước object — quy ước của project

Ultralytics/COCO định nghĩa small/medium/large theo pixel gốc. Ở đây dùng `area_ratio` chuẩn hóa. Ngưỡng chia (ví dụ small < 1%, medium 1–10%, large > 10% diện tích ảnh) là **quy ước của project**, phải ghi rõ trong báo cáo và không nói là chuẩn COCO. Có thể xem phân phối thực tế trước rồi chọn ngưỡng, nhưng phải chọn **trước** khi xem kết quả model.

---

# 9. Xác minh annotation: `src/02_verify_rod.py`

## 9.1. Kiểm tra tự động toàn bộ label (mọi split)

Với **dữ liệu gốc**, chấp nhận 5 trường `class xc yc w h` hoặc `class x1 y1 ... xN yN` (N ≥ 3). Phân biệt rõ **tọa độ thô nằm ngoài [0,1]** với lỗi của dữ liệu processed: converter đã log các trường hợp box cần clip, nên phải kiểm tra mức tràn biên, tolerance và số lượng clip, không âm thầm cho PASS hoặc coi toàn bộ raw là hỏng chỉ vì sai số nhỏ. Polygon phải có diện tích bao ngoài dương. Sau chuyển đổi, mỗi dòng của **bản detection** phải có đúng 5 trường và tọa độ/hộp phải hợp lệ trong [0,1] (có epsilon số học đã khai báo). Kiểm tra:

```text
class_id là số nguyên, thuộc [0, nc-1]
0 <= xc <= 1
0 <= yc <= 1
0 <  w  <= 1
0 <  h  <= 1
bbox không vượt ra ngoài ảnh: xc - w/2 >= -eps và xc + w/2 <= 1 + eps (tương tự y)
bbox không bất hợp lý (diện tích cực nhỏ, ví dụ w*h nhỏ hơn ngưỡng ghi rõ trong báo cáo)
không có dòng trùng lặp hoàn toàn
mỗi ảnh có file nhãn tương ứng (file nhãn rỗng được phép nhưng phải đếm)
mỗi file nhãn có ảnh tương ứng
ảnh đọc được (mở bằng PIL/cv2, không hỏng)
```

Kết quả: `results/lam_sach_du_lieu/verify/verify_report.json` gồm số lỗi theo từng loại, và **danh sách file lỗi** (`bad_labels.csv`).

**PASS/FAIL tự động:**

- Lỗi nghiêm trọng (sai số trường theo cả hai định dạng, class ngoài phạm vi, tọa độ ngoài [0,1], box không có diện tích, ảnh hỏng): xử lý bằng cách ghi rõ. Nếu số lỗi rất nhỏ và cô lập, có thể tạo **bản sao đã lọc** (không sửa gốc) và ghi vào báo cáo số lượng đã loại; nếu lỗi lan rộng → FAIL.
- Script trả `exit code != 0` khi FAIL.

Ngưỡng "rất nhỏ" nên đặt trước khi chạy (ví dụ ≤ 0.1% file; con số này là quy ước dự kiến của project, có thể chỉnh nhưng phải chốt trước khi xem kết quả).

## 9.2. Kiểm tra trực quan — bắt buộc

Vẽ bbox lên **20–50 ảnh** với tên class thật, lưu `results/lam_sach_du_lieu/verify/viz/`. Cách chọn ảnh (seed cố định):

- ảnh có nhiều object nhất;
- ảnh chứa class hiếm (mỗi class hiếm ít nhất 1–2 ảnh);
- object nhỏ;
- cảnh đông, nền phức tạp;
- mỗi class trong danh sách "liên quan di chuyển" ít nhất một ảnh;
- một số ảnh từ **mỗi** split (train/valid/test);
- ảnh Bike/Bicycle/Motorcycle để đối chiếu định nghĩa.

Ghi kết quả xem vào `results/lam_sach_du_lieu/verify/visual_review.md`: mỗi ảnh xem xong ghi OK/lệch/nghi ngờ. Nếu bbox bị **lệch** (không bám vật thể, bị lật trục x/y, sai tỉ lệ, sai class rõ ràng):

```text
STOP — KHÔNG train chính thức cho tới khi hiểu nguyên nhân
```

## 9.3. Kiểm tra ảnh trùng/rò rỉ giữa split

Hash MD5 từng ảnh, tìm ảnh trùng hoàn toàn giữa train/valid/test. Kết quả `split_overlap.json`. Trùng hoàn toàn giữa test và train là lỗi cần xử lý/ghi rõ. Ảnh gần giống (frame liên tiếp) không phát hiện được bằng MD5; ghi vào giới hạn.

---

# 10. Quyết định: 25 class hay subset

Chốt **sau** khi có số liệu thật từ Ngày 2. Mặc định là **Phương án A** trừ khi gate bên dưới buộc chuyển sang B.

## 10.1. Phương án A — dùng toàn bộ 25 class (ưu tiên)

Ưu: không filter/remap, ít rủi ro sai label, tận dụng toàn bộ dữ liệu, triển khai nhanh, mAP theo class đầy đủ.
Nhược: có thể có class rất hiếm làm AP không ổn định; Road/Building có thể làm mAP tổng khó diễn giải; thời gian train lâu hơn nếu ảnh nhiều.

## 10.2. Phương án B — subset 8–12 class

Ứng viên tham khảo (**không** phải danh sách chốt): Person, Car, Bike, Motorcycle, Stairs, Electrical Pole, Dustbin, Manhole, Guard rail, Traffic Cone.

Chỉ chọn B khi gate bên dưới kích hoạt. Không chốt subset bằng cảm tính.

## 10.3. Bảng dữ liệu để ra quyết định (điền từ `class_counts.csv`)

| class_id | class_name | train_obj | train_img | valid_obj | median_area_ratio | liên quan di chuyển? (Có/Không/Cân nhắc) | đủ dữ liệu? |
|---|---|---|---|---|---|---|---|
| (điền từ dataset thật) | | | | | | | |

## 10.4. Tiêu chí đề xuất (quy ước project, chốt trước khi xem kết quả model)

1. **Gate thời gian/GPU:** đo thời gian/epoch và VRAM trên Kaggle GPU (smoke với fraction nhỏ; nếu quota cho phép, đo thêm một epoch với toàn train để hiệu chỉnh ngoại suy), rồi dự trù **HPO + 2 × epochs_final × thời gian/epoch + validation, overhead, backup**. Chốt `epochs_final` từ ngân sách trước E3/E5; **30 chỉ là ứng viên**, không phải mặc định. Nếu vượt quota: thu hẹp HPO trước, điều chỉnh `imgsz`/batch/epoch cho **cả hai** final runs theo một config chung. Chỉ bớt class nhưng vẫn giữ nguyên số ảnh không bảo đảm giảm đáng kể thời gian training; không dùng subset như một lối tắt GPU nếu chưa đo.
2. **Gate dữ liệu:** class có quá ít object/ảnh ở train (ngưỡng chốt trước, ví dụ vài chục) được đánh dấu "không đủ dữ liệu để kết luận"; trong A vẫn giữ nhưng báo cáo nói rõ AP class đó không đáng tin; trong B có thể loại.
3. **Gate liên quan:** class có ý nghĩa bài toán (vật cản/nguy cơ cho người đi bộ) được ưu tiên.
4. **Gate nhất quán nhãn:** class nhãn mơ hồ/định nghĩa chồng (kết quả từ visual review) cân nhắc gộp hoặc loại — nhưng gộp class là thay đổi nhãn, cần lý do và ghi rõ.
5. Quyết định phải ghi vào `results/lam_sach_du_lieu/decision_25_vs_subset.md` với số liệu train/valid dẫn chứng, **không** dùng test.

## 10.5. Nếu dùng subset: `src/04_make_subset.py`

- đọc `data/processed/rod_detection/data.yaml` của bản đã chuyển sang box, nhận danh sách class giữ lại (theo **tên**, không theo ID);
- lọc annotation, chỉ giữ dòng thuộc class được chọn;
- remap `class_id` mới **liên tục từ 0**;
- xử lý ảnh chỉ còn nhãn rỗng sau lọc (quyết định giữ làm background hay bỏ; ghi rõ, áp dụng thống nhất mọi split);
- áp dụng cho cả train/valid/test theo cùng bảng mapping;
- tạo `data_subset.yaml` riêng;
- lưu bảng:

```text
old_class_id, old_class_name, new_class_id, new_class_name
```

→ `results/lam_sach_du_lieu/subset/class_remap.csv`

- **verify lại** toàn bộ bằng `02_verify_rod.py` trên dataset mới (class_id trong [0, nc_mới-1]) và visual verify lại 20+ ảnh;
- lưu `subset_report.json` (số ảnh/object trước–sau).

Nếu dùng subset thì **tất cả** experiment (E3, E5, ...) phải dùng cùng subset.

---

# 11. Smoke test

Chạy **sau** khi verify PASS và đã có quyết định 25 vs subset.

Chạy trên **Kaggle GPU**, đo VRAM và thời gian/epoch để xác định ngân sách cho HPO và hai final runs. Sau Smoke Test, làm HPO trên TRAIN/VALID và freeze một cấu hình chung vào `configs/final_experiment.yaml` **trước** E3/E5. Xem [nguồn khoa học và tài liệu kỹ thuật](../references.md).

```text
YOLO11n pretrained, 1–2 epoch, fraction nhỏ (ví dụ 0.05–0.1), cùng data.yaml định dùng
```

Mục đích: kiểm tra chuỗi `data.yaml → dataloader → train → validation → best.pt`.

PASS khi:

- dataloader không lỗi, không cảnh báo label nghiêm trọng (corrupt/ignored labels);
- model train được (loss là số hữu hạn, không NaN);
- có `best.pt` và `results.csv`;
- validation chạy xong và ghi được các metric;
- ghi được thời gian/epoch và VRAM ước tính, để dùng cho gate thời gian.

**Smoke test không được dùng làm kết quả experiment chính** và phải lưu ở thư mục riêng (`results/lam_sach_du_lieu/smoke/`) tách khỏi `results/runs/`.

## 11.1. Ranh giới giữa smoke, HPO và final runs

| Bước | Mục đích | Output | Được dùng TEST? |
|---|---|---|---|
| Smoke | chứng minh pipeline chạy, đo sơ bộ GPU | `results/lam_sach_du_lieu/smoke/smoke_profile.json` | Không |
| HPO pilot | khảo sát ít cấu hình trên TRAIN/VALID | `results/hparam/hparam_results.csv`, `hparam_decision.md` | Không |
| E3/E5 final | so initialization trong cùng ngân sách | checkpoints, `args.yaml`, run metadata | Chỉ sau khi freeze, đánh giá cuối |

**Kiểm tra bàn giao lên Kaggle:** bản detection `data/processed/rod_detection/` là nguồn ưu tiên; xác minh số ảnh/label/object từng split, `names`/`nc`, checksum `data.yaml` và phiên bản dataset. Trên Kaggle không được chọn `data.yaml` đầu tiên bằng `rglob` khi có nhiều candidate; phải in danh sách, chọn đúng root, kiểm tra và tạo YAML portable ở `/kaggle/working`. Không đưa local `data.yaml` chứa `path` Windows vào thẳng training trên Linux.

**Quy tắc PASS trước HPO:** đã review bằng mắt 20–50 ảnh, xác nhận split không leak nghiêm trọng, converter được kiểm thử; nếu chỉ có kiểm tra tự động mà chưa review bằng mắt, ghi PENDING và không đánh dấu PASS hoàn toàn. Cơ sở HPO: [Bergstra & Bengio 2012, Li et al. 2018](../references.md); giới hạn của pilot ngắn phải ghi rõ.

---

# 12. Điều kiện PASS trước training (checklist)

```text
[ ] Dataset tìm thấy, gốc xác định đúng
[ ] data.yaml đọc được, class_mapping.csv sinh từ data.yaml thật
[ ] Số ảnh/label/object thống kê từ script (không dùng số mô tả)
[ ] verify tự động PASS (hoặc lỗi cô lập đã xử lý và ghi rõ)
[ ] Không trùng ảnh hoàn toàn giữa các split (hoặc đã ghi rõ)
[ ] Visual verification 20–50 ảnh PASS
[ ] class_counts.csv + biểu đồ đã có
[ ] Quyết định 25 class vs subset đã ghi (không dùng test)
[ ] Smoke test PASS
[ ] Ngân sách HPO và hai final runs đã ước tính từ thời gian/epoch và VRAM thực đo
[ ] Trước E3/E5: HPO trên TRAIN/VALID hoàn tất, cấu hình chung đã freeze (không dùng TEST)
```

Thiếu bất kỳ mục nào → **không** bắt đầu E3/E5.

---

# 13. Dữ liệu đưa vào báo cáo (Chương 3)

- nguồn, license, ngày tải;
- cấu trúc, split, số ảnh/label/object **thực tế**;
- bảng class và phân phối;
- phân phối object/ảnh, kích thước bbox;
- phân tích class imbalance và ảnh hưởng dự kiến tới AP per class;
- kết quả verify, ví dụ ảnh visual verification;
- quyết định 25 class vs subset và lý do bằng số liệu;
- giới hạn (rò rỉ frame liên tiếp, class không hẳn là vật cản...).

---

# 14. Prompt sinh code phân tích/verify

```text
Bạn là kỹ sư Deep Learning.

Dataset: ROD gốc (YOLO TXT hỗn hợp box/polygon); bản detection đã có ở `data/processed/rod_detection/`. Cấu trúc raw hiện có `data/raw/archive/ROD-Dataset/dataset/` với data.yaml, train/valid/test
mỗi split có images/ và labels/. Không giả định class ID; đọc names từ data.yaml.
Không sửa dataset gốc. Không hard-code đường dẫn.

Hãy tạo:
1. src/01_inspect_rod.py
   - tự tìm data.yaml (argparse --root), kiểm tra cấu trúc
   - xuất class_mapping.csv, dataset_summary.json
   - thống kê: ảnh/label/object mỗi split, objects/class, images/class,
     objects/image, bbox w/h/area_ratio, empty/missing label/missing image,
     hash MD5 tìm ảnh trùng giữa split
   - xuất class_counts.csv và các biểu đồ PNG
2. src/02_verify_rod.py
   - kiểm tra box 5 trường hoặc polygon 1 + 2N trường (N>=3), class hợp lệ, tọa độ [0,1],
     bbox trong ảnh, dòng trùng, ảnh đọc được
   - xuất verify_report.json + bad_labels.csv
   - vẽ N ảnh (--n 20..50, seed cố định) ưu tiên nhiều object, class hiếm, object nhỏ
   - exit code != 0 nếu FAIL

Yêu cầu: argparse, pathlib, seed cố định, log rõ ràng, không thư viện lạ ngoài
ultralytics/pandas/matplotlib/opencv/pillow/pyyaml, chạy được cả local và Kaggle.
Kèm lệnh chạy và điều kiện PASS/FAIL.
```
