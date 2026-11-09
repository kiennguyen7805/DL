"""Package the verified current E3/E4/E6 evidence for report editing on the web."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/30epochs'
OUT = ROOT / 'output'
STAMP = '20261028'
MODELS = {'E3': 'YOLO11n', 'E4': 'YOLO11s', 'E6': 'Faster R-CNN ResNet-50 FPN V2'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()

def main():
    reports = {(m,s):read(RESULTS/m/s/'metrics.json') for m in MODELS for s in ['valid','test']}
    summary = read(RESULTS/'summary_E3_E4_E6.json')
    e6 = reports['E6','test']
    threshold = e6['operating_threshold']
    assert threshold == 0.6
    assert read(RESULTS/'E6/checkpoint/manifest.json')['is_complete'] is True
    table = '| Mô hình | Split | Epoch run | Ảnh | Ngưỡng | mAP50 | mAP50–95 | Precision | Recall | F2 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in summary:
        table += f"| {row['model']} — {MODELS[row['model']]} | {row['split'].upper()} | 30 | {row['images']} | {row['operating_threshold']:.2f} | " + ' | '.join(f'{row[k]:.6f}' for k in ['map50','map50_95','precision','recall','f2']) + ' |\n'
    class_names = {int(r['class_id']):r['class_name'] for r in csv.DictReader((RESULTS/'E6/test/per_class.csv').open(encoding='utf-8'))}
    classes = []
    winners = {m:0 for m in MODELS}
    class_table = '| Lớp | GT | AP50 | AP50–95 | TP | FP | FN |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for item in e6['per_class']:
        cls = item['class_id']
        comparison = {'class_id':cls,'class_name':class_names[cls]}
        for model in MODELS:
            comparison[model+'_AP50_95'] = reports[model,'test']['per_class'][cls]['ap50_95']
        maximum = max(comparison[m+'_AP50_95'] for m in MODELS)
        leads = [m for m in MODELS if abs(comparison[m+'_AP50_95']-maximum)<1e-12]
        comparison['leaders'] = ','.join(leads)
        for model in leads:
            winners[model] += 1
        classes.append(comparison)
        class_table += f"| {class_names[cls]} | {item['gt_count']} | {item['ap50']:.6f} | {item['ap50_95']:.6f} | {item['tp']} | {item['fp']} | {item['fn']} |\n"
    train_path = RESULTS/'E6/training_30_complete/metrics.csv'
    train = list(csv.DictReader(train_path.open(encoding='utf-8',newline='')))
    train_table = '| Epoch | mAP50 trong log | mAP50–95 trong log | mAR@100 | Loss |\n|---|---:|---:|---:|---:|\n'
    for row in train[-3:]:
        train_table += '| ' + row['epoch'] + ' | ' + ' | '.join(f'{float(row[k]):.6f}' for k in ['map50','map5095','mar100','loss_mean']) + ' |\n'
    comparison_path=OUT/'SO_SANH_THEO_LOP_TEST_E3_E4_E6.csv'
    save(comparison_path,csv_text(classes))
    delta = {key:(reports['E4','test'][key]-e6[key])*100 for key in ['precision','recall','f2','map50','map50_95']}
    prompt = '''Hãy cập nhật báo cáo nguồn Paper_ACL_E6_30Epoch_BanMoi_27102026.docx và PDF trong gói này bằng toàn bộ kết quả đã xác minh trong KET_QUA_MOI_E3_E4_E6.md. Đây là bản cập nhật mới sau khi đã chạy đủ VALID và TEST cho best checkpoint E6 của run hoàn tất 30 epoch; thay thế các ghi chú cũ “E6 chưa đánh giá lại”, “E6 chỉ 10 epoch” hoặc “E6 dừng ở 28/30”. E3/E4/E6 đều đã hoàn tất run 30 epoch; bảng so sánh hiện dùng các best checkpoint được chọn trong các run đó. Không tự khẳng định best checkpoint của cả ba đều xuất hiện ở epoch 30: riêng log E6 cho biết best tại epoch 30.

Dùng đúng bảng sáu dòng VALID/TEST trong file Markdown và CSV. Ngưỡng E3/E4/E6 lần lượt là 0,27/0,24/0,60. Mỗi ngưỡng được chọn bằng micro-F2 trên VALID rồi khóa trước TEST, với khóa riêng cho từng mô hình. Không nói có một khóa chung được tạo trước ba lần TEST: app/evaluation_lock.json chỉ là chỉ mục local nhập sau đánh giá, tham chiếu các khóa gốc. Thay cả ngưỡng E6 0,57/0,59 lẫn mọi số E6 cũ trong phần kết quả hiện hành. Nếu nhắc lại số cũ để so sánh lịch sử, phải gắn rõ nhãn checkpoint cũ.

Cập nhật nhất quán Abstract/tóm tắt, thiết lập thí nghiệm, kết quả VALID/TEST, bảng theo lớp, biểu đồ, phần ứng dụng/demo, thảo luận, giới hạn và kết luận. Bổ sung bảng training epoch 28/29/30, dùng CSV đủ 30 epoch để vẽ đường cong. Phân biệt mAP50–95 trong log E6 (best 0,652066 tại epoch 30) với evaluator chung sau training (VALID 0,660507; TEST 0,677107). mAR@100 trong log không phải Recall tại ngưỡng vận hành 0,60. Không dùng TEST để chọn lại checkpoint, threshold hoặc giải thích rằng ngưỡng được tối ưu trên TEST.

Kết luận cho bảng mới: E4 cao nhất cả năm chỉ số TEST tổng hợp. E6 cao hơn E3 về Precision TEST, nhưng thấp hơn E3 ở mAP50, mAP50–95, Recall và F2. Đánh giá cải thiện theo đúng số: hoàn tất thêm epoch không có nghĩa mọi chỉ số TEST đều tăng. Nếu phân tích theo lớp, dùng CSV so sánh mới; không giữ số lớp dẫn đầu cũ. Đây là evaluator COCO-style nội bộ, không phải pycocotools chính thức. Không tuyên bố kết quả CPU/latency cũ đã đo lại cho checkpoint mới; giữ thì ghi baseline cũ.

Giữ bố cục, trích dẫn và phương pháp không liên quan. Giữ hoặc cập nhật mục lục và kê khai AI 75% theo mẫu đã có; cập nhật số trang thực tế. Bổ sung giới hạn: heuristic phân vùng bbox chỉ là prototype, không suy ra khoảng cách hoặc độ an toàn thực tế. Nếu chỉnh slide baocao_final.pptx đi kèm, cập nhật cả biểu đồ, kết luận và speaker notes theo cùng số liệu; giữ theme/tỷ lệ. Không tạo code Gradio. Không bịa số liệu. Trả báo cáo DOCX và PDF mới, kèm danh sách các mục đã thay. Render và kiểm tra toàn bộ trang để sửa tràn bảng, đè chữ, lỗi tiếng Việt và số trang trước khi xuất.
'''
    content=f'''# Kết quả mới E3/E4/E6 — E6 hoàn tất 30 epoch và đã đánh giá đủ VALID/TEST

Cập nhật ngày 28/10/2026. Nguồn E6 mới: `results_valid_e6_30.zip`; nguồn E3/E4 giữ nguyên `results_danhgiae3e4.zip`. Nội dung này thay hướng dẫn cũ chưa có đánh giá E6 mới. Các tài liệu Word/PDF/PPTX đi kèm là nguồn để chỉnh trên web, chưa được sửa nội dung trực tiếp trong lần đóng gói này.

## Bảng VALID/TEST phải dùng

{table}
VALID: 3.318 ảnh, 5.320 đối tượng. TEST: 1.611 ảnh, 2.354 đối tượng. E6 VALID: TP=4.495, FP=1.491, FN=825. E6 TEST: TP=1.999, FP=648, FN=355. Ngưỡng E6 **0,60** được chọn bằng micro-F2 trên VALID trước TEST; không dùng lại ngưỡng 0,57 của checkpoint trước.

## Checkpoint và kiểm chứng

- E6: Faster R-CNN ResNet-50 FPN V2, 25 lớp ROD và một background, run 30/30, `is_complete=true`. Best SHA256 `{e6['checkpoint_sha256']}`.
- Đã xác minh SHA256 trọng số, predictions, metrics VALID, threshold sweep và hash khóa được TEST tham chiếu; thứ tự/tổng số ảnh khớp dataset Final local. Đã tính lại các metric từ prediction cache bằng evaluator trong app và khớp kết quả được cung cấp.
- Mỗi mô hình có khóa VALID riêng tại `metrics/<model>/evaluation_lock.json`. Chỉ mục khóa trong app được tạo sau khi nhập kết quả, không phải một lần khóa chung trước TEST.
- AP nội suy 101 điểm, IoU 0,50:0,05:0,95; class-aware matching; AP maxDet=100, P/R/F2 maxDet=300, IoU vận hành 0,50. Gọi là COCO-style nội bộ, không phải pycocotools chính thức.

## Kết quả huấn luyện E6

{train_table}
Log đủ 30 epoch nằm ở `metrics/E6/training/metrics.csv`. Epoch 30 có mAP50–95 cao nhất log: 0,6520662308, vượt mốc best cũ ở epoch 15 (0,6518420577). Không trộn đường cong log training với số tính lại bằng evaluator chung. mAR@100 không thay thế Recall ở ngưỡng 0,60.

## Nhận xét có thể chèn vào báo cáo

> Ba mô hình E3, E4 và E6 đều đã hoàn tất các run 30 epoch và được đánh giá bằng cùng protocol trên 3.318 ảnh VALID và 1.611 ảnh TEST. Với ngưỡng được chọn riêng trên VALID, E4 đạt chất lượng TEST cao nhất ở cả mAP50, mAP50–95, Precision, Recall và F2. E6 sử dụng ngưỡng 0,60, đạt mAP50 0,877393; mAP50–95 0,677107; Precision 0,755195; Recall 0,849193 và F2 0,828567. So với E3, E6 có Precision cao hơn nhưng các chỉ số mAP, Recall và F2 thấp hơn. Do đó, kết quả hiện có không ủng hộ kết luận E6 tốt nhất tổng thể.

E4 so với E6 trên TEST: Precision +{delta['precision']:.4f}, Recall +{delta['recall']:.4f}, F2 +{delta['f2']:.4f}, mAP50 +{delta['map50']:.4f}, mAP50–95 +{delta['map50_95']:.4f} **điểm phần trăm**; E4 có ít hơn {e6['fn']-reports['E4','test']['fn']} FN và {e6['fp']-reports['E4','test']['fp']} FP. AP50–95 theo lớp dẫn đầu: {', '.join(f'{m} {winners[m]}/25' for m in MODELS)}; nếu có hòa điểm thì mỗi mô hình đồng hạng được tính một lần.

## E6 theo lớp trên TEST — checkpoint mới, ngưỡng 0,60

{class_table}
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

{prompt}
'''
    save(ROOT/'CAP_NHAT_BAO_CAO_KET_QUA_E3_E4_E6.md',content)
    verified='# Kết quả đã xác minh E3/E4/E6\n\n'+table+'\nE6 mới: run hoàn tất 30/30, ngưỡng VALID 0,60, đã đánh giá đủ VALID/TEST. Nguồn `results_valid_e6_30.zip`. E4 dẫn đầu cả năm chỉ số TEST tổng hợp.\n'
    save(RESULTS/'VERIFIED_RESULTS.md',verified)
    status=f'''# E6 current checkpoint — COMPLETE and evaluated

Source: results_valid_e6_30.zip. Training complete 30/30; best SHA256 `{e6['checkpoint_sha256']}`.
VALID 3318 images / 5320 objects; TEST 1611 images / 2354 objects. VALID-selected threshold 0.60; original evaluation lock retained byte-for-byte. Metrics independently reproduced from prediction caches. App weights/config now point to this checkpoint.
Historical evidence before this import is archived at results/history/E6_before_30epoch_import_20261027.zip; old weights remain in original supplied training archives.
'''
    save(RESULTS/'E6/CHECKPOINT_STATUS.md',status)
    save(RESULTS/'E6/training_30_complete/README.md',status+'\nTraining log best mAP50-95: 0.6520662307739258 at epoch 30.\n')
    provenance_path=RESULTS/'E6/training_30_complete/INTEGRITY_AND_PROVENANCE.json'
    provenance=read(provenance_path)
    provenance['new_checkpoint_common_valid_test_evaluation']='COMPLETE: results_valid_e6_30.zip'
    provenance['current_valid_test_checkpoint_sha256']=e6['checkpoint_sha256']
    provenance['current_valid_test_threshold']=0.6
    save(provenance_path,json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (RESULTS/'E6/checkpoint/training_metrics.csv').write_bytes(train_path.read_bytes())

    readme_path=ROOT/'README.md'
    lines=readme_path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| E6 |'):
            lines[i]='| E6 | Faster R-CNN ResNet-50 FPN V2 COCO Pretrained | 30/30 epoch; best checkpoint mới đã đánh giá đủ VALID/TEST, ngưỡng 0,60 |'
        elif line.startswith('**Kết quả cập nhật:**'):
            lines[i]='**Kết quả cập nhật:** E3/E4/E6 hoàn tất các run 30 epoch, đã đánh giá đủ VALID/TEST. E6 mới nhập từ `results_valid_e6_30.zip`, ngưỡng VALID 0,60. Bảng hiện hành và bằng chứng tại [`results/30epochs/`](results/30epochs/).'
        elif line.startswith('- E6: [`results/30epochs/E6/checkpoint'):
            lines[i]=f"- E6: [`results/30epochs/E6/checkpoint/best_weights.pth`](results/30epochs/E6/checkpoint/best_weights.pth), SHA256 `{e6['checkpoint_sha256']}`; run hoàn tất 30/30, ngưỡng VALID 0,60."
        elif line.startswith('- E6 VALID/TEST:'):
            lines[i]='- E6 VALID/TEST: ngưỡng **0,60**, VALID mAP50–95 0,660507, TEST mAP50–95 0,677107. Bảng ba mô hình: [`summary_E3_E4_E6.csv`](results/30epochs/summary_E3_E4_E6.csv).'
        elif line.startswith('- Gói nhẹ để gắn vào Kaggle'):
            lines[i]='- Gói input Kaggle mới: [`results_e6_30epochs_eval.zip`](results_zips/results_e6_30epochs_eval.zip). Kết quả đánh giá mới nhất: `results_zips/results_valid_e6_30.zip`.'
        elif line.startswith('- E6 VALID/TEST trên Kaggle'):
            lines[i]='- E6 VALID/TEST trên Kaggle: [`Kaggle_E6_30epochs_Valid_Test.ipynb`](notebooks/Kaggle_E6_30epochs_Valid_Test.ipynb).'
        else:
            lines[i]=line.replace('0,27/0,24/0,57','0,27/0,24/0,60')
    note='''> Cập nhật 28/10/2026: kết quả hiện hành là bộ `results/30epochs/`; E6 dùng best mới sau 30 epoch và ngưỡng 0,60. `app/evaluation_lock.json` là chỉ mục local của các khóa gốc riêng từng model, được nhập sau đánh giá. Kết quả `results/E*/metrics/` và benchmark cũ bên dưới là lịch sử; không coi chúng là số mới. Chạy `python scripts/validate_final_test_results.py` để đối chiếu ba kết quả hiện hành.\n'''
    if note.strip() not in '\n'.join(lines):
        lines.insert(2,note)
    save(readme_path,'\n'.join(lines)+'\n')
    for filename in ['docs/plans/EXPERIMENT_STATUS.md','docs/plans/MASTER_RESEARCH_ROADMAP.md']:
        path=ROOT/filename
        original=path.read_text(encoding='utf-8')
        prefix='''> Trạng thái hiện hành 28/10/2026: E3/E4/E6 đã hoàn tất run 30 epoch và có đủ VALID/TEST. E6 mới từ `results_valid_e6_30.zip`, threshold 0,60, TEST mAP50–95 0,677107. Xem `results/30epochs/VERIFIED_RESULTS.md`. Các phần kế hoạch và mốc cũ bên dưới giữ để truy vết lịch sử.\n\n'''
        if not original.startswith(prefix):
            save(path,prefix+original)
    guide=ROOT/'docs/kaggle/E6_30EPOCHS_VALID_TEST.md'
    original=guide.read_text(encoding='utf-8')
    save(guide,original.replace('Chưa có số đánh giá mới cho đến khi notebook hoàn tất trên dữ liệu thật.','Đã nhập kết quả mới từ `results_valid_e6_30.zip`: ngưỡng VALID 0,60, TEST mAP50–95 0,677107; bằng chứng ở `results/30epochs/E6/`.'))

    # Standalone scientific plots; CSV sources remain in the package for redrawing.
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib_report_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets=OUT/'E6_30epoch_report_figures'
    assets.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    epochs=[int(r['epoch']) for r in train]
    axes[0].plot(epochs,[float(r['map5095']) for r in train],marker='.',label='Training-log VALID mAP50-95')
    axes[0].set(xlabel='Epoch',ylabel='mAP50-95',title='E6: 30 completed epochs',xticks=[1,5,10,15,20,25,30])
    axes[1].plot(epochs,[float(r['loss_mean']) for r in train],color='#ca6b24')
    axes[1].set(xlabel='Epoch',ylabel='Mean training loss',title='E6 training loss')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(assets/'E6_training_30epochs.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    metrics=['map50','map50_95','precision','recall','f2']
    for i,model in enumerate(MODELS):
        ax.bar([j+(i-1)*.25 for j in range(5)],[reports[model,'test'][k]*100 for k in metrics],width=.24,label=model)
    ax.set(xticks=list(range(5)),xticklabels=['mAP50','mAP50-95','Precision','Recall','F2'],ylabel='Percent',ylim=(0,100),title='TEST: best checkpoints from completed 30-epoch runs')
    ax.legend(ncol=3,loc='lower right');ax.grid(axis='y',alpha=.2)
    fig.savefig(assets/'TEST_E3_E4_E6_30epochs.png',dpi=200);plt.close(fig)

    package=OUT/f'GOI_BAO_CAO_KET_QUA_MOI_E6_30EPOCH_{STAMP}.zip'
    entries={
        'KET_QUA_MOI_E3_E4_E6.md':content.encode('utf-8'),
        'PROMPT_DAN_LEN_WEB.txt':prompt.encode('utf-8'),
        'HUONG_DAN.md':('''# Cách dùng gói báo cáo mới

Tải Word/PDF trong `tai_lieu_nguon/` cùng `KET_QUA_MOI_E3_E4_E6.md` lên công cụ sửa báo cáo trên web, rồi dán `PROMPT_DAN_LEN_WEB.txt`. Nếu công cụ nhận ZIP, có thể tải cả gói. Báo cáo nguồn chưa được sửa trực tiếp trong lần đóng gói này. Các CSV/JSON và biểu đồ đi kèm là bằng chứng/nguồn vẽ lại. Không cần upload trọng số, source code app hoặc dữ liệu ảnh.

Kết quả E6 mới đã đủ VALID/TEST, threshold 0,60; không dùng gói bàn giao cũ nói chưa đánh giá lại. File slide nguồn chỉ dùng nếu bạn muốn sửa thêm slide. Yêu cầu công cụ trả DOCX/PDF mới, kèm danh sách mục đã cập nhật.
''').encode('utf-8'),
    }
    def add(path,relative):
        entries[relative]=path.read_bytes()
    for name in ['Paper_ACL_E6_30Epoch_BanMoi_27102026.docx','Paper_ACL_E6_30Epoch_BanMoi_27102026.pdf','baocao_final.pptx']:
        add(ROOT/name,'tai_lieu_nguon/'+name)
    add(ROOT/'handoff_report_slide/NguyenXuanMong.pdf','tai_lieu_nguon/NguyenXuanMong.pdf')
    for model in MODELS:
        for split in ['valid','test']:
            for name in ['metrics.json','metadata.json','per_class.csv','threshold_sweep.csv']:
                path=RESULTS/model/split/name
                if path.exists():add(path,f'metrics/{model}/{split}/{name}')
        add(RESULTS/model/'evaluation_lock.json',f'metrics/{model}/evaluation_lock.json')
        add(RESULTS/model/'checkpoint/manifest.json',f'metrics/{model}/checkpoint/manifest.json')
    for name in ['summary_E3_E4_E6.csv','summary_E3_E4_E6.json','VERIFIED_RESULTS.md']:
        add(RESULTS/name,'metrics/'+name)
    for path in [train_path, RESULTS/'E6/training_30_complete/epochs_28_29_30.csv']:
        add(path,'metrics/E6/training/'+path.name)
    add(RESULTS/'E6/IMPORT_AUDIT.json','metrics/E6/IMPORT_AUDIT.json')
    add(comparison_path,'metrics/'+comparison_path.name)
    for path in assets.glob('*.png'):add(path,'figures/'+path.name)
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):z.writestr(name,data)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.pth','.pt','.jsonl')) for n in z.namelist())
        for name in ['KET_QUA_MOI_E3_E4_E6.md','PROMPT_DAN_LEN_WEB.txt','HUONG_DAN.md']:
            assert '\ufffd' not in z.read(name).decode('utf-8')
    print(json.dumps({'package':str(package),'bytes':package.stat().st_size,'files':len(entries),'class_leaders':winners,'zip_crc':'PASS'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
