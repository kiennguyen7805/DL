# Chạy 30 epoch bằng code notebook cũ

Cập nhật 2026-10-18 theo yêu cầu: dùng lại notebook đã chạy 10 epoch, nâng giới hạn lên 30. Các notebook 10 epoch gốc, YAML freeze, dataset, checkpoint local và calibration không bị sửa.

## File upload Kaggle

| Model | Notebook | HF lưu tiếp | Input |
|---|---|---|---|
| E3 | notebooks/Kaggle_E3_30epochs_AutoResume.ipynb | YOLO 11n/ | ROD Final; thêm e3_resume_epoch010.zip nếu chưa có recovery HF |
| E4 | notebooks/Kaggle_E4_30epochs_AutoResume.ipynb | YOLO 11s/ | ROD Final |
| E6 | notebooks/Kaggle_E6_30epochs_AutoResume.ipynb | Faster R-CNN/ | ROD Final |

Repo vẫn là iraprp/deeplearning, Secret HF_TOKEN2. Bật một GPU CUDA và Internet. Import lại notebook mới và Save Version > Save & Run All. Các notebook cũ 10 epoch không đổi.

**Lần này chạy tiếp đến 30, không tự dừng ở 20.** Nếu đủ thời gian, có thể hoàn tất cả epoch 11–30 trong cùng phiên. Nếu chạm deadline cũ (YOLO khoảng 10 giờ, E6 9,5 giờ) thì lưu rồi dừng, phiên sau tiếp tục. Không thể đảm bảo thời lượng khi chưa có log/GPU thực tế. Không chạy đồng thời hai notebook cho cùng model vì sẽ tranh ghi latest pointer.

## Thay đổi tối thiểu

E4 giữ nguyên data preflight và YAML training nhúng của notebook Kaggle_E4_YOLO11s_HF_AutoResume.ipynb; đổi E4_BUDGET từ 10 thành 30. Dùng trực tiếp model.train(resume=True, data=..., device=0), callback backup mỗi epoch như trước. Không custom Trainer, không Tee, không redirect stdout, không runner run_extension hoặc kiểm tra start callback bổ sung.

E6 giữ Dataset, DataLoader, forward/backward, optimizer, scheduler, backup 1.800 minibatch và vòng validation của notebook Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb; đổi giới hạn loop từ 10 thành 30. Trường epochs=10 trong config identity được giữ là cấu hình nguồn để khớp checkpoint cũ; ngân sách mới ghi target_epochs=30 trong manifest. Không giả rằng lịch E6 gốc đã thiết kế cho 30 epoch.

E3 dùng cùng cơ chế YOLO đã chạy ở notebook E4 nhưng model yolo11n, class/scale guard n và YAML E3. Vì E3 ban đầu tải Output về máy, khi chưa có snapshot HF sẽ nhận Input bundle ZIP hoặc thư mục Kaggle tự giải nén có source_manifest.json và checksum đúng.

Chỉ bổ sung phần cần để tiếp tục an toàn: nhận manifest cũ COMPLETE tại 10 (không coi là hoàn tất ngân sách mới), tìm checkpoint hợp lệ có tiến độ cao hơn giữa folder gốc và E3_30epochs/E4_30epochs/E6_30epochs từ lần thử trước, bảo toàn đường dẫn output đã lưu trong checkpoint YOLO và sao lưu full state ở đúng epoch 20/30. Không sửa trọng số hoặc reset optimizer/LR. Hai nguồn cùng tiến độ ưu tiên folder gốc. Nếu nguồn nào sai checksum/config thì dừng để kiểm tra, không âm thầm train lại từ đầu.

Sau khi chuyển sang bản đơn giản, HF cập nhật latest ở folder gốc như bảng trên; các snapshot UUID cũ vẫn còn. E3 lần đầu có thể lấy recovery từ E3_30epochs trước khi cần Input. Checkpoint E3_30epochs chỉ ở 10 là bản bootstrap đã lưu, chưa chứng minh epoch 11 hoàn tất.

## Kết quả và giới hạn

Mỗi epoch YOLO có full recovery được giữ trước final_eval strip optimizer, results.csv, args.yaml, best và manifest SHA256 như notebook cũ. E6 giữ full model/optimizer/scheduler/scaler/RNG/offset/metrics. Best có thể thuộc epoch trước; không đồng nhất với checkpoint đúng mốc.

Mốc 10 đã có ở nguồn. resume_epoch020.pt và resume_epoch030.pt nằm trong Output khi đạt mốc; snapshot HF tại completed_epoch=20/30 cũng giữ recovery đúng mốc. Tải Output sau phiên, không coi runtime là backup bền vững. HF lỗi thì checkpoint chỉ còn local cần tải ngay. COMPLETE mới chỉ khi đủ 30 và checkpoint verify PASS. Không còn file training_status.json/execution.log do wrapper mới tạo; xem log native Kaggle và last_hf_manifest.json.

Không thay batch, imgsz, seed, augmentation hoặc lịch LR. E3/E4 tiếp tục lịch linear 30 epoch; E6 tiếp tục StepLR(3,0.1). E6 LR sau epoch 10 đã là 0,000005 và sẽ tiếp tục giảm; không hứa train thêm sẽ cải thiện. Không dùng TEST khi train hoặc điều chỉnh cấu hình/threshold. Giữ nguyên kết quả baseline 10 và evaluation lock của app. Các mốc mới phải đánh giá bằng evaluator chung trên VALID trước khi chốt đánh giá bổ sung.

## Kiểm thử và tái tạo

Builder chính src/build_30epoch_notebooks.py hiện gọi src/build_simple_30epoch_notebooks.py. Bản runner phức tạp src/rod_train_30epochs.py không còn được nhúng/chạy trong notebook; một số hàm kiểm checksum/input của nó vẫn được builder tái sử dụng. Không cần upload source Python.

53 tests PASS, 0 FAIL: notebook syntax, giữ nguyên cấu hình/dataset E4, vòng training E6 giống bản gốc, tương thích manifest cũ COMPLETE tại 10, không bỏ checkpoint extension mới hơn, hashes/ZIP, class mapping, resume state và optimizer/scheduler CPU. Kết quả: artifacts/training_30epochs/local_tests.xml. Đây không phải training ROD hoặc GPU integration.

Kaggle run mới: CHƯA XÁC MINH. Không sửa/dừng phiên Kaggle hiện có từ máy local. Trước khi thay notebook, tải Output của phiên cũ nếu có tiến độ chưa đồng bộ HF.

Cập nhật chẩn đoán HF: lỗi upload giờ ghi HTTP status và thông báo server đã che token/URL; không còn chỉ hiện BadRequestError. Đã kiểm thử thông báo lỗi và chống lộ token. Chưa xác định nguyên nhân HF 400 của phiên thật vì log cũ đã bỏ chi tiết server. Bản E3 kế thừa tên output /kaggle/working/rod_e4_yolo11n_hf/ từ notebook nguồn; đây vẫn là YOLO11n, không phải E4. Cần giữ Output khi upload thất bại.


## HF repository: 13ffa/deep (current decision)

### Download completed E3/E4 checkpoints locally

From the project root, with Python and `huggingface_hub` installed:

```powershell
python -m scripts.download_hf_checkpoints
```

The default source is `13ffa/deep`: E3 uses `YOLO 11n/latest.json`, E4 uses `YOLO 11s/latest.json`. Authentication uses the `HF_TOKEN` environment variable or `hf auth login`; do not put tokens in source files. Repository type is detected automatically, or pass `--repo-type dataset` / `--repo-type model`.

Both downloads use one pinned repository commit. The script requires COMPLETE at epoch 30, validates model/config/dataset identity, pointer/manifest/artifact SHA256, and resolves compact `best.pt` aliases. It downloads full recovery, best weights, metrics and configuration without loading PyTorch or writing to HF. To download only one model use `--models E3` or `--models E4`. Use `--epochs 10` only for an intentionally completed 10-epoch run.

Verified files are stored at `artifacts/hf_checkpoints/E3/epoch030/<snapshot-id>/` and the equivalent E4 path. `download_receipt.json` records the repository commit and best checksum after verification. Existing app checkpoints, confidence calibration and locked test results remain tied to the previous weights; downloading does not activate new weights in the app.

All three 30-epoch notebooks write to 13ffa/deep. Folders remain YOLO 11n/, YOLO 11s/, Faster R-CNN/. Enable Kaggle Secret HF_TOKEN with WRITE permission on 13ffa/deep. Repository type (dataset/model) is detected at runtime; no repository is auto-created.

When the destination has no verified recovery, notebooks use HF_TOKEN2 with READ permission on the source dataset repo iraprp/deeplearning. Both the original folders and extension folders E3_30epochs/, E4_30epochs/, E6_30epochs/ are checked there. All source files and latest pointers remain untouched. Once the destination has recovery, further runs resume from the destination without requiring the old secret.

Enable BOTH secrets on the notebook for initial migration: HF_TOKEN (new account destination) and HF_TOKEN2 (old source). Do not paste tokens into code. Checkpoints still undergo model/config/dataset/epoch/checksum validation. Failed access or invalid checkpoints stop execution instead of silently restarting. No change to the training configuration/fingerprint, scheduler or weights. No training/upload has been executed locally to validate the new account permissions.

The old account's private quota failure remains. No remote cleanup has been executed. Local epoch-13 recovery not yet uploaded is NOT automatically read from Kaggle Output on another session: retain the downloaded files. If HF source recovery is only epoch 12, training continues at epoch 13, not 14. Check the printed restored epoch.

## Compact HF uploads (current policy)

Enabled in all three 30-epoch destination stores only. Repository: 13ffa/deep (HF_TOKEN). iraprp/deeplearning remains read-only (HF_TOKEN2). No training hyperparameter, model weight, dataset, or training config fingerprint changed. Historical 10-epoch notebooks remain untouched.

YOLO uploads: recovery_full.pt (full optimizer/EMA/scaler state), best.pt when distinct, results.csv, args.yaml, config.json, environment.json, manifest.json. last.pt is no longer duplicated remotely; all original local output files remain. If best is byte-identical to recovery, manifest artifact_aliases maps best.pt to recovery_full.pt. Updated restore validates and resolves this alias. E6 uploads checkpoint.pt (full resume state), best_weights.pth, metrics.csv, config.json, environment.json, manifest.json; identical best/recovery bytes can be aliased too.

E3/E4 upload once after each completed epoch. E6 local recovery remains every 1800 batches. E6 remote upload occurs at completed epoch boundaries, deadline, or at a local-save boundary at least 3600 seconds after the preceding successful remote save. The hour is approximate, checked at existing batch-save boundaries; it is not a hard timer. A killed runtime can lose work since its last HF upload; keep Kaggle Output as backup.

After a successful snapshot upload, manifest check, latest-pointer confirmation and remote file-presence check, retention removes only older compact-v1 snapshots from the same destination model folder/config/dataset. It retains latest recovery (with best), complete epoch 10/20/30 snapshots if present, and a previous best-containing snapshot if current has no best. Complete epoch-20 state is retained separately from a partial epoch-21 state. Unknown/unmanaged snapshots, other model folders, other configs, future progress, and the old source repo are not deleted. A conditional parent_commit rejects concurrent writes; a cleanup conflict preserves extra snapshots and does not discard the new successful checkpoint. Only one writer per model folder is supported.

This bounds the managed LIVE FILE TREE, not all Hugging Face storage. Git/LFS history and pre-existing unmanaged snapshots can continue consuming quota. No automatic super_squash_history during training: the three models share a repository and history rewriting is irreversible. Existing 40/60 GB has NOT been cleaned, and no exact storage saving is claimed. Use the separate backup/cleanup process while all writers are stopped to reclaim eligible historical data. Original epoch-10 source checkpoints are not copied into each new snapshot.

Local verification: 69 tests PASS, 0 FAIL (1 warning); includes existing notebook/resume regression tests and 11 compact-upload tests. Mock HF only; no real network upload, deletion, GPU training or TEST evaluation executed. JUnit: artifacts/training_30epochs/compact_upload_tests.xml. Upload logs now print exact staged upload_MiB so actual payload size can be observed on Kaggle.
