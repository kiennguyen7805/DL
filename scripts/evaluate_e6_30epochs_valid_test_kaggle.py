"""Evaluate the completed E6 30-epoch best checkpoint: VALID, lock, then TEST."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import torch
import torchvision
import yaml
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor


# Kaggle inputs: ROD Final and results_e6_30epochs_eval.zip (or the full ZIP).
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working/evaluation_E6_30epochs_complete')
DATASET_ROOT = None  # Set explicitly only if Kaggle finds multiple ROD datasets.
E6_ZIP = None        # Optional explicit path to results_e6_282930.zip or compact ZIP.
EXPECTED_BEST_SHA256 = 'a75beb6e3d8629bf9066bc8a03a0b44de93a720aa3d7b3663f0641b732ae8a8f'
BATCH_SIZE = 2       # Lower to 1 if GPU memory runs out.
EXPECTED_SPLIT_IMAGES = {'valid': 3318, 'test': 1611}
COLLECTION_CONF, NMS_IOU, MAX_DET = 0.01, 0.7, 300
THRESHOLDS = [round(i / 100, 2) for i in range(1, 96)]
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
AP_MAX_DET, OPERATING_MAX_DET = 100, 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def find_dataset() -> tuple[Path, Path, dict]:
    candidates = [Path(DATASET_ROOT)] if DATASET_ROOT else list(INPUT_ROOT.rglob('data.yaml'))
    valid = []
    for yaml_path in candidates:
        if yaml_path.is_dir():
            yaml_path = yaml_path / 'data.yaml'
        if not yaml_path.is_file():
            continue
        try:
            config = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(config, dict) or 'names' not in config:
            continue
        names = config['names']
        if isinstance(names, dict):
            names = [names[i] if i in names else names[str(i)] for i in range(25)]
        if len(names) != 25:
            continue
        valid.append((yaml_path.parent, yaml_path, config))
    if len(valid) != 1:
        raise RuntimeError(f'Attach exactly one ROD Final dataset or set DATASET_ROOT; found {len(valid)}: {[str(x[0]) for x in valid]}')
    return valid[0]


def validate_manifest(manifest):
    if (manifest.get('model') != 'fasterrcnn_resnet50_fpn_v2' or
            manifest.get('target_epochs') != 30 or
            manifest.get('completed_epoch') != 30 or
            manifest.get('is_complete') is not True or
            manifest.get('class_count') != 25 or
            manifest.get('sha256', {}).get('best_weights.pth') != EXPECTED_BEST_SHA256):
        raise RuntimeError('Expected the COMPLETE 30/30 E6 best weights from results_e6_282930.zip; old 28-epoch weights are not accepted.')


def find_checkpoint() -> tuple[Path, dict, Path | None]:
    if E6_ZIP:
        zip_candidates = [Path(E6_ZIP)]
    else:
        zip_candidates = []
        for filename in ('results_e6_30epochs_eval.zip', 'results_e6_282930.zip'):
            zip_candidates = list(INPUT_ROOT.rglob(filename))
            if zip_candidates:
                break  # prefer the compact evaluation bundle if both are attached
    if len(zip_candidates) > 1:
        raise RuntimeError(f'Multiple E6 ZIP files found; set E6_ZIP explicitly: {[str(p) for p in zip_candidates]}')
    if zip_candidates:
        archive_path = zip_candidates[0]
        with zipfile.ZipFile(archive_path) as archive:
            manifests = [n for n in archive.namelist() if n.endswith('/last_hf_manifest.json')]
            if len(manifests) != 1:
                raise RuntimeError(f'Expected one last_hf_manifest.json in {archive_path}; found {len(manifests)}')
            manifest = json.loads(archive.read(manifests[0]))
            validate_manifest(manifest)
            relative = str(Path(manifests[0]).parent / 'best_weights.pth').replace('\\', '/')
            if relative not in archive.namelist():
                raise RuntimeError('Archive manifest found but best_weights.pth is missing beside it.')
            target_dir = OUTPUT_ROOT / 'E6' / 'checkpoint'
            target_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = target_dir / 'best_weights.pth'
            with archive.open(relative) as source, checkpoint.open('wb') as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            expected = manifest.get('sha256', {}).get('best_weights.pth')
            if not expected or sha256_file(checkpoint) != expected:
                raise RuntimeError('Extracted E6 checkpoint SHA256 does not match archive manifest.')
            return checkpoint, manifest, archive_path
    # Also allow attaching the ZIP after Kaggle has extracted it.
    matches = []
    for path in INPUT_ROOT.rglob('last_hf_manifest.json'):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        try:
            validate_manifest(manifest)
        except RuntimeError:
            continue
        checkpoint = path.parent / 'best_weights.pth'
        if checkpoint.is_file():
            matches.append((checkpoint, manifest))
    if len(matches) != 1:
        raise FileNotFoundError(f'Attach results_e6_30epochs_eval.zip or results_e6_282930.zip (or extracted output); found {len(matches)} matching completed checkpoints.')
    source, manifest = matches[0]
    target_dir = OUTPUT_ROOT / 'E6' / 'checkpoint'
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = target_dir / 'best_weights.pth'
    shutil.copy2(source, checkpoint)
    if sha256_file(checkpoint) != manifest.get('sha256', {}).get('best_weights.pth'):
        raise RuntimeError('Extracted E6 checkpoint SHA256 does not match archive manifest.')
    return checkpoint, manifest, None


def split_images(data_root: Path, config: dict, split: str) -> tuple[Path, Path]:
    aliases = {'valid': ('valid', 'val'), 'test': ('test',)}[split]
    candidates = []
    configured = config.get('val' if split == 'valid' else 'test')
    if configured:
        p = Path(str(configured))
        candidates.append(p if p.is_absolute() else data_root / p)
    for alias in aliases:
        candidates.extend((data_root / alias / 'images', data_root / 'images' / alias))
    images_dir = next((p for p in candidates if p.is_dir() and any(
        f.suffix.lower() in {'.jpg', '.jpeg', '.png'} for f in p.iterdir())), None)
    if images_dir is None:
        raise FileNotFoundError(f'Could not locate {split}/images beneath {data_root}. Checked: {candidates}')
    # ROD layout is split/images and split/labels.
    labels_dir = images_dir.parent / 'labels'
    if not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing labels directory: {labels_dir}')
    return images_dir, labels_dir


def image_paths(images_dir: Path, split: str) -> list[Path]:
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    expected = EXPECTED_SPLIT_IMAGES[split]
    if len(paths) != expected:
        raise RuntimeError(f'{split.upper()} expected {expected} images, found {len(paths)} at {images_dir}')
    return paths


def ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    if not label_path.is_file():
        raise RuntimeError(f'Missing label: {label_path}')
    rows = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5 or int(values[0]) != values[0] or not 0 <= values[0] < 25:
            raise RuntimeError(f'Invalid YOLO label: {label_path}')
        cls, x, y, w, h = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise RuntimeError(f'Invalid normalized box: {label_path}')
        rows.append([int(cls), (x-w/2)*width, (y-h/2)*height,
                     (x+w/2)*width, (y+h/2)*height])
    return rows


def build_model(device: torch.device):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=640, max_size=960,
        box_score_thresh=0.0, box_nms_thresh=NMS_IOU,
        box_detections_per_img=MAX_DET,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 26)
    state = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    if tuple(state['roi_heads.box_predictor.cls_score.weight'].shape) != (26, features):
        raise RuntimeError('E6 checkpoint classification head is not 26 classes.')
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def collect_split(model, device, split, data_root, config, dataset_yaml, checkpoint, ckpt_sha):
    images_dir, labels_dir = split_images(data_root, config, split)
    paths = image_paths(images_dir, split)
    out = OUTPUT_ROOT / 'E6' / split
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'predictions.jsonl'
    metadata_path = out / 'metadata.json'
    metadata = {
        'model': 'E6', 'split': split, 'images': len(paths),
        'checkpoint_sha256': ckpt_sha, 'data_yaml_sha256': sha256_file(dataset_yaml),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'inference': {'conf': COLLECTION_CONF, 'iou': NMS_IOU, 'max_det': MAX_DET,
                      'batch_size': BATCH_SIZE, 'min_size': 640, 'max_size': 960},
    }
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(saved_metadata.get(key) != value for key, value in metadata.items()):
            raise RuntimeError(f'{split} cache metadata differs; refusing to reuse old predictions.')
    elif cache.exists():
        raise RuntimeError(f'{split} prediction cache exists without metadata.')
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    records = []
    if cache.exists():
        raw_lines = cache.read_bytes().splitlines(keepends=True)
        valid_lines = []
        for i, line in enumerate(raw_lines):
            try:
                record = json.loads(line)
            except Exception:
                if i != len(raw_lines) - 1:
                    raise RuntimeError(f'Corrupt {split} cache before final line; remove cache to restart.')
                break  # discard only a possibly interrupted final write
            valid_lines.append(line)
            records.append(record)
        if len(valid_lines) != len(raw_lines):
            cache.write_bytes(b''.join(valid_lines))
        if len(records) > len(paths) or any(r['image'] != p.name for r, p in zip(records, paths)):
            raise RuntimeError(f'{split} cache order/content does not match the frozen split.')

    started = time.monotonic()
    with cache.open('a', encoding='utf-8') as stream:
        for offset in range(len(records), len(paths), BATCH_SIZE):
            batch = paths[offset:offset+BATCH_SIZE]
            tensors, dimensions = [], []
            for path in batch:
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    dimensions.append(rgb.size)
                    tensors.append(pil_to_tensor(rgb).float().div_(255))
            inputs = [tensor.to(device) for tensor in tensors]
            with torch.inference_mode():
                outputs = model(inputs)
            for path, (width, height), prediction in zip(batch, dimensions, outputs, strict=True):
                boxes = prediction['boxes'].detach().cpu().tolist()
                labels = prediction['labels'].detach().cpu().tolist()
                scores = prediction['scores'].detach().cpu().tolist()
                preds = [[int(label)-1, float(score), *map(float, box)]
                         for label, score, box in zip(labels, scores, boxes, strict=True)
                         if int(label) > 0 and float(score) >= COLLECTION_CONF]
                record = {
                    'image': path.name, 'width': width, 'height': height,
                    'gt': ground_truth(labels_dir / (path.stem + '.txt'), width, height),
                    'pred': preds,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.append(record)
            stream.flush()
            if len(records) % 100 < BATCH_SIZE or len(records) == len(paths):
                print(f'E6 {split.upper()} {len(records)}/{len(paths)}; {time.monotonic()-started:.0f}s', flush=True)
    if len(records) != len(paths) or len({r['image'] for r in records}) != len(paths):
        raise RuntimeError(f'{split} cache incomplete or has duplicate images.')
    return records, cache, metadata


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def operating_score(records, threshold):
    tp = fp = fn = 0
    for row in records:
        used = set()
        predictions = sorted((p for p in row['pred'] if p[1] >= threshold), key=lambda p: -p[1])[:OPERATING_MAX_DET]
        for cls, score, *box in predictions:
            candidates = [(iou(box, gt[1:]), idx) for idx, gt in enumerate(row['gt'])
                          if idx not in used and gt[0] == cls]
            best, idx = max(candidates, default=(0.0, -1))
            if best >= 0.5:
                used.add(idx)
                tp += 1
            else:
                fp += 1
        fn += len(row['gt']) - len(used)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f2 = 5*precision*recall/(4*precision+recall) if precision or recall else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f2': f2}


def calibrate_valid(records, out):
    sweep = []
    for threshold in THRESHOLDS:
        score = operating_score(records, threshold)
        sweep.append({'threshold': threshold, **score})
    chosen = max(sweep, key=lambda row: (row['f2'], row['precision'], row['threshold']))
    with (out / 'threshold_sweep.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['threshold','tp','fp','fn','precision','recall','f2'])
        writer.writeheader(); writer.writerows(sweep)
    return chosen


def class_metrics(records, class_id, match_iou, threshold=None, max_det=AP_MAX_DET):
    gt_by_image = {r['image']: [g[1:] for g in r['gt'] if g[0] == class_id] for r in records}
    gt_count = sum(map(len, gt_by_image.values()))
    predictions = []
    for row in records:
        selected = sorted(row['pred'], key=lambda p: -p[1])[:max_det]
        predictions.extend((float(p[1]), row['image'], p[2:]) for p in selected
                           if p[0] == class_id and (threshold is None or p[1] >= threshold))
    predictions.sort(key=lambda item: -item[0])
    used = {name: set() for name in gt_by_image}
    outcomes = []
    for score, name, box in predictions:
        candidates = [(iou(box, gt), idx) for idx, gt in enumerate(gt_by_image[name]) if idx not in used[name]]
        best, idx = max(candidates, default=(0.0, -1))
        hit = best >= match_iou
        if hit:
            used[name].add(idx)
        outcomes.append((score, hit))
    if threshold is not None:
        tp = sum(hit for _, hit in outcomes); fp = len(outcomes)-tp
        return {'gt_count': gt_count, 'tp': tp, 'fp': fp, 'fn': gt_count-tp}
    if gt_count == 0:
        return {'gt_count': 0, 'ap': None}
    tp = fp = 0; curve = []
    for _, hit in outcomes:
        tp += int(hit); fp += int(not hit)
        curve.append((tp/gt_count, tp/(tp+fp)))
    ap = sum(max((precision for recall, precision in curve if recall >= r), default=0.0)
             for r in (i/100 for i in range(101))) / 101
    return {'gt_count': gt_count, 'ap': ap}


def evaluate(records, threshold, names, split, ckpt_sha, prediction_sha, data_sha):
    per_class = []
    for cls in range(25):
        aps = [class_metrics(records, cls, iou_t)['ap'] for iou_t in IOU_THRESHOLDS]
        op = class_metrics(records, cls, 0.5, threshold, OPERATING_MAX_DET)
        per_class.append({
            'class_id': cls, 'class_name': names[cls], 'gt_count': op['gt_count'],
            'ap50': aps[0], 'ap50_95': sum(aps)/len(aps) if op['gt_count'] else None,
            'tp': op['tp'], 'fp': op['fp'], 'fn': op['fn'],
        })
    active = [p for p in per_class if p['gt_count']]
    tp, fp, fn = (sum(p[k] for p in per_class) for k in ('tp','fp','fn'))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f2 = 5*precision*recall/(4*precision+recall) if precision or recall else 0.0
    protocol = {
        'iou_thresholds': list(IOU_THRESHOLDS), 'ap_recall_points': 101,
        'max_detections_per_image': AP_MAX_DET,
        'operating_max_detections_per_image': OPERATING_MAX_DET,
        'match': 'class-aware greedy, score descending', 'area': 'all',
        'crowd_ignore': False, 'note': 'COCO-style interpolation; ROD has no crowd/ignore labels',
    }
    return {
        'model': 'E6', 'split': split, 'status': 'FULL_EVALUATION',
        'images': len(records), 'ground_truth_objects': sum(p['gt_count'] for p in per_class),
        'operating_threshold': threshold, 'precision': precision, 'recall': recall,
        'f2': f2, 'tp': tp, 'fp': fp, 'fn': fn,
        'map50': sum(p['ap50'] for p in active)/len(active) if active else None,
        'map50_95': sum(p['ap50_95'] for p in active)/len(active) if active else None,
        'checkpoint_sha256': ckpt_sha, 'prediction_sha256': prediction_sha,
        'data_yaml_sha256': data_sha, 'protocol': protocol, 'per_class': per_class,
    }


def write_metrics(result, out):
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (out / 'per_class.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['class_id','class_name','gt_count','ap50','ap50_95','tp','fp','fn'])
        writer.writeheader(); writer.writerows(result['per_class'])


def main():
    global CHECKPOINT_PATH
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle Settings > Accelerator > GPU before running.')
    device = torch.device('cuda:0')
    if BATCH_SIZE < 1:
        raise ValueError('BATCH_SIZE must be positive.')
    print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__, flush=True)
    data_root, dataset_yaml, config = find_dataset()
    names = config['names']
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(25)]
    names = [str(n) for n in names]
    CHECKPOINT_PATH, manifest, source_zip = find_checkpoint()
    ckpt_sha, data_sha = sha256_file(CHECKPOINT_PATH), sha256_file(dataset_yaml)
    if data_sha != manifest.get('dataset_sha256'):
        raise RuntimeError('ROD data.yaml SHA256 differs from the training dataset manifest; attach the original ROD Final dataset.')
    (OUTPUT_ROOT / 'E6' / 'checkpoint' / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    metrics_source = None
    if source_zip:
        with zipfile.ZipFile(source_zip) as archive:
            manifest_member = next(n for n in archive.namelist() if n.endswith('/last_hf_manifest.json'))
            match = manifest_member.rsplit('/', 1)[0] + '/metrics.csv'
            if match in archive.namelist():
                raw_metrics = archive.read(match)
                if hashlib.sha256(raw_metrics).hexdigest() != manifest.get('sha256', {}).get('metrics.csv'):
                    raise RuntimeError('Training metrics checksum differs from manifest.')
                metrics_source = raw_metrics.decode('utf-8')
    if metrics_source is not None:
        (OUTPUT_ROOT / 'E6' / 'checkpoint' / 'training_metrics.csv').write_text(metrics_source, encoding='utf-8')
    print(f'Dataset: {data_root}\nCheckpoint: {CHECKPOINT_PATH}\nSHA256: {ckpt_sha}\nRun: {manifest.get("completed_epoch")}/{manifest.get("target_epochs")} epochs; complete={manifest.get("is_complete")}', flush=True)
    model = build_model(device)

    # VALID first: collect predictions, select micro-F2 threshold, then write a run-specific lock.
    valid_records, valid_cache, valid_meta = collect_split(
        model, device, 'valid', data_root, config, dataset_yaml, CHECKPOINT_PATH, ckpt_sha)
    valid_out = OUTPUT_ROOT / 'E6' / 'valid'
    chosen = calibrate_valid(valid_records, valid_out)
    valid_result = evaluate(valid_records, chosen['threshold'], names, 'valid', ckpt_sha,
                            sha256_file(valid_cache), data_sha)
    valid_result['status'] = 'FULL_VALID'
    valid_result['calibration'] = {k: chosen[k] for k in ('threshold','precision','recall','f2','tp','fp','fn')}
    write_metrics(valid_result, valid_out)

    lock = {
        'model': 'E6', 'checkpoint_sha256': ckpt_sha, 'data_yaml_sha256': data_sha,
        'completed_epoch': manifest.get('completed_epoch'), 'target_epochs': manifest.get('target_epochs'),
        'training_complete': manifest.get('is_complete'), 'threshold': chosen['threshold'],
        'threshold_selection': 'VALID micro-F2; ties precision then threshold',
        'valid_prediction_sha256': sha256_file(valid_cache),
        'valid_metrics_sha256': sha256_file(valid_out / 'metrics.json'),
        'threshold_sweep_sha256': sha256_file(valid_out / 'threshold_sweep.csv'),
    }
    lock_path = OUTPUT_ROOT / 'E6' / 'evaluation_lock.json'
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lock_sha = sha256_file(lock_path)
    print(f"VALID threshold={chosen['threshold']:.2f} P={chosen['precision']:.4f} R={chosen['recall']:.4f} F2={chosen['f2']:.4f}; lock={lock_sha}", flush=True)

    # TEST is opened only after the VALID-selected threshold and lock have been written.
    if sha256_file(CHECKPOINT_PATH) != ckpt_sha or sha256_file(lock_path) != lock_sha:
        raise RuntimeError('Checkpoint or evaluation lock changed before TEST.')
    test_records, test_cache, test_meta = collect_split(
        model, device, 'test', data_root, config, dataset_yaml, CHECKPOINT_PATH, ckpt_sha)
    test_out = OUTPUT_ROOT / 'E6' / 'test'
    test_result = evaluate(test_records, chosen['threshold'], names, 'test', ckpt_sha,
                           sha256_file(test_cache), data_sha)
    if sha256_file(CHECKPOINT_PATH) != ckpt_sha or sha256_file(lock_path) != lock_sha:
        raise RuntimeError('Checkpoint or evaluation lock changed during TEST.')
    test_result['status'] = 'FINAL_AFTER_VALID_LOCK'
    test_result['lock_sha256'] = lock_sha
    write_metrics(test_result, test_out)
    (test_out / 'metadata.json').write_text(json.dumps({**test_meta, 'lock_sha256': lock_sha}, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (valid_out / 'metadata.json').write_text(json.dumps(valid_meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    summary = []
    for result in (valid_result, test_result):
        summary.append({k: result[k] for k in ('model','split','images','operating_threshold','map50','map50_95','precision','recall','f2')})
    summary_csv = OUTPUT_ROOT / 'summary_E6.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (OUTPUT_ROOT / 'summary_E6.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    archive = shutil.make_archive(
        str(OUTPUT_ROOT.parent / 'evaluation_E6_30epochs_complete_results'), 'zip',
        root_dir=OUTPUT_ROOT.parent, base_dir=OUTPUT_ROOT.name,
    )
    print('\nE6 VALID:', {k: valid_result[k] for k in ('images','operating_threshold','map50','map50_95','precision','recall','f2')})
    print('E6 TEST :', {k: test_result[k] for k in ('images','operating_threshold','map50','map50_95','precision','recall','f2')})
    print('Download ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
