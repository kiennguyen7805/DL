"""Independent extensions of the audited epoch-10 runs; no fresh training fallback.

The notebook builder embeds this file, HFStore, and the existing RODDataset.
Imports of GPU packages are deliberately deferred for CPU contract tests.
"""
import csv
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath


MILESTONES = (10, 20, 30)
TARGET_EPOCHS = 30


def checked_member(name):
    if '\\' in name or ':' in name or name.startswith('/'):
        raise ValueError('Unsafe bundle member: ' + name)
    path = PurePosixPath(name)
    if '..' in path.parts or not path.parts:
        raise ValueError('Unsafe bundle member: ' + name)
    return name


def next_stage(completed):
    if type(completed) is not int or not 10 <= completed <= 30:
        raise ValueError('Expected 10..30 completed epochs')
    return next((epoch for epoch in MILESTONES if epoch > completed), None)


def read_csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]


def validate_metrics(rows, completed):
    if [int(r['epoch']) for r in rows] != list(range(1, completed + 1)):
        raise RuntimeError('Metrics must contain exactly contiguous epochs 1..completed')
    return rows


def atomic_copy(source, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + '.tmp')
    shutil.copyfile(source, temporary)
    os.replace(temporary, dest)


def extract_bundle(path, destination, experiment, hash_file):
    """Only extract the flat, hashed payload of a locally audited bundle."""
    destination = Path(destination)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 12:
            raise RuntimeError('Duplicate/too many bundle members')
        for entry in archive.infolist():
            checked_member(entry.filename)
            if '/' in entry.filename or entry.is_dir() or entry.file_size > 2 * 1024**3:
                raise RuntimeError('Invalid bundle member')
        manifest = json.loads(archive.read('source_manifest.json'))
        if manifest.get('experiment') != experiment or manifest.get('completed_epoch') != 10:
            raise RuntimeError('Wrong source model or epoch')
        hashes = manifest.get('sha256', {})
        if set(names) != set(hashes) | {'source_manifest.json'} or not hashes:
            raise RuntimeError('Unexpected bundle artifacts')
        if archive.testzip() is not None:
            raise RuntimeError('Bundle CRC failed')
        destination.mkdir(parents=True, exist_ok=True)
        for name, expected in hashes.items():
            target = destination / name
            if target.exists():
                if hash_file(target) != expected:
                    raise RuntimeError('Refusing to overwrite different source artifact')
                continue
            temp = target.with_suffix(target.suffix + '.tmp')
            with archive.open(name) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if hash_file(temp) != expected:
                temp.unlink()
                raise RuntimeError('Bundle SHA256 failed: ' + name)
            os.replace(temp, target)
    return manifest


def check_yolo(payload, experiment, names, frozen, expected_epoch=None):
    completed = int(payload.get('epoch', -1)) + 1
    if not 10 <= completed <= 30 or (expected_epoch is not None and completed != expected_epoch):
        raise RuntimeError('Wrong YOLO checkpoint epoch')
    optimizer = payload.get('optimizer')
    if not isinstance(optimizer, dict) or not optimizer.get('state') or not optimizer.get('param_groups'):
        raise RuntimeError('YOLO checkpoint is stripped or has no optimizer state')
    if not isinstance(payload.get('scaler'), dict) or payload.get('ema') is None:
        raise RuntimeError('Missing YOLO scaler/EMA')
    if payload.get('version') != '8.4.165':
        raise RuntimeError('Checkpoint Ultralytics version differs from audited 8.4.165')
    actual = payload.get('train_args', {})
    excluded = {'task', 'mode', 'model', 'data', 'device', 'project', 'name', 'exist_ok',
                'save_period', 'experiment_id', 'main_budget_epochs'}
    for key, value in frozen.items():
        if key not in excluded and actual.get(key) != value:
            raise RuntimeError('YOLO training argument mismatch: ' + key)
    if actual.get('epochs') != 30 or actual.get('cos_lr') is not False:
        raise RuntimeError('Expected original 30-epoch linear LR schedule')
    ema = payload['ema']
    scale = 'n' if experiment == 'E3' else 's'
    ckpt_names = ema.names
    ordered = [ckpt_names[i] if i in ckpt_names else ckpt_names[str(i)] for i in range(25)] if isinstance(ckpt_names, dict) else list(ckpt_names)
    if ema.yaml.get('scale') != scale or getattr(ema, 'nc', None) != 25 or ordered != names:
        raise RuntimeError('Wrong YOLO architecture/classes')
    return completed


def check_e6(saved, expected_hash, dataset_hash, manifest=None, train_count=18151):
    required = ('model', 'optimizer', 'scheduler', 'scaler', 'python_rng', 'numpy_rng',
                'torch_rng', 'cuda_rng', 'metrics_rows', 'best_map', 'loss_sum', 'loss_count')
    if any(k not in saved for k in required):
        raise RuntimeError('Missing E6 full recovery state')
    if saved.get('config_hash') != expected_hash or saved.get('dataset_hash') != dataset_hash:
        raise RuntimeError('E6 source config/dataset mismatch')
    done, ep, offset = (saved.get(k) for k in ('completed_epoch', 'current_epoch', 'next_sample'))
    if (any(type(x) is not int for x in (done, ep, offset)) or
            not 10 <= done <= 30 or ep != done or not 0 <= offset <= train_count or
            (offset % 2 and offset != train_count) or (done == 30 and offset)):
        raise RuntimeError('Invalid E6 epoch/sample progress')
    validate_metrics(saved['metrics_rows'], done)
    scheduler = saved['scheduler']
    expected_lr = 0.005 * 0.1 ** (done // 3)
    if (scheduler.get('last_epoch') != done or scheduler.get('step_size') != 3 or
            scheduler.get('gamma') != 0.1 or scheduler.get('base_lrs') != [0.005] or
            not math.isclose(scheduler['_last_lr'][0], expected_lr, rel_tol=1e-8)):
        raise RuntimeError('E6 scheduler state does not match original StepLR(3, 0.1)')
    groups = saved['optimizer'].get('param_groups', [])
    if not saved['optimizer'].get('state') or not groups or any(
            not math.isclose(g['lr'], expected_lr, rel_tol=1e-8) or
            g.get('momentum') != 0.9 or g.get('weight_decay') != 0.0005 for g in groups):
        raise RuntimeError('E6 optimizer mismatch')
    if saved['loss_count'] != (offset + 1) // 2 or not math.isfinite(float(saved['loss_sum'])):
        raise RuntimeError('E6 loss/sample progress mismatch')
    if saved['model']['roi_heads.box_predictor.cls_score.weight'].shape[0] != 26:
        raise RuntimeError('E6 must have 26 outputs including background')
    if manifest and (manifest['completed_epoch'] != done or manifest.get('next_sample', 0) != offset):
        raise RuntimeError('E6 manifest/payload progress mismatch')
    return done, offset


def runtime_environment():
    import torch
    import torchvision
    return {'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}


def save_milestone(work, files, epoch, store):
    """Keep exact-epoch weights distinct from cumulative-best; upload immutable index."""
    if epoch not in MILESTONES:
        return
    index = {'experiment': EXPERIMENT, 'epoch': epoch, 'sha256': {}}
    for name, source in files.items():
        dest = work / 'milestones' / f'epoch{epoch:03d}' / name
        if dest.exists() and sha256_file(dest) != sha256_file(source):
            raise RuntimeError('Refusing to overwrite milestone: ' + str(dest))
        if not dest.exists():
            atomic_copy(source, dest)
        index['sha256'][name] = sha256_file(dest)
    latest = json.loads((work / 'last_hf_manifest.json').read_text(encoding='utf-8'))
    index['snapshot'] = latest['snapshot']
    path = work / 'milestones' / f'epoch{epoch:03d}' / 'index.json'
    atomic_json(path, index)
    store._retry(lambda: store.api.upload_file(path_or_fileobj=str(path),
        path_in_repo=store.folder + f'/milestones/epoch{epoch:03d}.json',
        repo_id=HF_REPO_ID, repo_type=store.repo_type, token=store.token,
        commit_message=f'{EXPERIMENT}: exact epoch {epoch}'))
    remote = store._download(store.folder + f'/milestones/epoch{epoch:03d}.json')
    if sha256_file(remote) != sha256_file(path):
        raise RuntimeError('Milestone index upload verification failed')


def discover_source(input_root, destination, experiment, hash_file):
    """Kaggle may expose the uploaded ZIP as an already extracted directory."""
    root = Path(input_root)
    candidates = list(root.rglob(experiment.lower() + '_resume_epoch010.zip'))
    observed = []
    for path in root.rglob('source_manifest.json'):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        observed.append(str(path) + ' [' + str(manifest.get('experiment')) + ']')
        if manifest.get('experiment') == experiment:
            candidates.append(path)
    if len(candidates) != 1:
        mounted = [p.name for p in root.iterdir()] if root.is_dir() else []
        raise RuntimeError(f'{experiment}: expected exactly one ZIP or extracted source bundle, found {len(candidates)}. '
                           f'Candidates: {[str(p) for p in candidates]}; manifests: {observed}; '
                           f'mounted inputs: {mounted}. Attach the audited {experiment.lower()}_resume_epoch010 bundle. '
                           'No training was started; never restart at epoch 1.')
    path = candidates[0]
    print('Using audited source:', path, flush=True)
    if path.suffix.lower() == '.zip':
        return extract_bundle(path, destination, experiment, hash_file)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('completed_epoch') != 10:
        raise RuntimeError('Wrong extracted source epoch')
    hashes = manifest.get('sha256', {})
    if not isinstance(hashes, dict) or not hashes or len(hashes) > 10:
        raise RuntimeError('Extracted source manifest missing hashes')
    # Validate every source before copying any file, and prohibit path traversal/symlinks.
    for name, expected in hashes.items():
        checked_member(name)
        if '/' in name or (path.parent / name).resolve().parent != path.parent.resolve():
            raise RuntimeError('Unsafe extracted artifact name')
        source = path.parent / name
        if not source.is_file() or hash_file(source) != expected:
            raise RuntimeError('Extracted source missing artifact or SHA256 mismatch: ' + name)
    for name, expected in hashes.items():
        target = Path(destination) / name
        if target.exists() and hash_file(target) != expected:
            raise RuntimeError('Refusing to overwrite different extracted artifact: ' + name)
        if not target.exists():
            atomic_copy(path.parent / name, target)
        if hash_file(target) != expected:
            raise RuntimeError('Copied artifact SHA256 mismatch: ' + name)
    return manifest


def restore_legacy_hf(api, token, kind, experiment, origin, work):
    """Read the verified epoch-10 HF folder; never write its pointer/artifacts."""
    folder, model = {'E4': ('YOLO 11s', 'yolo11s.pt'),
                     'E6': ('Faster R-CNN', 'fasterrcnn_resnet50_fpn_v2')}[experiment]
    legacy = HFStore(api, token, kind, folder, model, LEGACY_CONFIG_SHA256,
                     origin['dataset_yaml_sha256'], work / 'legacy_hf_readonly', max_epoch=10)
    restored = legacy.restore()
    if restored is None:
        raise RuntimeError('No epoch-10 HF snapshot in ' + folder + '; refusing fresh training')
    manifest, files = restored
    if manifest['completed_epoch'] != 10 or manifest.get('next_sample', 0) != 0:
        raise RuntimeError('Expected complete epoch 10 in original HF folder')
    required = ('checkpoint.pt', 'best_weights.pth') if experiment == 'E6' else (
        'recovery_full.pt', 'best.pt', 'results.csv', 'args.yaml')
    if any(name not in files for name in required):
        raise RuntimeError('Original HF snapshot lacks full recovery/best/metrics')
    # Cross-check full recovery against the actual locally audited checkpoint.
    recovery_name = required[0]
    if sha256_file(files[recovery_name]) != origin['sha256'][recovery_name]:
        raise RuntimeError('HF recovery differs from audited local epoch-10 recovery; review provenance before resuming')
    atomic_json(work / 'legacy_source_manifest.json', manifest)
    print('Downloaded verified epoch 10 from', HF_REPO_ID, folder, flush=True)
    return manifest, files


def run_extension():
    import torch
    import yaml
    from kaggle_secrets import UserSecretsClient
    if not torch.cuda.is_available():
        raise RuntimeError('Select one Kaggle CUDA GPU before running')
    work = Path('/kaggle/working') / (EXPERIMENT.lower() + '_extension30')
    work.mkdir(parents=True, exist_ok=True)
    source = work / 'source_epoch010'
    # E3 came from downloaded Kaggle Output. E4/E6 already have full-state HF snapshots.
    origin = (discover_source('/kaggle/input', source, EXPERIMENT, sha256_file)
              if EXPERIMENT == 'E3' else SOURCE_ORIGIN)
    if sha256_file(source_yaml) != origin['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML differs from audited local Final')
    if class_names != origin['class_names']:
        raise RuntimeError('Dataset class mapping mismatch')
    identity = {'experiment': EXPERIMENT, 'target_epochs': 30, 'milestones': list(MILESTONES),
                'source_manifest': origin, 'protocol': 'extension30-v1',
                'scheduler_policy': 'preserve_original', 'test_used': False}
    config_hash, dataset_hash = fingerprint(identity), sha256_file(source_yaml)
    atomic_json(work / 'config_audit.json', identity)
    environment = runtime_environment()
    atomic_json(work / 'environment.json', environment)
    token = UserSecretsClient().get_secret(HF_SECRET_NAME)
    api, kind = connect_hf(token)
    model_name = {'E3': 'yolo11n.pt', 'E4': 'yolo11s.pt', 'E6': 'fasterrcnn_resnet50_fpn_v2'}[EXPERIMENT]
    store = HFStore(api, token, kind, EXPERIMENT + '_30epochs', model_name,
                    config_hash, dataset_hash, work / 'hf_backup', max_epoch=30)
    restored = store.restore()
    if restored:
        manifest, files = restored
        atomic_json(work / 'last_hf_manifest.json', manifest)
        previous_env = json.loads(files['environment.json'].read_text(encoding='utf-8'))
        for key in ('torch', 'torchvision', 'cuda'):
            if previous_env[key] != environment[key]:
                raise RuntimeError('Runtime changed since extension snapshot: ' + key)
    else:
        if EXPERIMENT == 'E3':
            files = {name: source / name for name in origin['sha256']}
            manifest = {'completed_epoch': 10, 'next_sample': 0}
        else:
            manifest, files = restore_legacy_hf(api, token, kind, EXPERIMENT, origin, work)
    deadline = time.monotonic() + 9 * 3600
    atomic_json(work / 'training_status.json', {'experiment': EXPERIMENT,
        'completed_epoch': manifest['completed_epoch'], 'target_epoch': 30,
        'status': 'PREFLIGHT_PASS', 'next_stage': next_stage(manifest['completed_epoch'])})
    if EXPERIMENT == 'E6':
        run_e6(work, source, origin, files, manifest, store, deadline)
    else:
        run_yolo(work, source, files, manifest, store, deadline)
    latest = json.loads((work / 'last_hf_manifest.json').read_text(encoding='utf-8'))
    atomic_json(work / 'training_status.json', {'experiment': EXPERIMENT,
        'completed_epoch': latest['completed_epoch'], 'next_sample': latest.get('next_sample', 0),
        'target_epoch': 30, 'status': 'COMPLETE' if latest['is_complete'] else 'CONTINUE',
        'next_stage': next_stage(latest['completed_epoch'])})
    print('Verified HF status:', latest['completed_epoch'], '/30;',
          'COMPLETE' if latest['is_complete'] else 'CONTINUE', flush=True)


def run_yolo(work, source, files, manifest, store, deadline):
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    if ultralytics.__version__ != '8.4.165':
        raise RuntimeError('Requires Ultralytics 8.4.165')
    frozen = yaml.safe_load(FROZEN_YAML)
    checkpoint = torch.load(files['recovery_full.pt'], map_location='cpu', weights_only=False)
    done = check_yolo(checkpoint, EXPERIMENT, class_names, frozen, manifest['completed_epoch'])
    run = work / 'train'
    run.mkdir(exist_ok=True)
    rows = read_csv_rows(files['results.csv'])
    validate_metrics(rows, done)
    atomic_copy(files['results.csv'], run / 'results.csv')
    resume = run / 'weights' / 'last.pt'
    atomic_copy(files['recovery_full.pt'], resume)
    atomic_copy(files['best.pt'], run / 'weights' / 'best.pt')
    if done == 30:
        print(EXPERIMENT, '30/30 COMPLETE; checkpoint and metrics PASS, no training')
        return
    boundary = next_stage(done)
    print(EXPERIMENT, f'resume epoch {done + 1}; stop at {boundary}; original LR horizon=30', flush=True)
    # check_resume restores historical args, including obsolete paths. Relocate only
    # I/O after the framework restores args, before model/dataloader construction.
    class RelocatedTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            if not self.resume:
                raise RuntimeError('Expected true resume, fresh training prohibited')
            self.args.data = str(working_yaml)
            self.args.project = str(work)
            self.args.name = 'train'
            self.args.save_dir = str(run)
            self.args.exist_ok = True
            self.args.device = '0'
            self.args.split = 'val'

    def verify_start(trainer):
        if trainer.start_epoch != done or trainer.epochs != 30:
            raise RuntimeError('Resume start/scheduler horizon mismatch')
        # Framework reconstructs LambdaLR from original args and restored epoch.
        for key in ('lr0', 'lrf', 'cos_lr', 'close_mosaic', 'batch', 'imgsz'):
            if getattr(trainer.args, key) != checkpoint['train_args'][key]:
                raise RuntimeError('Effective argument changed: ' + key)
        print('Resume preflight PASS:', done + 1, 'LR groups', [g['lr'] for g in trainer.optimizer.param_groups], flush=True)

    def backup(trainer):
        epoch = int(trainer.epoch) + 1
        payload = torch.load(trainer.last, map_location='cpu', weights_only=False)
        check_yolo(payload, EXPERIMENT, class_names, frozen, epoch)
        validate_metrics(read_csv_rows(trainer.csv), epoch)
        full = work / 'recovery_full.pt'
        atomic_copy(trainer.last, full)  # before final_eval strips optimizer
        artifacts = {'recovery_full.pt': full, 'best.pt': Path(trainer.best),
                     'results.csv': Path(trainer.csv), 'args.yaml': run / 'args.yaml',
                     'config_audit.json': work / 'config_audit.json',
                     'environment.json': work / 'environment.json'}
        status = store.save(artifacts, epoch, {'target_epochs': 30,
            'next_stage': next_stage(epoch), 'lr_groups': [g['lr'] for g in trainer.optimizer.param_groups]})
        atomic_json(work / 'last_hf_manifest.json', status)
        save_milestone(work, artifacts, epoch, store)
        trainer.stop = trainer.stop or epoch >= boundary or time.monotonic() >= deadline
        print(EXPERIMENT, epoch, '/30 saved and verified on HF', flush=True)

    if done == 10:
        bootstrap = {'recovery_full.pt': resume, 'best.pt': run / 'weights/best.pt',
                     'results.csv': run / 'results.csv', 'args.yaml': files['args.yaml'],
                     'config_audit.json': work / 'config_audit.json', 'environment.json': work / 'environment.json'}
        atomic_json(work / 'last_hf_manifest.json', store.save(bootstrap, 10, {'target_epochs': 30}))
        save_milestone(work, bootstrap, 10, store)
    model = YOLO(str(resume))
    model.add_callback('on_train_start', verify_start)
    model.add_callback('on_model_save', backup)
    model.train(trainer=RelocatedTrainer, resume=True, data=str(working_yaml), device=0)


def run_e6(work, source, origin, files, manifest, store, deadline):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    # Dataset/collate are embedded unchanged from the actual E6 training runner.
    train_ds, val_ds = RODDataset(TRAIN_IMAGES), RODDataset(VAL_IMAGES)
    saved = torch.load(files['checkpoint.pt'], map_location='cpu', weights_only=False)
    done, offset = check_e6(saved, origin['original_config_hash'], origin['dataset_yaml_sha256'], manifest)
    if done == 30:
        print('E6 30/30 COMPLETE; checkpoint and metrics PASS, no training')
        return
    boundary = next_stage(done)
    net = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=640, max_size=960)
    net.roi_heads.box_predictor = FastRCNNPredictor(net.roi_heads.box_predictor.cls_score.in_features, 26)
    net.load_state_dict(saved['model'], strict=True)
    net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    scaler.load_state_dict(saved['scaler'])
    torch.backends.cudnn.benchmark = False
    rows, best_map = saved['metrics_rows'], float(saved['best_map'])
    atomic_copy(files['best_weights.pth'], work / 'best_weights.pth')
    random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    print('E6 resume:', done + 1, 'sample offset', offset, 'LR', scheduler.get_last_lr(),
          'stop at', boundary, '(original StepLR preserved)', flush=True)

    def persist(ep, off, loss_sum, loss_count):
        payload = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'current_epoch': ep, 'completed_epoch': ep, 'next_sample': off,
            'loss_sum': loss_sum, 'loss_count': loss_count, 'metrics_rows': rows, 'best_map': best_map,
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'config_hash': origin['original_config_hash'], 'dataset_hash': origin['dataset_yaml_sha256']}
        check_e6(payload, origin['original_config_hash'], origin['dataset_yaml_sha256'])
        path = work / 'checkpoint.pt'
        torch.save(payload, path.with_suffix('.tmp'))
        os.replace(path.with_suffix('.tmp'), path)
        with (work / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        artifacts = {n: work / n for n in ('checkpoint.pt', 'best_weights.pth', 'metrics.csv', 'config_audit.json', 'environment.json')}
        status = store.save(artifacts, ep, {'next_sample': off, 'target_epochs': 30, 'next_stage': next_stage(ep)})
        atomic_json(work / 'last_hf_manifest.json', status)
        if off == 0:
            save_milestone(work, artifacts, ep, store)

    if done == 10 and offset == 0:
        persist(10, 0, 0.0, 0)
    for ep in range(done, boundary):
        permutation = np.random.default_rng(10000 + ep).permutation(len(train_ds)).tolist()
        off = offset if ep == done else 0
        loss_sum = saved['loss_sum'] if ep == done else 0.0
        count = saved['loss_count'] if ep == done else 0
        loader = DataLoader(Subset(train_ds, permutation[off:]), batch_size=2, shuffle=False,
            num_workers=2, collate_fn=collate, pin_memory=True,
            generator=torch.Generator().manual_seed(20000 + ep))
        net.train()
        for images, targets in loader:
            images = [im.cuda(non_blocking=True) for im in images]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = sum(net(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError('E6 loss NaN/Inf')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            off += len(images); count += 1; loss_sum += float(loss.detach().cpu())
            if count % 200 == 0:
                print(f'E6 epoch {ep + 1}/30 samples {off}/{len(train_ds)}', flush=True)
            if count % 1800 == 0 or time.monotonic() >= deadline:
                persist(ep, off, loss_sum, count)
            if time.monotonic() >= deadline:
                print('Safe stop: resume same epoch from verified batch offset next session', flush=True)
                return
        if off != len(train_ds):
            raise RuntimeError('Incomplete epoch')
        metric = MeanAveragePrecision(iou_type='bbox')
        net.eval()
        with torch.inference_mode():
            loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2,
                collate_fn=collate, generator=torch.Generator().manual_seed(8000))
            for images, targets in loader:
                predictions = net([im.cuda() for im in images])
                metric.update([{k: v.cpu() for k, v in p.items() if k in ('boxes','scores','labels')} for p in predictions],
                              [{k: v.cpu() for k, v in t.items() if k in ('boxes','labels')} for t in targets])
        scores = metric.compute()
        scheduler.step()  # exactly once after complete epoch + VALID
        score = float(scores['map'])
        if score > best_map:
            best_map = score
            tmp = work / 'best_weights.tmp'
            torch.save(net.state_dict(), tmp); os.replace(tmp, work / 'best_weights.pth')
        rows.append({'epoch': ep + 1, 'map50': float(scores['map_50']), 'map5095': score,
                     'mar100': float(scores['mar_100']), 'loss_mean': loss_sum / count})
        persist(ep + 1, 0, 0.0, 0)
        print('E6 completed', ep + 1, '/30; VALID mAP50-95', score, flush=True)
        if time.monotonic() >= deadline:
            return
