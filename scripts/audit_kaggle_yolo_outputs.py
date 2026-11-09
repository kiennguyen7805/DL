"""Read trusted user-downloaded Kaggle ZIPs and salvage verified recovery files.

No training, inference, dataset changes, remote calls, or checkpoint overwrites.
Run from the project root with a Python environment containing torch/ultralytics.
"""
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/training_30epochs/recovered'


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
        result.update(chunk)
    return result.hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise RuntimeError('Unsafe ZIP member: ' + name)


def audit(experiment, archive_name, prefix, run_prefix, scale):
    import torch
    import yaml
    from ultralytics import YOLO
    source = ROOT / archive_name
    with source.open('rb') as stream:
        archive_sha = digest(stream)
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError('Duplicate ZIP members')
        for name in names:
            safe_member(name)
        if archive.testzip():
            raise RuntimeError('ZIP CRC failed')
        staged = [(name,json.loads(archive.read(name))) for name in names
                  if name.startswith(prefix+'/hf_backup/snapshots/') and name.endswith('/manifest.json')]
        if not staged:
            raise RuntimeError('No failed-upload staging manifest')
        manifest_name, manifest = max(staged, key=lambda item:item[1]['completed_epoch'])
        epoch = manifest['completed_epoch']
        stage = manifest_name.rsplit('/',1)[0]
        artifact_hashes = {}
        for name, expected in manifest['sha256'].items():
            safe_member(name)
            if '/' in name:
                raise RuntimeError('Expected flat staged artifact')
            with archive.open(stage+'/'+name) as stream:
                artifact_hashes[name]=digest(stream)
            if artifact_hashes[name]!=expected:
                raise RuntimeError('Staged artifact checksum mismatch: '+name)
        current = json.loads(archive.read(prefix+'/last_hf_manifest.json'))
        rows = [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(
            io.StringIO(archive.read(run_prefix+'/results.csv').decode('utf-8-sig')))]
        if [int(r['epoch']) for r in rows]!=list(range(1,epoch+1)):
            raise RuntimeError('Non-contiguous metrics')
        if archive.read(run_prefix+'/results.csv') != archive.read(stage+'/results.csv'):
            raise RuntimeError('Run/staging metrics differ')
        environment=json.loads(archive.read(prefix+'/environment.json'))
        folder=OUT/f'{experiment}_epoch{epoch:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        paths={}
        # Deduplicate identical checkpoint bytes within the salvaged package.
        canonical={}
        for name in ('recovery_full.pt','best.pt','args.yaml','results.csv','config.json','environment.json'):
            expected=artifact_hashes[name]
            if expected in canonical:
                paths[name]=canonical[expected]
                continue
            path=folder/name
            if path.exists():
                with path.open('rb') as stream:
                    if digest(stream)!=expected:raise RuntimeError('Refusing overwrite: '+str(path))
            else:
                temp=path.with_suffix(path.suffix+'.tmp')
                with archive.open(stage+'/'+name) as src, temp.open('wb') as dest:
                    shutil.copyfileobj(src,dest)
                with temp.open('rb') as stream:
                    if digest(stream)!=expected:raise RuntimeError('Extraction checksum failed')
                temp.replace(path)
            canonical[expected]=str(path.relative_to(ROOT))
            paths[name]=canonical[expected]
        checkpoint=torch.load(ROOT/paths['recovery_full.pt'],map_location='cpu',weights_only=False)
        if int(checkpoint['epoch'])+1!=epoch or checkpoint['train_args']['epochs']!=30:
            raise RuntimeError('Checkpoint epoch/schedule mismatch')
        if not checkpoint['optimizer']['state'] or not checkpoint['optimizer']['param_groups']:
            raise RuntimeError('Missing optimizer')
        if not isinstance(checkpoint['scaler'],dict) or checkpoint.get('ema') is None:
            raise RuntimeError('Missing scaler/EMA')
        ema=checkpoint['ema']
        if ema.yaml.get('scale')!=scale or ema.nc!=25:
            raise RuntimeError('Wrong architecture/class count')
        expected_names=yaml.safe_load((ROOT/'data/processed/rod_detection_final/data.yaml').read_text(encoding='utf-8'))['names']
        actual_names=ema.names
        actual_names=[actual_names[i] for i in range(25)] if isinstance(actual_names,dict) else list(actual_names)
        if actual_names!=expected_names:raise RuntimeError('Class mapping mismatch')
        if any(not torch.isfinite(v).all() for v in ema.state_dict().values() if v.is_floating_point()):
            raise RuntimeError('Nonfinite model weights')
        native=checkpoint['train_results']
        if list(native['epoch'])!=list(range(1,epoch+1)):
            raise RuntimeError('Checkpoint metrics mismatch')
        for key in ('metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)'):
            if not math.isclose(float(native[key][-1]),float(rows[-1][key]),abs_tol=1e-8):
                raise RuntimeError('CSV/checkpoint metric mismatch: '+key)
        for name in (prefix+'/recovery_full.pt',run_prefix+'/weights/last.pt'):
            with archive.open(name) as stream:
                if digest(stream)!=artifact_hashes['recovery_full.pt']:
                    raise RuntimeError('Recovery and run checkpoint copies differ')
        loaded=YOLO(str(ROOT/paths['recovery_full.pt']))
        if loaded.task!='detect' or len(loaded.names)!=25:
            raise RuntimeError('Actual model load failed')
        result={'experiment':experiment,'archive':archive_name,'archive_sha256':archive_sha,
            'zip_crc':'PASS','staging_sha256':'PASS','checkpoint_load_cpu':'PASS',
            'checkpoint_epoch':epoch,'resume_next_epoch':epoch+1,'original_schedule_epochs':30,
            'optimizer_states':len(checkpoint['optimizer']['state']),'scaler_present':True,
            'ema_present':True,'class_mapping':'PASS','finite_weights':'PASS',
            'metrics_contiguous_epochs':len(rows),'metrics_last_epoch_valid_native':rows[-1],
            'local_record_last_verified_hf_epoch':current['completed_epoch'],
            'failed_upload_manifest':manifest,'artifact_sha256':artifact_hashes,
            'recovered_paths':paths,'environment':environment,
            'hf_current_remote_state':'NOT QUERIED',
            'upload_error_reason':'MISSING: original log suppressed HTTP response; no log/HTML/ipynb in archive',
            'training_run_locally':False,'test_split_accessed':False}
        for name,value in [('failed_upload_manifest.json',manifest),('last_verified_hf_manifest.json',current),('recovery_audit.json',result)]:
            path=folder/name
            text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
            if path.exists() and path.read_text(encoding='utf-8')!=text:
                raise RuntimeError('Refusing report overwrite: '+str(path))
            path.write_text(text,encoding='utf-8')
        print(experiment,'recovered epoch',epoch,'next',epoch+1,'CPU model load PASS',flush=True)
        del loaded,checkpoint
        return result


def main():
    results=[audit('E3','results_nano.zip','rod_e4_yolo11n_hf','rod_main/E3_pretrained','n'),
             audit('E4','results_small.zip','rod_e4_yolo11s_hf','rod_e4_yolo11s_hf/train','s')]
    report=['# Recovery E3/E4 from user Kaggle Output ZIPs','',
        'Read-only audit of sources; selectively extracted verified unique artifacts. No training, TEST, checkpoint replacement or HF calls.','',
        '| Run | Local epoch | Last verified HF epoch in saved evidence | Next epoch | P VALID | R VALID | mAP50 VALID | mAP50-95 VALID |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m=r['metrics_last_epoch_valid_native']
        report.append('| '+r['experiment']+' | '+str(r['checkpoint_epoch'])+' | '+str(r['local_record_last_verified_hf_epoch'])+' | '+str(r['resume_next_epoch'])+' | '+
            ' | '.join(m[k] for k in ('metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)'))+' |')
    report += ['', 'Both ZIP CRC, staged manifest SHA256, contiguous CSV, actual CPU YOLO load, 25-class mapping, finite weights, optimizer and EMA/scaler checks PASS.',
        'The epoch-13 snapshot is local staging, not evidence of successful remote upload. Last verified remote epoch is 12 only according to these downloaded manifests; live HF was not queried.',
        'Metrics above are native Ultralytics VALID at epoch 13, not common evaluator TEST results. Do not replace baseline reports/calibration with them.',
        'Original server-side cause of BadRequestError remains MISSING. Archives contain no notebook/log HTML with the underlying response. Do not guess storage quota or permissions.',
        'Both full recovery and best checkpoint are byte-identical at epoch 13 in each ZIP; recovered_paths maps both roles to the same canonical file without a second large copy.',
        'Resume must use epoch 13 recovery (next epoch 14). The old notebook may only find epoch 12 on HF; do not restart it without first providing the recovered progress.',
        '', 'Detailed sources, source ZIP SHA256, checkpoint hashes, manifests and actual metrics: E3_epoch013/recovery_audit.json and E4_epoch013/recovery_audit.json.', '']
    (OUT/'RECOVERY_REPORT.md').write_text('\n'.join(report),encoding='utf-8')


if __name__=='__main__':main()
