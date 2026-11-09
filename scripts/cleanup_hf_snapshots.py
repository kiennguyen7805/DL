"""Manifest-driven HF pruning. No torch, GPU, training or model deserialization.

BACKUP is read-only remotely. APPLY requires the verified backup uploaded as
Kaggle Input and unchanged repository HEAD. Unknown paths are always retained.
"""
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

REPO = 'iraprp/deeplearning'
KIND = 'dataset'
FOLDERS = {'YOLO 11n':'yolo11n.pt', 'YOLO 11s':'yolo11s.pt',
           'Faster R-CNN':'fasterrcnn_resnet50_fpn_v2',
           'E3_30epochs':'yolo11n.pt', 'E4_30epochs':'yolo11s.pt',
           'E6_30epochs':'fasterrcnn_resnet50_fpn_v2'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def valid_path(path):
    p=PurePosixPath(path)
    if not path or p.is_absolute() or '..' in p.parts or '\\' in path or ':' in path:
        raise ValueError('Unsafe path: '+path)
    return path


def json_bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2).encode('utf-8')


def choose_retention(records, pointers):
    """Keep pointer targets, latest per config lineage, best, complete milestones.

    Best is taken from the latest snapshot of each config/dataset lineage. Also
    retain the most advanced snapshot containing best if latest lacks that file.
    Equal progress: prefer pointer target, otherwise deterministic snapshot name.
    """
    keep=set(pointers.values())
    lineages={}
    for snap,m in records.items():
        key=(m['folder'],m['model'],m['config_sha256'],m['dataset_sha256'])
        lineages.setdefault(key,[]).append(snap)
    for snapshots in lineages.values():
        def progress(s):
            m=records[s]
            return m['completed_epoch'],m.get('next_sample',0),s in keep,s
        keep.add(max(snapshots,key=progress))
        best=[s for s in snapshots if {'best.pt','best_weights.pth'} & set(records[s]['sha256'])]
        if best:keep.add(max(best,key=progress))
        for epoch in (10,20,30):
            milestone=[s for s in snapshots if records[s]['completed_epoch']==epoch
                       and records[s].get('next_sample',0)==0]
            if milestone:keep.add(max(milestone,key=progress))
    return keep


def audit(api,download,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    head=api.repo_info(repo_id=REPO,repo_type=KIND).sha
    entries={x.path:x.size for x in api.list_repo_tree(repo_id=REPO,repo_type=KIND,
             revision=head,recursive=True) if hasattr(x,'size')}
    records={};raw_metadata={};unknown={};pointers={}
    def read_json(remote):
        if entries.get(remote,3*1024**2)>2*1024**2:raise ValueError('Unexpected large JSON: '+remote)
        file=download(repo_id=REPO,repo_type=KIND,revision=head,filename=remote,
                      cache_dir=str(output/'metadata_cache'))
        raw=Path(file).read_bytes();raw_metadata[remote]=raw
        return json.loads(raw)
    for folder,model in FOLDERS.items():
        prefix=folder+'/snapshots/'
        groups={p.rsplit('/',1)[0] for p in entries if p.startswith(prefix)}
        for snap in sorted(groups):
            try:
                if snap.count('/')!=2:raise ValueError('Unexpected nested snapshot')
                m=read_json(snap+'/manifest.json')
                epoch=m.get('completed_epoch');offset=m.get('next_sample',0)
                if (m.get('schema')!=2 or m.get('snapshot')!=snap or m.get('folder')!=folder
                    or m.get('model')!=model or type(epoch) is not int or not 0<=epoch<=30
                    or type(offset) is not int or offset<0):
                    raise ValueError('Unknown manifest/progress schema')
                for field in ('config_sha256','dataset_sha256'):
                    if not re.fullmatch('[0-9a-f]{64}',m.get(field,'')):raise ValueError('Missing fingerprint')
                hashes=m.get('sha256',{})
                if not hashes or not ({'checkpoint.pt','recovery_full.pt'} & set(hashes)):
                    raise ValueError('No recovery checkpoint')
                expected={snap+'/manifest.json'}
                for name,sha in hashes.items():
                    valid_path(name)
                    if '/' in name or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('Unsafe artifact/hash')
                    expected.add(snap+'/'+name)
                actual={p for p in entries if p.startswith(snap+'/')}
                if expected!=actual:raise ValueError('Missing or unrecognized files; keep entire snapshot')
                records[snap]=m
            except (ValueError,KeyError,TypeError) as error:
                unknown[snap]=str(error)
        pointer=folder+'/latest.json'
        if pointer in entries:
            p=read_json(pointer);snap=p.get('snapshot')
            if snap not in records:raise RuntimeError('STOP: latest points to an unknown/incomplete snapshot: '+folder)
            raw=raw_metadata[snap+'/manifest.json']
            if hashlib.sha256(raw).hexdigest()!=p.get('manifest_sha256'):
                raise RuntimeError('STOP: pointer/manifest checksum mismatch: '+folder)
            pointers[folder]=snap
    keep=choose_retention(records,pointers)
    delete_snapshots=sorted(set(records)-keep)
    delete_files=sorted(p for p in entries if any(p.startswith(s+'/') for s in delete_snapshots))
    backup_hashes={}
    for snap in keep:
        for name,sha in records[snap]['sha256'].items():backup_hashes[snap+'/'+name]=sha
        remote=snap+'/manifest.json'
        backup_hashes[remote]=hashlib.sha256(raw_metadata[remote]).hexdigest()
    for folder in pointers:
        remote=folder+'/latest.json'
        backup_hashes[remote]=hashlib.sha256(raw_metadata[remote]).hexdigest()
    if not keep:raise RuntimeError('No recognized snapshots; nothing will be deleted')
    plan={'schema':1,'repository':REPO,'repo_type':KIND,'head':head,
          'keep_snapshots':sorted(keep),'delete_snapshots':delete_snapshots,
          'delete_files':delete_files,'unknown_keep':unknown,'backup_sha256':backup_hashes,
          'original_files':sorted(entries),'logical_delete_bytes':sum(entries[p] for p in delete_files),
          'retained_file_bytes':sum(entries[p] for p in backup_hashes),
          'notes':'Logical bytes are not physical storage/quota savings. Unknown paths retained. '
                  'Best uses latest saved best per lineage; no model deserialization or metric claims.'}
    (output/'cleanup_plan.json').write_bytes(json_bytes(plan))
    print('HEAD:',head,'KEEP:',len(keep),'DELETE:',len(delete_snapshots),'UNKNOWN kept:',len(unknown),flush=True)
    print('Logical bytes selected for deletion:',plan['logical_delete_bytes'],
          '(NOT guaranteed quota release)',flush=True)
    return plan


class ExtractedBackup:
    """Read-only view when Kaggle automatically extracts the uploaded ZIP."""
    def __init__(self,root):self.root=Path(root).resolve()
    def namelist(self):return [p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()]
    def testzip(self):return None  # Not a ZIP; each content blob is verified below.
    def open(self,name):
        valid_path(name);p=(self.root/name).resolve()
        if not p.is_relative_to(self.root):raise RuntimeError('Backup symlink escapes Input directory')
        return p.open('rb')
    def read(self,name):
        with self.open(name) as stream:return stream.read()


def verify_backup(archive,expected_sha):
    archive=Path(archive)
    checked=archive/'cleanup_plan.json' if archive.is_dir() else archive
    if not re.fullmatch('[0-9a-f]{64}',expected_sha) or digest(checked)!=expected_sha:
        raise RuntimeError('Backup SHA256 missing/mismatch; no deletion')
    reader=nullcontext(ExtractedBackup(archive)) if archive.is_dir() else zipfile.ZipFile(archive)
    with reader as z:
        if z.testzip():raise RuntimeError('Backup CRC failed')
        names=z.namelist()
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate ZIP entries')
        for name in names:valid_path(name)
        plan=json.loads(z.read('cleanup_plan.json'))
        if plan.get('schema')!=1 or plan.get('repository')!=REPO or plan.get('repo_type')!=KIND:
            raise RuntimeError('Wrong cleanup plan')
        for path,sha in plan['backup_sha256'].items():
            valid_path(path)
            h=hashlib.sha256()
            with z.open('blobs/'+sha) as file:
                for block in iter(lambda:file.read(4*1024**2),b''):h.update(block)
            if h.hexdigest()!=sha:raise RuntimeError('Backup artifact mismatch: '+path)
        # Delete only known snapshots; never latest pointers or unknown files.
        keep=set(plan['keep_snapshots']);deletes=set(plan['delete_snapshots'])
        if keep & deletes:raise RuntimeError('KEEP/DELETE overlap')
        expected=[]
        for snap in deletes:
            valid_path(snap)
            if snap.count('/')!=2 or snap.split('/')[0] not in FOLDERS or snap.split('/')[1]!='snapshots':
                raise RuntimeError('Unsafe deletion target')
            if snap in plan['unknown_keep']:raise RuntimeError('Attempt to delete unknown snapshot')
            expected.extend(p for p in plan['original_files'] if p.startswith(snap+'/'))
        if sorted(expected)!=sorted(plan['delete_files']):raise RuntimeError('Unexpected deletion paths')
        if set(plan['backup_sha256']) & set(plan['delete_files']):raise RuntimeError('Deleting protected artifact')
    return plan


def backup(api,download,output):
    output=Path(output)
    plan=audit(api,download,output)
    hashes=plan['backup_sha256']
    # Conservative: complete retained files + largest download cache + reserve.
    minimum=2*plan['retained_file_bytes']+1024**3
    free=shutil.disk_usage(output).free
    if free<minimum:raise RuntimeError(f'Backup needs about {minimum/1024**3:.2f} GB free, only {free/1024**3:.2f} GB available; no deletion')
    archive=output/('rod_hf_cleanup_backup_'+plan['head'][:12]+'.zip')
    if archive.exists():raise RuntimeError('Backup already exists; use a new output directory')
    written=set()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr('cleanup_plan.json',json_bytes(plan))
        z.writestr('RESTORE.txt','Read cleanup_plan.json; restore each remote path from blobs/<SHA256>. No torch.load needed.\n')
        for index,(remote,sha) in enumerate(hashes.items(),1):
            if sha in written:continue
            print(f'BACKUP {index}/{len(hashes)}: {remote}',flush=True)
            with tempfile.TemporaryDirectory(dir=output,prefix='download_') as cache:
                file=download(repo_id=REPO,repo_type=KIND,filename=remote,revision=plan['head'],cache_dir=cache)
                if digest(file)!=sha:raise RuntimeError('Remote SHA256 mismatch: '+remote)
                z.write(file,'blobs/'+sha)
            written.add(sha)
    sha=digest(archive)
    verify_backup(archive,sha)
    (output/'backup_sha256.txt').write_text(sha+'  '+archive.name+'\n',encoding='utf-8')
    print('BACKUP VERIFIED:',archive,'\nSHA256:',sha,flush=True)
    print('PLAN_SHA256 (if Kaggle extracts ZIP):',digest(output/'cleanup_plan.json'),flush=True)
    print('Download this ZIP, then attach it as Kaggle Input before APPLY. Nothing deleted.',flush=True)
    return archive,sha


def apply(api,archive,expected_sha,output,delete_operation,download,stopped=False,squash=False,confirm_squash=''):
    if not stopped:raise RuntimeError('Stop all training/upload sessions before APPLY')
    if squash and confirm_squash!=REPO:
        raise RuntimeError('Squash removes commit history irreversibly; confirm exact repository name')
    plan=verify_backup(archive,expected_sha)
    head=api.repo_info(repo_id=REPO,repo_type=KIND).sha
    if head!=plan['head']:raise RuntimeError('Repository changed since backup. No deletion; run BACKUP again')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt={'repository':REPO,'status':'IN_PROGRESS','verified_backup_digest':expected_sha,
             'backup_form':'extracted' if Path(archive).is_dir() else 'zip',
             'before_head':head,'deleted_files':[],'history_squashed':False}
    log=output/'cleanup_receipt.json'
    try:
        files=plan['delete_files']
        for start in range(0,len(files),100):
            batch=files[start:start+100]
            result=api.create_commit(repo_id=REPO,repo_type=KIND,revision='main',parent_commit=head,
                operations=[delete_operation(path_in_repo=p) for p in batch],
                commit_message='ROD: prune backed-up redundant snapshots')
            head=result.oid
            receipt['deleted_files'].extend(batch)
            receipt['after_head']=head
            log.write_bytes(json_bytes(receipt))
            print('Deleted files:',len(receipt['deleted_files']),'/',len(files),flush=True)
        actual=set(api.list_repo_files(repo_id=REPO,repo_type=KIND,revision=head))
        expected=set(plan['original_files'])-set(files)
        if actual!=expected:raise RuntimeError('Post-delete file inventory mismatch; history untouched')
        # Check exact LFS object identities and pointers at the new HEAD by SHA.
        for index,(remote,sha) in enumerate(plan['backup_sha256'].items(),1):
            print(f'POST-VERIFY {index}/{len(plan["backup_sha256"])}: {remote}',flush=True)
            with tempfile.TemporaryDirectory(dir=output,prefix='verify_') as cache:
                path=download(repo_id=REPO,repo_type=KIND,filename=remote,revision=head,cache_dir=cache)
                if digest(path)!=sha:raise RuntimeError('Retained artifact SHA256 mismatch: '+remote)
        if squash:
            if api.repo_info(repo_id=REPO,repo_type=KIND).sha!=head:
                raise RuntimeError('Concurrent write detected; history NOT squashed')
            api.super_squash_history(repo_id=REPO,repo_type=KIND,branch='main',
                                     commit_message='ROD: retain verified current checkpoints after backup')
            receipt['history_squashed']=True
            receipt['after_head']=api.repo_info(repo_id=REPO,repo_type=KIND).sha
        receipt['status']='PASS'
        receipt['quota_release']='NOT VERIFIED; check HF account storage after server processing'
    except Exception:
        receipt['status']='FAIL_OR_PARTIAL; inspect deleted_files before retry'
        raise
    finally:log.write_bytes(json_bytes(receipt))
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt
