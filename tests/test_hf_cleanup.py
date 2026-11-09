import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cleanup_hf_snapshots import REPO,apply,audit,backup,choose_retention,digest,verify_backup


class FakeHF:
    def __init__(self,root):
        self.root=root;root.mkdir();self.sha='a'*40;self.commits=[];self.squashed=False
    def repo_info(self,**kw):return SimpleNamespace(sha=self.sha)
    def list_repo_tree(self,**kw):
        return [SimpleNamespace(path=p.relative_to(self.root).as_posix(),size=p.stat().st_size)
                for p in self.root.rglob('*') if p.is_file()]
    def list_repo_files(self,**kw):return [x.path for x in self.list_repo_tree()]
    def download(self,filename,**kw):return self.root/filename
    def create_commit(self,operations,parent_commit,**kw):
        assert parent_commit==self.sha
        for op in operations:(self.root/op.path_in_repo).unlink()
        self.commits.append(operations);self.sha='b'*40
        return SimpleNamespace(oid=self.sha)
    def super_squash_history(self,**kw):self.squashed=True;self.sha='c'*40


def snapshot(api,name,epoch,offset=0,best=True,config='c'*64):
    snap='YOLO 11s/snapshots/'+name;directory=api.root/snap;directory.mkdir(parents=True)
    files={'recovery_full.pt':f'checkpoint {epoch} {offset}'.encode()}
    if best:files['best.pt']=b'saved best'
    for k,v in files.items():(directory/k).write_bytes(v)
    m={'schema':2,'folder':'YOLO 11s','model':'yolo11s.pt','snapshot':snap,
       'completed_epoch':epoch,'next_sample':offset,'config_sha256':config,
       'dataset_sha256':'d'*64,'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}}
    (directory/'manifest.json').write_text(json.dumps(m))
    return snap,m


def setup_repo(tmp_path):
    api=FakeHF(tmp_path/'remote')
    snapshot(api,'epoch10',10)
    snapshot(api,'epoch11',11)
    current,_=snapshot(api,'epoch12',12)
    pointer={'snapshot':current,'manifest_sha256':digest(api.root/current/'manifest.json')}
    (api.root/'YOLO 11s/latest.json').write_text(json.dumps(pointer))
    (api.root/'README.md').write_text('keep me')
    return api


def test_retains_latest_best_milestones_and_each_config():
    records={}
    for name,epoch,offset,has_best,config in [('ten',10,0,True,'a'),('eleven',11,0,True,'a'),
          ('twelve',12,0,False,'a'),('mid',12,99,False,'a'),('other',9,0,True,'b')]:
        records[name]={'folder':'YOLO 11s','model':'yolo11s.pt','config_sha256':config,
          'dataset_sha256':'d','completed_epoch':epoch,'next_sample':offset,
          'sha256':{'best.pt':'sha'} if has_best else {'recovery_full.pt':'sha'}}
    keep=choose_retention(records,{'YOLO 11s':'twelve'})
    assert keep=={'ten','eleven','twelve','mid','other'}


def test_backup_is_read_only_and_deduplicates_bytes(tmp_path):
    api=setup_repo(tmp_path)
    file,sha=backup(api,api.download,tmp_path/'output')
    p=verify_backup(file,sha)
    assert p['delete_snapshots']==['YOLO 11s/snapshots/epoch11']
    assert not api.commits and not api.squashed
    assert len(p['keep_snapshots'])==2


def test_unknown_snapshots_are_not_deleted(tmp_path):
    api=setup_repo(tmp_path)
    (api.root/'YOLO 11s/snapshots/epoch11/unexpected.pt').write_bytes(b'unique')
    plan=audit(api,api.download,tmp_path/'out')
    assert not plan['delete_files']
    assert 'YOLO 11s/snapshots/epoch11' in plan['unknown_keep']


def test_changed_head_rejected_before_deletion(tmp_path):
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    api.sha='new'
    with pytest.raises(RuntimeError,match='Repository changed'):
        apply(api,file,sha,tmp_path/'apply',SimpleNamespace,api.download,stopped=True)
    assert not api.commits


def test_bad_backup_hash_rejected(tmp_path):
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    with pytest.raises(RuntimeError,match='Backup SHA256'):
        apply(api,file,'0'*64,tmp_path/'apply',SimpleNamespace,api.download,stopped=True)
    assert not api.commits


def test_delete_only_planned_files_and_verify_remaining(tmp_path):
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    pointer_before=(api.root/'YOLO 11s/latest.json').read_bytes()
    receipt=apply(api,file,sha,tmp_path/'apply',SimpleNamespace,api.download,stopped=True)
    assert receipt['status']=='PASS' and len(api.commits)==1
    assert not api.squashed
    assert (api.root/'YOLO 11s/latest.json').read_bytes()==pointer_before
    assert (api.root/'README.md').exists()
    assert not (api.root/'YOLO 11s/snapshots/epoch11/best.pt').exists()
    assert (api.root/'YOLO 11s/snapshots/epoch10/best.pt').exists()


def test_squash_requires_explicit_confirmation(tmp_path):
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    with pytest.raises(RuntimeError,match='irreversibly'):
        apply(api,file,sha,tmp_path/'apply',SimpleNamespace,api.download,stopped=True,squash=True)
    assert not api.commits and not api.squashed


def test_squash_only_after_backup_delete_and_verification(tmp_path):
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    result=apply(api,file,sha,tmp_path/'apply',SimpleNamespace,api.download,
                 stopped=True,squash=True,confirm_squash=REPO)
    assert result['status']=='PASS' and api.squashed


def test_bad_latest_pointer_stops_whole_cleanup(tmp_path):
    api=setup_repo(tmp_path)
    (api.root/'YOLO 11s/latest.json').write_text(json.dumps({'snapshot':'missing','manifest_sha256':'bad'}))
    with pytest.raises(RuntimeError,match='latest points'):
        audit(api,api.download,tmp_path/'out')
    assert not api.commits


def test_training_must_be_stopped(tmp_path):
    api=setup_repo(tmp_path)
    with pytest.raises(RuntimeError,match='Stop all training'):
        apply(api,'missing.zip','0'*64,tmp_path/'apply',SimpleNamespace,api.download)


def test_kaggle_extracted_backup_is_verified_by_plan_and_blob_hashes(tmp_path):
    import zipfile
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    extracted=tmp_path/'input';extracted.mkdir()
    # Fixture is the archive generated by backup(), not arbitrary external ZIPs.
    with zipfile.ZipFile(file) as z:z.extractall(extracted)
    plan_hash=digest(extracted/'cleanup_plan.json')
    result=apply(api,extracted,plan_hash,tmp_path/'apply',SimpleNamespace,api.download,stopped=True)
    assert result['status']=='PASS' and result['backup_form']=='extracted'


def test_corrupt_extracted_backup_is_rejected_before_delete(tmp_path):
    import zipfile
    api=setup_repo(tmp_path);file,sha=backup(api,api.download,tmp_path/'out')
    extracted=tmp_path/'input';extracted.mkdir()
    with zipfile.ZipFile(file) as z:z.extractall(extracted)
    next((extracted/'blobs').iterdir()).write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='Backup artifact mismatch'):
        apply(api,extracted,digest(extracted/'cleanup_plan.json'),tmp_path/'apply',SimpleNamespace,api.download,stopped=True)
    assert not api.commits
