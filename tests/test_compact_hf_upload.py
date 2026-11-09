"""Offline tests only: artifact selection, retention CAS and backup cadence."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_kaggle_hf_autoresume import FakeHF
from tests.test_simple_30epochs import cell


class CompactHF(FakeHF):
    def __init__(self,root):
        super().__init__(root);root.mkdir();self.version=0;self.deletions=[];self.reject_delete=False
    def repo_info(self,**kwargs):return SimpleNamespace(sha=str(self.version))
    def upload_folder(self,**kwargs):super().upload_folder(**kwargs);self.version+=1
    def upload_file(self,**kwargs):super().upload_file(**kwargs);self.version+=1
    def create_commit(self,operations,parent_commit,**kwargs):
        assert kwargs['repo_id']=='13ffa/deep'
        assert parent_commit==str(self.version)
        if self.reject_delete:raise RuntimeError('simulated concurrent commit')
        for operation in operations:
            assert '/snapshots/' in operation.path_in_repo
            assert not operation.path_in_repo.endswith('/latest.json')
            self.deletions.append(operation.path_in_repo)
            (self.remote/operation.path_in_repo).unlink()
        self.version+=1
        return SimpleNamespace(oid=str(self.version))


@pytest.fixture
def compact_store(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(
        CommitOperationDelete=lambda **kw:SimpleNamespace(**kw)))
    env={};exec(cell('E4',4),env)
    class LocalStore(env['HFStore']):
        def _download(self,remote):return self.api.remote/remote
        def _retry(self,action):return action()
    api=CompactHF(tmp_path/'remote')
    store=LocalStore(api,'new-token','dataset','YOLO 11s','yolo11s.pt','c'*64,'d'*64,
                     tmp_path/'work',compact=True)
    sources=tmp_path/'sources';sources.mkdir()
    files={}
    for name in ['recovery_full.pt','last.pt','best.pt','results.csv','args.yaml','config.json','environment.json']:
        path=sources/name
        path.write_bytes(b'full optimizer/scaler/ema' if name.endswith('.pt') else name.encode())
        files[name]=path
    return store,api,files


def test_duplicate_checkpoint_removed_and_best_alias_roundtrips(compact_store):
    store,api,files=compact_store
    original=files['recovery_full.pt'].read_bytes()
    m=store.save(files,13)
    assert 'last.pt' not in m['sha256'] and 'best.pt' not in m['sha256']
    assert m['artifact_aliases']=={'best.pt':'recovery_full.pt'}
    restored,local=store.restore()
    assert local['best.pt']==local['recovery_full.pt']
    assert local['recovery_full.pt'].read_bytes()==original
    assert files['last.pt'].read_bytes()==original  # Local full backups remain untouched.
    assert 'storage_policy' in restored


def test_distinct_best_is_preserved(compact_store):
    store,api,files=compact_store
    files['best.pt'].write_bytes(b'different best from previous epoch')
    m=store.save(files,13)
    assert 'best.pt' in m['sha256'] and m['artifact_aliases']=={}
    assert store.restore()[1]['best.pt'].read_bytes()==b'different best from previous epoch'


def test_only_latest_and_complete_milestones_remain(compact_store):
    store,api,files=compact_store
    for epoch in [10,11,12,13,19,20,21,29,30]:
        store.save(files,epoch,{'next_sample':0})
    manifests=list(api.remote.rglob('manifest.json'))
    assert sorted(json.loads(p.read_text())['completed_epoch'] for p in manifests)==[10,20,30]
    assert store.restore()[0]['completed_epoch']==30
    assert api.deletions and not (store.work/'retention_warning.json').exists()


def test_partial_epoch_does_not_replace_complete_milestone(compact_store):
    store,api,files=compact_store
    store.save(files,20,{'next_sample':0})
    store.save(files,20,{'next_sample':3600})
    store.save(files,20,{'next_sample':7200})
    assert len(list(api.remote.rglob('manifest.json')))==2
    assert store.restore()[0]['next_sample']==7200


def test_unmanaged_snapshots_and_other_models_not_deleted(compact_store):
    store,api,files=compact_store
    store.compact=False;old=store.save(files,9);store.compact=True
    other=api.remote/'Faster R-CNN/snapshots/unique/checkpoint.pt'
    other.parent.mkdir(parents=True);other.write_bytes(b'unique other model')
    store.save(files,11);store.save(files,12)
    assert (api.remote/old['snapshot']/'manifest.json').exists()
    assert other.read_bytes()==b'unique other model'


def test_cleanup_conflict_keeps_new_checkpoint_and_old_copy(compact_store):
    store,api,files=compact_store
    store.save(files,11)
    api.reject_delete=True
    store.save(files,12)
    assert store.restore()[0]['completed_epoch']==12
    assert len(list(api.remote.rglob('manifest.json')))==2
    assert json.loads((store.work/'retention_warning.json').read_text())['retention']=='SKIPPED'


def test_pointer_upload_failure_does_not_prune_previous(compact_store):
    store,api,files=compact_store
    store.save(files,11);api.fail_pointer=True
    with pytest.raises(OSError):store.save(files,12)
    assert store.restore()[0]['completed_epoch']==11
    assert not api.deletions


def test_missing_required_resume_payload_fails_before_upload(compact_store):
    store,api,files=compact_store
    del files['recovery_full.pt']
    with pytest.raises(RuntimeError,match='Compact upload'):
        store.save(files,13)
    assert api.version==0


def test_unknown_file_inside_managed_snapshot_prevents_deletion(compact_store):
    store,api,files=compact_store
    m=store.save(files,11)
    unknown=api.remote/m['snapshot']/'unique.bin';unknown.write_bytes(b'unclassified')
    store.save(files,12)
    assert unknown.exists() and (unknown.parent/'manifest.json').exists()


def test_read_only_source_cannot_be_pruned(compact_store):
    store,api,files=compact_store
    m=store.save(files,12)
    store.repo_id='iraprp/deeplearning'
    with pytest.raises(RuntimeError,match='source repository'):store._prune_compact(m)


def test_e6_remote_cadence_keeps_local_recovery_and_forces_deadline(tmp_path):
    code=cell('E6',5)
    tree=ast.parse(code)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='persist')
    clock=[0];calls=[]
    state=SimpleNamespace(state_dict=lambda:{'state':'preserved'})
    fake_torch=SimpleNamespace(save=lambda payload,p:Path(p).write_text(json.dumps(payload)),
        get_rng_state=lambda:[1],cuda=SimpleNamespace(get_rng_state_all=lambda:[[2]]))
    def save(files,done,extra):calls.append((done,extra['next_sample']));return dict(extra,completed_epoch=done)
    from src.kaggle_hf_shared import atomic_json
    metric=tmp_path/'metrics.csv';metric.write_text('epoch,map\n')
    env={'net':state,'optimizer':state,'scheduler':state,'scaler':state,'torch':fake_torch,
      'random':SimpleNamespace(getstate=lambda:1),'np':SimpleNamespace(random=SimpleNamespace(get_state=lambda:2)),
      'best_map':0.5,'metrics_rows':[],'config_hash':'c','dataset_hash':'d','E6_MODEL':'fasterrcnn_resnet50_fpn_v2',
      'E6_WORK':tmp_path,'E6_DEADLINE':10000,'E6_HF_INTERVAL_SECONDS':3600,'last_hf_sync':0,
      'time':SimpleNamespace(monotonic=lambda:clock[0]),'os':__import__('os'),
      'store':SimpleNamespace(save=save),'write_metrics':lambda:metric,'atomic_json':atomic_json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'e6-persist','exec'),env)
    persist=env['persist']
    persist(13,3600,13,1.0,1)
    assert (tmp_path/'checkpoint.pt').exists() and calls==[]
    clock[0]=4000;persist(13,7200,13,1.0,2)
    clock[0]=4100;persist(14,0,14,0.0,0)
    clock[0]=10001;persist(14,300,14,1.0,1)
    assert calls==[(13,7200),(14,0),(14,300)]
