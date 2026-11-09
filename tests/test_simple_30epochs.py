"""Regression tests for minimally extending the original successful notebooks."""
import json
from pathlib import Path

import pytest
import yaml

from tests.test_kaggle_hf_autoresume import make_store

ROOT = Path(__file__).resolve().parents[1]


def cell(exp, index):
    book = json.loads((ROOT/f'notebooks/Kaggle_{exp}_30epochs_AutoResume.ipynb').read_text(encoding='utf-8'))
    return ''.join(book['cells'][index]['source'])


def test_legacy_complete_at_ten_is_compatible_but_new_complete_still_requires_30():
    env = {};exec(cell('E4',4),env)
    validate = env['validate_progress']
    m = {'schema':2,'folder':'YOLO 11s','model':'yolo11s.pt','config_sha256':'c',
         'dataset_sha256':'d','completed_epoch':10,'is_complete':True,'sha256':{'ckpt':'sha'}}
    assert validate(m,'YOLO 11s','yolo11s.pt','c','d')==10
    assert validate(dict(m,is_complete=False,target_epochs=30),'YOLO 11s','yolo11s.pt','c','d')==10
    for edits in ({'completed_epoch':20},{'target_epochs':30},{'completed_epoch':31}):
        with pytest.raises(RuntimeError):
            validate(dict(m,**edits),'YOLO 11s','yolo11s.pt','c','d')
    assert validate(dict(m,completed_epoch=30,target_epochs=30),'YOLO 11s','yolo11s.pt','c','d')==30


def test_http_error_detail_retained_without_secret_leak(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    from src.kaggle_hf_shared import HFStore
    store,_=make_store(tmp_path)
    store._retry=HFStore._retry.__get__(store,HFStore)
    store.token='hf_test_private_secret'
    class Rejected(Exception):
        response=SimpleNamespace(status_code=400)
        server_message='Storage limit. token=hf_test_private_secret https://example.test/upload?secret=abc'
    monkeypatch.setattr('src.kaggle_hf_shared.time.sleep',lambda _:None)
    def action():raise Rejected('request rejected')
    with pytest.raises(RuntimeError,match='HTTP 400; Storage limit') as error:
        store._retry(action)
    result=capsys.readouterr().out+str(error.value)
    assert 'hf_test_private_secret' not in result
    assert 'secret=abc' not in result
    assert '[REDACTED]' in result
    log=(store.work/'hf_upload_errors.jsonl').read_text(encoding='utf-8')
    records=[json.loads(line) for line in log.splitlines()]
    assert len(records)==3 and all(r['http_status']==400 for r in records)
    assert 'hf_test_private_secret' not in log and 'secret=abc' not in log


def test_plain_text_hf_response_and_bearer_are_logged(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    from src.kaggle_hf_shared import HFStore
    store,_=make_store(tmp_path)
    store._retry=HFStore._retry.__get__(store,HFStore)
    class Rejected(Exception):
        response=SimpleNamespace(status_code=403,text='Storage policy: Bearer privateCredential https://host/path?token=private')
    monkeypatch.setattr('src.kaggle_hf_shared.time.sleep',lambda _:None)
    def action():raise Rejected('rejected')
    with pytest.raises(RuntimeError,match='Storage policy'):
        store._retry(action)
    log=(store.work/'hf_upload_errors.jsonl').read_text(encoding='utf-8')
    assert 'privateCredential' not in log and 'token=private' not in log
    assert 'Storage policy' in log


@pytest.mark.parametrize('experiment',['E3','E4','E6'])
def test_new_account_is_destination_original_is_read_only_source(experiment):
    code=cell(experiment,4)
    assert 'HF_REPO_ID = "13ffa/deep"' in code
    assert 'HF_SECRET_NAME = "HF_TOKEN"' in code
    assert 'iraprp/DEEP' not in code
    assert 'repo_id=self.repo_id' in code
    helper=code[code.index('def choose_resume'):code.index('def yolo_resume_directory')]
    assert 'current' in helper and 'previous' in helper
    assert "repo_id='iraprp/deeplearning'" in helper
    assert "get_secret('HF_TOKEN2')" in helper
    assert '.save(' not in helper


def test_newer_extension_checkpoint_is_not_discarded(tmp_path):
    old, api = make_store(tmp_path)
    env = {};exec(cell('E4',4),env)
    Base = env['HFStore']
    class LocalStore(Base):
        def _download(self, name):return api.remote/name
        def _retry(self, action):return action()
    env['HFStore']=LocalStore
    p=tmp_path/'resume.pt';p.write_bytes(b'epoch10')
    old.save({'recovery_full.pt':p},10)
    current=LocalStore(api,'fake','dataset','YOLO 11s','yolo11s.pt','c'*64,'d'*64,tmp_path/'new')
    previous=LocalStore(api,'fake','dataset','E4_30epochs','yolo11s.pt','e'*64,'d'*64,tmp_path/'extension')
    p.write_bytes(b'epoch17')
    previous.save({'recovery_full.pt':p},17)
    chosen=env['choose_resume'](current,'E4_30epochs','e'*64)
    assert chosen[0]['completed_epoch']==17
    assert chosen[1]['recovery_full.pt'].read_bytes()==b'epoch17'
    p.write_bytes(b'epoch18');current.save({'recovery_full.pt':p},18)
    assert env['choose_resume'](current,'E4_30epochs','e'*64)[0]['completed_epoch']==18


def test_cross_account_bootstrap_reads_old_token_and_writes_new_only(tmp_path,monkeypatch):
    import sys,shutil
    from types import SimpleNamespace
    env={};exec(cell('E4',4),env)
    class RoutedHF:
        roots={'iraprp/deeplearning':tmp_path/'old', '13ffa/deep':tmp_path/'new'}
        def __init__(self):
            self.reads=[];self.writes=[]
            for p in self.roots.values():p.mkdir()
        def list_repo_files(self,repo_id,token,**kwargs):
            self.reads.append((repo_id,token))
            return [p.relative_to(self.roots[repo_id]).as_posix()
                    for p in self.roots[repo_id].rglob('*') if p.is_file()]
        def upload_folder(self,folder_path,path_in_repo,repo_id,token,**kwargs):
            self.writes.append((repo_id,token))
            shutil.copytree(folder_path,self.roots[repo_id]/path_in_repo)
        def upload_file(self,path_or_fileobj,path_in_repo,repo_id,token,**kwargs):
            self.writes.append((repo_id,token))
            shutil.copy2(path_or_fileobj,self.roots[repo_id]/path_in_repo)
    class LocalStore(env['HFStore']):
        def _download(self,name):return self.api.roots[self.repo_id]/name
        def _retry(self,action):return action()
    api=RoutedHF();env['HFStore']=LocalStore
    source=LocalStore(api,'old-read','dataset','YOLO 11s','yolo11s.pt','c'*64,'d'*64,
                      tmp_path/'source',repo_id='iraprp/deeplearning')
    checkpoint=tmp_path/'recovery.pt';checkpoint.write_bytes(b'epoch12')
    source.save({'recovery_full.pt':checkpoint},12)
    before={p.relative_to(api.roots['iraprp/deeplearning']).as_posix():p.read_bytes()
            for p in api.roots['iraprp/deeplearning'].rglob('*') if p.is_file()}
    api.writes=[]
    def get_secret(name):
        assert name=='HF_TOKEN2'
        return 'old-read'
    monkeypatch.setitem(sys.modules,'kaggle_secrets',SimpleNamespace(
        UserSecretsClient=lambda:SimpleNamespace(get_secret=get_secret)))
    destination=LocalStore(api,'new-write','model','YOLO 11s','yolo11s.pt','c'*64,'d'*64,tmp_path/'destination')
    chosen=env['choose_resume'](destination,'E4_30epochs','e'*64)
    assert chosen[0]['completed_epoch']==12 and api.writes==[]
    assert ('iraprp/deeplearning','old-read') in api.reads
    checkpoint.write_bytes(b'epoch13')
    destination.save({'recovery_full.pt':checkpoint},13)
    assert all(pair==('13ffa/deep','new-write') for pair in api.writes)
    after={p.relative_to(api.roots['iraprp/deeplearning']).as_posix():p.read_bytes()
           for p in api.roots['iraprp/deeplearning'].rglob('*') if p.is_file()}
    assert before==after
    # Once migration succeeded, a missing source secret must not block resume.
    monkeypatch.delitem(sys.modules,'kaggle_secrets')
    assert env['choose_resume'](destination,'E4_30epochs','e'*64)[0]['completed_epoch']==13


def test_e4_frozen_config_identical_to_original():
    import ast
    def frozen(text,name):
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        return yaml.safe_load(ast.literal_eval(node.value))
    original=json.loads((ROOT/'notebooks/Kaggle_E4_YOLO11s_HF_AutoResume.ipynb').read_text(encoding='utf-8'))
    assert frozen(''.join(original['cells'][5]['source']),'FROZEN_E4_YAML')==frozen(cell('E4',5),'FROZEN_E4_YAML')
    assert ''.join(original['cells'][3]['source'])==cell('E4',3)


def test_e6_preserves_original_config_fingerprint_and_training_loop():
    original=json.loads((ROOT/'notebooks/Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb').read_text(encoding='utf-8'))
    old=''.join(original['cells'][5]['source'])
    new=cell('E6',5)
    start=old.index('        for images, targets in loader:')
    end=old.index('            if running_count % E6_BACKUP_BATCHES',start)
    assert old[start:end] in new
    # Progress logs sit between the unchanged optimizer step and recovery check.
    recovery_start=old.index('            if running_count % E6_BACKUP_BATCHES',start)
    recovery_end=old.index('        if stop:',recovery_start)
    assert old[recovery_start:recovery_end] in new
    assert 'now - train_last_log >= 60' in new
    assert 'now - valid_last_log >= 60' in new
    assert 'E6 VALID computing mAP...' in new
    assert 'saving checkpoint...' in new
    assert '"epochs": 10,' in new  # source identity retained, budget is external metadata
    assert 'E6_EPOCHS, E6_BATCH, E6_BACKUP_BATCHES = 30, 2, 1800' in new


@pytest.mark.parametrize('exp', ['E3','E4'])
def test_yolo_guard_still_checks_30_schedule_and_correct_model(exp):
    code=cell(exp,5)
    assert f'{exp}_BUDGET = 30' in code
    assert 'frozen["epochs"] != 30' in code
    assert f'{exp}_MODEL = "yolo11'+('n.pt"' if exp=='E3' else 's.pt"') in code
    assert 'No verified recovery; refusing restart at epoch 1' in code
