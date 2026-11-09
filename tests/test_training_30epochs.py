"""CPU contract tests; never run ROD training or TEST inference."""
import ast
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.kaggle_hf_shared import HFStore, validate_progress
from src.rod_train_30epochs import (check_e6, check_yolo, checked_member, extract_bundle,
                                    next_stage, validate_metrics, discover_source, restore_legacy_hf)
from tests.test_kaggle_hf_autoresume import make_store

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('done,target', [(10,20), (11,20), (19,20), (20,30), (29,30), (30,None)])
def test_next_stage(done, target):
    assert next_stage(done) == target


@pytest.mark.parametrize('value', [0,9,31,True,10.5])
def test_invalid_progress(value):
    with pytest.raises(ValueError):
        next_stage(value)


@pytest.mark.parametrize('name', ['../x','/x','a\\x','C:/x','a/../../x'])
def test_zip_traversal(name):
    with pytest.raises(ValueError):
        checked_member(name)


def test_bundle_integrity_wrong_model_and_tamper(tmp_path):
    from src.kaggle_hf_shared import sha256_file
    path = tmp_path / 'bundle.zip'
    manifest = {'experiment':'E3', 'completed_epoch':10,
                'sha256':{'recovery_full.pt': hashlib.sha256(b'weights').hexdigest()}}
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('source_manifest.json', json.dumps(manifest))
        z.writestr('recovery_full.pt', b'weights')
    extract_bundle(path, tmp_path/'source', 'E3', sha256_file)
    with pytest.raises(RuntimeError, match='Wrong source'):
        extract_bundle(path, tmp_path/'other', 'E4', sha256_file)
    (tmp_path/'source/recovery_full.pt').write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='overwrite'):
        extract_bundle(path, tmp_path/'source', 'E3', sha256_file)


def test_metrics_gaps_rejected():
    assert validate_metrics([{'epoch': i} for i in range(1,11)], 10)
    for rows in ([{'epoch':i} for i in range(1,10)], [{'epoch':1}]*10):
        with pytest.raises(RuntimeError):
            validate_metrics(rows, 10)


@pytest.mark.parametrize('extracted', [False, True])
def test_discover_zip_or_kaggle_extracted_input(tmp_path, extracted):
    from src.kaggle_hf_shared import sha256_file
    inputs = tmp_path / 'input'; inputs.mkdir()
    manifest = {'experiment': 'E4', 'completed_epoch': 10,
                'sha256': {'recovery_full.pt': hashlib.sha256(b'real').hexdigest()}}
    if extracted:
        (inputs / 'source_manifest.json').write_text(json.dumps(manifest))
        (inputs / 'recovery_full.pt').write_bytes(b'real')
    else:
        with zipfile.ZipFile(inputs / 'e4_resume_epoch010.zip', 'w') as z:
            z.writestr('source_manifest.json', json.dumps(manifest))
            z.writestr('recovery_full.pt', b'real')
    assert discover_source(inputs, tmp_path / 'output', 'E4', sha256_file) == manifest
    assert (tmp_path / 'output/recovery_full.pt').read_bytes() == b'real'
    with pytest.raises(RuntimeError, match='found 0'):
        discover_source(inputs, tmp_path / 'other', 'E3', sha256_file)
    if extracted:
        (inputs / 'recovery_full.pt').write_bytes(b'corrupt')
        with pytest.raises(RuntimeError, match='SHA256'):
            discover_source(inputs, tmp_path / 'bad', 'E4', sha256_file)


def test_duplicate_sources_rejected_with_paths(tmp_path):
    from src.kaggle_hf_shared import sha256_file
    for folder in ('one', 'two'):
        p = tmp_path / folder; p.mkdir()
        (p / 'source_manifest.json').write_text(json.dumps({'experiment': 'E4'}))
    with pytest.raises(RuntimeError, match='found 2'):
        discover_source(tmp_path, tmp_path/'output', 'E4', sha256_file)


@pytest.mark.parametrize('experiment,folder,recovery,best', [
    ('E4','YOLO 11s','recovery_full.pt','best.pt'),
    ('E6','Faster R-CNN','checkpoint.pt','best_weights.pth')])
def test_download_legacy_hf_without_bundle_and_without_modifying_old_folder(tmp_path, monkeypatch, experiment, folder, recovery, best):
    import src.rod_train_30epochs as runtime
    from src.kaggle_hf_shared import sha256_file
    store, api = make_store(tmp_path, folder=folder)
    checkpoint = tmp_path/'old.pt'; checkpoint.write_bytes(b'audited epoch 10')
    files = {name:checkpoint for name in (recovery,best,'results.csv','args.yaml')}
    store.save(files,10,{'next_sample':0})
    before = {n:sha256_file(api.remote/n) for n in api.list_repo_files()}
    class ReadStore(HFStore):
        def _download(self, remote):
            return api.remote/remote
    monkeypatch.setattr(runtime,'HFStore',ReadStore,raising=False)
    from src.kaggle_hf_shared import atomic_json, HF_REPO_ID
    monkeypatch.setattr(runtime,'sha256_file',sha256_file,raising=False)
    monkeypatch.setattr(runtime,'atomic_json',atomic_json,raising=False)
    monkeypatch.setattr(runtime,'HF_REPO_ID',HF_REPO_ID,raising=False)
    monkeypatch.setattr(runtime,'LEGACY_CONFIG_SHA256',store.config_hash,raising=False)
    work = tmp_path/'new';work.mkdir()
    origin = {'dataset_yaml_sha256':store.dataset_hash,'sha256':{recovery:sha256_file(checkpoint)}}
    manifest, loaded = restore_legacy_hf(api,'fake','dataset',experiment,origin,work)
    assert manifest['completed_epoch']==10 and loaded[recovery].read_bytes()==b'audited epoch 10'
    assert before == {n:sha256_file(api.remote/n) for n in api.list_repo_files()}
    origin['sha256'][recovery]='wrong'
    with pytest.raises(RuntimeError,match='differs from audited'):
        restore_legacy_hf(api,'fake','dataset',experiment,origin,work)


def test_original_e4_config_hash_matches_observed_hf_hash():
    from src.kaggle_hf_shared import fingerprint
    config = json.loads((ROOT/'results/E4_YOLO11s/logs/config.txt').read_text(encoding='utf-8'))
    assert fingerprint(config)=='684c8c281633c537094601c84bf3cdea5b674126230a69b74004380d6276aa70'


def test_hf_thirty_epoch_progress_and_isolation(tmp_path):
    old, api = make_store(tmp_path)
    store = HFStore(api,'fake','dataset','E3_30epochs','yolo11n.pt','c'*64,'d'*64,tmp_path/'extension',max_epoch=30)
    store._download = old._download
    store._retry = old._retry
    p = tmp_path/'resume.pt'
    for ep in (10,11,20,21,30):
        p.write_bytes(str(ep).encode())
        manifest = store.save({'recovery_full.pt':p},ep)
        assert manifest['is_complete'] is (ep == 30)
        assert store.restore()[0]['completed_epoch'] == ep
    assert old.restore() is None
    with pytest.raises(RuntimeError):
        store.save({'recovery_full.pt':p},31)


def test_yolo_effective_args_and_optimizer_guard():
    frozen = {'epochs':30,'batch':8,'imgsz':640,'lr0':.01,'optimizer':'SGD'}
    payload = {'epoch':9,'version':'8.4.165','optimizer':{'state':{0:{}},'param_groups':[{}]},
               'scaler':{},'train_args':dict(frozen,cos_lr=False),
               'ema':SimpleNamespace(nc=25,yaml={'scale':'n'},names=list(range(25)))}
    assert check_yolo(payload,'E3',list(range(25)),frozen,10) == 10
    for edit in ({'epoch':-1},{'optimizer':None},{'version':'wrong'},
                 {'train_args':dict(frozen,cos_lr=True)},
                 {'ema':SimpleNamespace(nc=25,yaml={'scale':'s'},names=list(range(25)))}):
        with pytest.raises(RuntimeError):
            check_yolo(dict(payload,**edit),'E3',list(range(25)),frozen,10)


def test_e6_real_optimizer_scheduler_roundtrip_cpu(tmp_path):
    import torch
    model = torch.nn.Linear(2,1)
    opt = torch.optim.SGD(model.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=3,gamma=.1)
    for _ in range(10):
        opt.zero_grad(); model(torch.ones(1,2)).sum().backward(); opt.step(); scheduler.step()
    path = tmp_path/'toy.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict()},path)
    restored = torch.load(path,weights_only=True)
    model2 = torch.nn.Linear(2,1); model2.load_state_dict(restored['model'])
    opt2 = torch.optim.SGD(model2.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
    scheduler2 = torch.optim.lr_scheduler.StepLR(opt2,3,.1)
    opt2.load_state_dict(restored['optimizer']); scheduler2.load_state_dict(restored['scheduler'])
    for epoch in range(11,31):
        for m,o,s in ((model,opt,scheduler),(model2,opt2,scheduler2)):
            o.zero_grad();m(torch.ones(1,2)).sum().backward();o.step();s.step()
        assert scheduler2.last_epoch == epoch
        assert scheduler2.get_last_lr() == pytest.approx([.005*.1**(epoch//3)])
        for a,b in zip(model.parameters(),model2.parameters()):
            assert torch.equal(a,b)


def test_e6_invalid_state_rejected():
    import torch
    saved = {'model':{'roi_heads.box_predictor.cls_score.weight':torch.zeros(26,1)},
        'optimizer':{'state':{0:{}},'param_groups':[{'lr':5e-6,'momentum':.9,'weight_decay':.0005}]},
        'scheduler':{'last_epoch':10,'step_size':3,'gamma':.1,'base_lrs':[.005],'_last_lr':[5e-6]},
        'scaler':{},'python_rng':None,'numpy_rng':None,'torch_rng':None,'cuda_rng':None,
        'metrics_rows':[{'epoch':i} for i in range(1,11)],'best_map':.5,
        'loss_sum':0.,'loss_count':0,'completed_epoch':10,'current_epoch':10,'next_sample':0,
        'config_hash':'c','dataset_hash':'d'}
    assert check_e6(saved,'c','d') == (10,0)
    for k,v in [('next_sample',1),('config_hash','bad'),('completed_epoch',30)]:
        with pytest.raises(RuntimeError):
            check_e6(dict(saved,**{k:v}),'c','d')
    bad = copy.deepcopy(saved);bad['scheduler']['last_epoch']=0
    with pytest.raises(RuntimeError):
        check_e6(bad,'c','d')


@pytest.mark.parametrize('experiment',['E3','E4','E6'])
def test_standalone_notebook_syntax_trainval_and_namespaces(experiment):
    nb = json.loads((ROOT/f'notebooks/Kaggle_{experiment}_30epochs_AutoResume.ipynb').read_text(encoding='utf-8'))
    combined = ''
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        assert c['outputs'] == [] and c['execution_count'] is None
        code = ''.join(c['source']); combined += code
        if not code.lstrip().startswith('%pip'):
            compile(code,experiment,'exec')
    assert "TEST_IMAGES = None" in combined
    assert 'E5' not in combined
    assert 'run_extension()' not in combined
    assert 'class Tee' not in combined and 'RelocatedTrainer' not in combined
    assert 'max_epoch=30' in combined
    if experiment == 'E6':
        assert 'StepLR(optimizer, step_size=3, gamma=0.1)' in combined
        assert 'E6_EPOCHS, E6_BATCH, E6_BACKUP_BATCHES = 30, 2, 1800' in combined
    else:
        assert 'model.train(resume=True, data=str(working_yaml), device=0)' in combined
        assert experiment + '_BUDGET = 30' in combined


def test_checkpoint_paths_relocated_before_model_init():
    tree = ast.parse((ROOT/'src/rod_train_30epochs.py').read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_yolo')
    cls = next(n for n in run.body if isinstance(n,ast.ClassDef))
    class FakeTrainer:
        def check_resume(self, overrides):
            self.resume = True
            self.args = SimpleNamespace(data='old',project='old',name='old')
    env = {'DetectionTrainer':FakeTrainer,'working_yaml':Path('/new/rod.yaml'),
           'work':Path('/new/output'),'run':Path('/new/output/train')}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'relocate','exec'),env)
    t=env['RelocatedTrainer']();t.check_resume({})
    assert Path(t.args.data) == Path('/new/rod.yaml')
    assert Path(t.args.save_dir) == Path('/new/output/train')
    assert t.args.split == 'val'


@pytest.mark.parametrize('experiment', ['E3', 'E4', 'E6'])
def test_notebook_keeps_native_logging_and_no_mid_stage_stop(experiment):
    book = json.loads((ROOT / f'notebooks/Kaggle_{experiment}_30epochs_AutoResume.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    assert 'redirect_stdout' not in source and 'class Tee' not in source
    assert 'next_stage(' not in source and 'RelocatedTrainer' not in source
    assert '10.0' in source if experiment!='E6' else '9.5 * 3600' in source
