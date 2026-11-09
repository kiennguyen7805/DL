"""Minimal 30-epoch versions of the exact notebooks used for the 10-epoch runs.

No custom trainer, stream wrappers, new training loop, or artificial stop at 20.
Original notebooks/configs remain untouched.
"""
import ast
import copy
import json
from pathlib import Path

import yaml

from src.kaggle_hf_shared import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Original code changed: ' + before[:80])
    return text.replace(before, after, 1)


def add_e6_progress_logs(runner):
    """Only add observation; keep training, validation and recovery operations intact."""
    edits = [
        ('    net.eval()\n    for images, targets in loader:',
         '    net.eval()\n'
         '    valid_seen = 0\n'
         '    valid_started = valid_last_log = time.monotonic()\n'
         '    print(f"E6 VALID start: {len(val_ds)} images", flush=True)\n'
         '    for images, targets in loader:'),
        ('    values = evaluator.compute()',
         '        valid_seen += len(images)\n'
         '        now = time.monotonic()\n'
         '        if valid_seen == 1 or valid_seen == len(val_ds) or now - valid_last_log >= 60:\n'
         '            print(f"E6 VALID {valid_seen}/{len(val_ds)} images "\n'
         '                  f"({100 * valid_seen / len(val_ds):.1f}%) elapsed={now - valid_started:.0f}s", flush=True)\n'
         '            valid_last_log = now\n'
         '    print("E6 VALID computing mAP...", flush=True)\n'
         '    values = evaluator.compute()'),
        ('        net.train()\n        for images, targets in loader:',
         '        net.train()\n'
         '        epoch_started = train_last_log = time.monotonic()\n'
         '        session_batches = 0\n'
         '        start_offset = offset\n'
         '        print(f"E6 TRAIN epoch {ep + 1}/{E6_EPOCHS} start: "\n'
         '              f"samples={offset}/{len(train_ds)} lr={optimizer.param_groups[0][\'lr\']:.6g}", flush=True)\n'
         '        for images, targets in loader:'),
        ('            offset += len(images)\n',
         '            offset += len(images)\n'
         '            session_batches += 1\n'
         '            now = time.monotonic()\n'
         '            if session_batches == 1 or offset == len(train_ds) or now - train_last_log >= 60:\n'
         '                elapsed = max(now - epoch_started, 1e-6)\n'
         '                rate = (offset - start_offset) / elapsed\n'
         '                eta = (len(train_ds) - offset) / max(rate, 1e-6)\n'
         '                print(f"E6 TRAIN epoch {ep + 1}/{E6_EPOCHS} "\n'
         '                      f"samples={offset}/{len(train_ds)} ({100 * offset / len(train_ds):.1f}%) "\n'
         '                      f"batch={running_count} loss_mean={running_loss / running_count:.4f} "\n'
         '                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)\n'
         '                train_last_log = now\n'),
        ('        completed_epoch = ep + 1\n        persist(',
         '        print(f"E6 epoch {ep + 1}/{E6_EPOCHS} metrics: "\n'
         '              f"mAP50={score50:.6f} mAP50-95={score:.6f} mAR100={recall:.6f} "\n'
         '              f"best_mAP50-95={best_map:.6f}; saving checkpoint...", flush=True)\n'
         '        completed_epoch = ep + 1\n        persist('),
        ('print("E6 completed epoch", completed_epoch, "/30; VALID mAP50-95", score)',
         'print("E6 completed epoch", completed_epoch, "/30; VALID mAP50-95", score, flush=True)'),
    ]
    for before, after in edits:
        runner = replace_once(runner, before, after)
    return runner


def helper_code():
    source = (ROOT/'src/rod_train_30epochs.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    selected = ('checked_member','atomic_copy','extract_bundle','discover_source')
    helpers = '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
                          if isinstance(n,ast.FunctionDef) and n.name in selected)
    return 'import zipfile\nfrom pathlib import PurePosixPath\n' + helpers + '\n\n' + '''
def choose_resume(store, previous_folder, previous_hash):
    # Continue from the destination once it has a verified recovery.
    current = store.restore()
    previous = HFStore(store.api, store.token, store.repo_type, previous_folder,
        store.model, previous_hash, store.dataset_hash, store.work / 'previous', max_epoch=30).restore()
    choices = [item for item in (current, previous) if item is not None]
    if not choices:
        from kaggle_secrets import UserSecretsClient
        try:
            source_token = UserSecretsClient().get_secret('HF_TOKEN2')
        except Exception:
            raise RuntimeError('New repo has no recovery. Enable HF_TOKEN2 with READ access to iraprp/deeplearning; HF_TOKEN writes to 13ffa/deep.') from None
        if not source_token or not source_token.strip():
            raise RuntimeError('Missing read-only source credential HF_TOKEN2')
        for folder, config, suffix in ((store.folder, store.config_hash, 'source'),
                                       (previous_folder, previous_hash, 'source_extension')):
            source = HFStore(store.api, source_token, 'dataset', folder, store.model,
                config, store.dataset_hash, store.work / suffix, max_epoch=30,
                repo_id='iraprp/deeplearning')
            recovered = source.restore()  # Source errors propagate; never silently restart.
            if recovered is not None:
                choices.append(recovered)
        print('Read-only source: iraprp/deeplearning; uploads: 13ffa/deep', flush=True)
    if not choices:
        return None
    selected = max(choices, key=lambda item: (item[0]['completed_epoch'], item[0].get('next_sample',0)))
    print('Verified resume:', selected[0]['folder'], 'completed_epoch=', selected[0]['completed_epoch'],
          'next_sample=', selected[0].get('next_sample',0), flush=True)
    print('Training target: 30 total epochs; completed=', selected[0]['completed_epoch'],
          'remaining=', 30 - selected[0]['completed_epoch'],
          '; continue to 30 or session deadline, no 10-epoch stop', flush=True)
    print('Checkpoint destination: 13ffa/deep/' + store.folder, flush=True)
    return selected

def yolo_resume_directory(checkpoint):
    # Keep the original trainer and restore its historical output directory.
    args = torch.load(checkpoint, map_location='cpu', weights_only=False)['train_args']
    path = Path(args.get('save_dir') or str(Path(args['project']) / args['name']))
    if not path.resolve().is_relative_to(Path('/kaggle/working').resolve()):
        raise RuntimeError('Checkpoint output path is outside /kaggle/working')
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_e3_input():
    source = Path('/kaggle/working/e3_input_recovery')
    origin = discover_source('/kaggle/input', source, 'E3', sha256_file)
    if origin['dataset_yaml_sha256'] != sha256_file(source_yaml) or origin['class_names'] != class_names:
        raise RuntimeError('E3 input differs from ROD Final')
    return ({'completed_epoch':10,'next_sample':0}, {name:source/name for name in origin['sha256']})
'''


def build():
    audit = json.loads((ROOT/'artifacts/training_30epochs/source_audit.json').read_text(encoding='utf-8'))
    shared = (ROOT/'src/kaggle_hf_shared.py').read_text(encoding='utf-8')
    shared = replace_once(shared, 'HF_REPO_ID = "iraprp/deeplearning"', 'HF_REPO_ID = "13ffa/deep"')
    shared = replace_once(shared, 'HF_SECRET_NAME = "HF_TOKEN2"', 'HF_SECRET_NAME = "HF_TOKEN"')
    shared = shared.replace('Thiếu Kaggle Secret HF_TOKEN2', 'Thiếu Kaggle Secret HF_TOKEN')
    shared = shared.replace('max_epoch=10','max_epoch=30')
    shared = shared.replace('"YOLO 11s", "Faster R-CNN",', '"YOLO 11n", "YOLO 11s", "Faster R-CNN",')
    shared = replace_once(shared,
        'if manifest.get("is_complete") is not (epoch == max_epoch):',
        'legacy_ten = (max_epoch == 30 and epoch == 10 and manifest.get("is_complete") is True\n'
        '                  and manifest.get("target_epochs", 10) == 10)\n'
        '    if not legacy_ten and manifest.get("is_complete") is not (epoch == max_epoch):')
    for experiment in ('E3','E4','E6'):
        original = ('Kaggle_E6_FasterRCNN_HF_AutoResume.ipynb' if experiment=='E6'
                    else 'Kaggle_E4_YOLO11s_HF_AutoResume.ipynb')
        book = json.loads((ROOT/'notebooks'/original).read_text(encoding='utf-8'))
        # Derive the old extension fingerprint so a verified later checkpoint wins.
        identity = {'experiment':experiment,'target_epochs':30,'milestones':[10,20,30],
                    'source_manifest':audit[experiment]['source'],'protocol':'extension30-v1',
                    'scheduler_policy':'preserve_original','test_used':False}
        old_hash = fingerprint(identity)
        runner = ''.join(book['cells'][5]['source'])
        if experiment in ('E3','E4'):
            if experiment=='E3':
                tree = ast.parse(runner)
                assignment = next(n for n in tree.body if isinstance(n,ast.Assign) and
                                  any(isinstance(t,ast.Name) and t.id=='FROZEN_E4_YAML' for t in n.targets))
                config = yaml.safe_load((ROOT/'configs/e3_pretrained.yaml').read_text(encoding='utf-8'))
                config['main_budget_epochs'] = 10  # historical metadata, not new stop budget
                runner = runner.replace(ast.get_source_segment(runner, assignment),
                    'FROZEN_E4_YAML = '+repr(yaml.safe_dump(config,allow_unicode=True)))
                runner = runner.replace('E4','E3').replace('yolo11s','yolo11n').replace('YOLO11s','YOLO11n')
                runner = runner.replace('"YOLO 11s"','"YOLO 11n"').replace('!= "s"','!= "n"')
            tag = experiment
            runner = replace_once(runner, f'{tag}_BUDGET = 10', f'{tag}_BUDGET = 30')
            runner = replace_once(runner, 'restored = store.restore()',
                f'restored = choose_resume(store, "{tag}_30epochs", "{old_hash}")')
            runner = replace_once(runner,
                f'    resume_file = {tag}_OUTPUT / "train" / "weights" / "last.pt"',
                '    resume_run = yolo_resume_directory(recovered["recovery_full.pt"])\n'
                '    resume_file = resume_run / "weights" / "last.pt"')
            runner = runner.replace(f'{tag}_OUTPUT / "train" / name','resume_run / name')
            runner = replace_once(runner, '    completed, resume_file = 0, None',
                '    raise RuntimeError("No verified recovery; refusing restart at epoch 1")')
            if experiment=='E3':
                runner = replace_once(runner, 'if restored:\n    state, recovered = restored',
                    'if restored is None:\n    restored = load_e3_input()\n\nif restored:\n    state, recovered = restored')
            # Save exact milestone full states in the existing callback, no new callback.
            anchor = f'        atomic_json({tag}_OUTPUT / "last_hf_manifest.json", manifest)'
            runner = replace_once(runner, anchor, anchor+'\n'
                '        if epoch_done in (10, 20, 30):\n'
                f'            shutil.copy2(local_full, {tag}_OUTPUT / ("resume_epoch%03d.pt" % epoch_done))')
            runner = runner.replace('COMPLETED 10/10','COMPLETED 30/30').replace('"/10"','"/30"')
        else:
            runner = replace_once(runner, 'E6_EPOCHS, E6_BATCH, E6_BACKUP_BATCHES = 10, 2, 1800',
                                  'E6_EPOCHS, E6_BATCH, E6_BACKUP_BATCHES = 30, 2, 1800')
            # Preserve the epoch-10 source config identity; extension budget is logged separately.
            runner = replace_once(runner, '"epochs": E6_EPOCHS,', '"epochs": 10,')
            runner = replace_once(runner, 'restored = store.restore()',
                f'restored = choose_resume(store, "E6_30epochs", "{old_hash}")\n'
                'if restored is None:\n    raise RuntimeError("No verified E6 recovery; refusing restart at epoch 1")')
            # Keep existing local recovery cadence and exact training loop. Limit
            # remote traffic to epoch ends, deadline, or one mid-epoch save/hour.
            runner = replace_once(runner,
                'def persist(ep, offset, done, running_loss, running_count):',
                'last_hf_sync = time.monotonic()\nE6_HF_INTERVAL_SECONDS = 3600\n\n'
                'def persist(ep, offset, done, running_loss, running_count):\n'
                '    global last_hf_sync')
            runner = replace_once(runner, '    os.replace(temp, path)\n    files =',
                '    os.replace(temp, path)\n'
                '    if (offset > 0 and time.monotonic() < E6_DEADLINE\n'
                '        and time.monotonic() - last_hf_sync < E6_HF_INTERVAL_SECONDS):\n'
                '        print("E6 local recovery saved; HF waits for epoch end/hour/deadline", flush=True)\n'
                '        return\n'
                '    files =')
            runner = replace_once(runner, '    atomic_json(E6_WORK / "last_hf_manifest.json", latest)',
                '    atomic_json(E6_WORK / "last_hf_manifest.json", latest)\n'
                '    last_hf_sync = time.monotonic()')
            runner = runner.replace('COMPLETED 10/10','COMPLETED 30/30').replace('"/10"','"/30"').replace('/10; VALID','/30; VALID')
            runner = replace_once(runner, '    atomic_json(E6_WORK / "last_hf_manifest.json", latest)',
                '    atomic_json(E6_WORK / "last_hf_manifest.json", latest)\n'
                '    if offset == 0 and done in (10,20,30):\n'
                '        import shutil\n'
                '        shutil.copy2(path, E6_WORK / ("resume_epoch%03d.pt" % done))')
            runner = add_e6_progress_logs(runner)
        # All future manifests declare budget 30; legacy COMPLETE-at-10 is read-only compatible.
        shared_cell = shared + '\n\n' + helper_code()
        runner = runner.replace('dataset_hash, E4_OUTPUT / "hf_backup")','dataset_hash, E4_OUTPUT / "hf_backup", max_epoch=30, compact=True)')
        runner = runner.replace('dataset_hash, E3_OUTPUT / "hf_backup")','dataset_hash, E3_OUTPUT / "hf_backup", max_epoch=30, compact=True)')
        runner = runner.replace('dataset_hash, E6_WORK / "hf_backup")','dataset_hash, E6_WORK / "hf_backup", max_epoch=30, compact=True)')
        shared_cell = replace_once(shared_cell,
            '"dataset_sha256": self.dataset_hash,', '"dataset_sha256": self.dataset_hash, "target_epochs": 30,')
        book['cells'][4]['source'] = shared_cell.splitlines(keepends=True)
        book['cells'][5]['source'] = runner.splitlines(keepends=True)
        folder = {'E3':'YOLO 11n','E4':'YOLO 11s','E6':'Faster R-CNN'}[experiment]
        book['cells'][2]['source'] = [f'HF_FOLDER = {folder!r}\n',"print('HF checkpoint folder:', HF_FOLDER, flush=True)"]
        book['cells'][0]['source'] = [f'# {experiment}: code notebook cũ, nâng giới hạn từ 10 lên 30 epoch\n',
            'Dùng trainer/loop cũ, không Tee, không custom Trainer. Giữ hyperparameter và scheduler cũ.\n',
            'Tự chọn recovery đã xác minh có tiến độ cao hơn giữa folder gốc và folder mở rộng trước đó.\n',
            'Mục tiêu TỔNG 30 epoch: đã xong 10 thì chạy tiếp 20 epoch còn lại; không chia mỗi lần 10 epoch. Chỉ dừng sớm khi tới deadline phiên.\n',
            'E6 giữ local recovery mỗi 1800 batches; HF tại epoch end, mốc một giờ giữa epoch, hoặc deadline.\n',
            'Bật GPU, Internet; HF_TOKEN ghi 13ffa/deep, HF_TOKEN2 đọc iraprp/deeplearning khi repo mới chưa có recovery.\n',
            'Gắn ROD Final. E3 cần Input bundle nếu cả hai repo chưa có recovery.\n',
            f'HF lưu tiếp vào `13ffa/deep/{folder}/`. Repo cũ chỉ đọc. Không chạy đồng thời notebook khác cho cùng model.\n']
        book['cells'][6]['source'] = ['## Kết quả\n',
            'Kiểm tra completed_epoch=30, is_complete=true và checksum. Save Version/Download Output.\n',
            'Giữ mốc 10 cũ và resume_epoch020.pt/resume_epoch030.pt trong Output; mỗi snapshot HF bất biến cũng giữ tiến độ tương ứng.\n',
            'Không dùng TEST trong training. Không reset LR; E6 giữ StepLR cũ nên LR tiếp tục rất nhỏ.\n']
        for i,cell in enumerate(book['cells']):
            cell['id'] = experiment.lower()+'-simple30-'+str(i)
            if cell['cell_type']=='code':
                cell['execution_count']=None;cell['outputs']=[]
                code=''.join(cell['source'])
                if not code.lstrip().startswith('%pip'):
                    compile(code,experiment,'exec')
        output = ROOT/'notebooks'/f'Kaggle_{experiment}_30epochs_AutoResume.ipynb'
        output.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print('BUILT simple old-code 30:',output.name)


if __name__=='__main__':
    build()
