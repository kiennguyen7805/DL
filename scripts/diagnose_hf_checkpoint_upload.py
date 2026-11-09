"""Kaggle CPU-only upload_folder diagnostic on audited epoch-13 E3/E4 files.

Accept raw user Output ZIPs or their Kaggle-extracted contents. Never torch.load,
train, change latest.json, create a repository, or overwrite model artifacts.
"""
import hashlib
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from scripts.diagnose_hf_upload import error_detail

REPO_ID = 'iraprp/deeplearning'
SECRET_NAME = 'HF_TOKEN2'
SOURCES = {
    'E3': {'archive': 'results_nano.zip', 'model': 'yolo11n.pt', 'folder': 'YOLO 11n',
           'config': '707ac9a6ed2eb1545007c8dd4ac752429731ca9b29cd585f8319c375d3de0d70',
           'checkpoint_sha256': '664afee355b5a63f9101010fe14fc4d7c2ec42a2c21afb2d710be1255cda5fba'},
    'E4': {'archive': 'results_small.zip', 'model': 'yolo11s.pt', 'folder': 'YOLO 11s',
           'config': '684c8c281633c537094601c84bf3cdea5b674126230a69b74004380d6276aa70',
           'checkpoint_sha256': 'eec7239685124fcd3603df628fdca26bf2fc954afe0a937545fabd41ea424bc6'},
}
DATASET_SHA256 = '84a776e32c59063930b3e036d91876921163f48de70a9e85eb8c2b3cc2ddcf52'
REQUIRED = {'recovery_full.pt','best.pt','last.pt','args.yaml','results.csv','config.json','environment.json'}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise RuntimeError('Unsafe archive/artifact path')


def check_manifest(manifest, expected):
    if (manifest.get('schema') != 2 or manifest.get('completed_epoch') != 13 or
            manifest.get('model') != expected['model'] or manifest.get('folder') != expected['folder'] or
            manifest.get('config_sha256') != expected['config'] or
            manifest.get('dataset_sha256') != DATASET_SHA256):
        raise RuntimeError('Source is not the audited epoch-13 model/config/dataset')
    hashes = manifest.get('sha256', {})
    if set(hashes) != REQUIRED:
        raise RuntimeError('Missing or unexpected epoch-13 artifacts')
    for name in hashes:
        safe_name(name)
        if '/' in name:
            raise RuntimeError('Expected flat artifact names')
    if hashes['recovery_full.pt'] != expected['checkpoint_sha256']:
        raise RuntimeError('Recovery hash differs from locally audited Output')


def prepare_source(input_root, stage, experiment, expected):
    """Read only the failed local snapshot, never cached successful epoch-12 files."""
    root, stage = Path(input_root), Path(stage)
    archives = list(root.rglob(expected['archive']))
    extracted = []
    for path in root.rglob('manifest.json'):
        if '/hf_backup/snapshots/' not in path.as_posix():
            continue
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if manifest.get('model') == expected['model'] and manifest.get('completed_epoch') == 13:
            extracted.append((path, manifest))
    if len(archives) + len(extracted) != 1:
        raise RuntimeError(f'{experiment}: attach exactly one {expected["archive"]} or its extracted Output. '
                           f'ZIPs={len(archives)}, extracted epoch-13 snapshots={len(extracted)}. '
                           'Do not attach duplicate raw and extracted copies.')
    stage.mkdir(parents=True, exist_ok=False)
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError('Duplicate archive member')
            for name in names:
                safe_name(name)
            if archive.testzip():
                raise RuntimeError('Output ZIP CRC failed')
            choices = []
            for name in names:
                if '/hf_backup/snapshots/' in name and name.endswith('/manifest.json'):
                    m = json.loads(archive.read(name))
                    if m.get('model') == expected['model'] and m.get('completed_epoch') == 13:
                        choices.append((name, m))
            if len(choices) != 1:
                raise RuntimeError('Missing or duplicate epoch-13 staged snapshot in Output ZIP')
            name, manifest = choices[0]
            check_manifest(manifest, expected)
            prefix = name.rsplit('/',1)[0]
            for artifact in sorted(REQUIRED):
                with archive.open(prefix+'/'+artifact) as source, (stage/artifact).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
            (stage/'manifest.json').write_bytes(archive.read(name))
    else:
        path, manifest = extracted[0]
        check_manifest(manifest, expected)
        for artifact in sorted(REQUIRED | {'manifest.json'}):
            source = path.parent/artifact
            if source.resolve().parent != path.parent.resolve():
                raise RuntimeError('Extracted source escapes expected snapshot directory')
            shutil.copyfile(source, stage/artifact)
    for artifact, expected_hash in manifest['sha256'].items():
        if sha256_file(stage/artifact) != expected_hash:
            raise RuntimeError('Source SHA256 mismatch: '+artifact)
    return manifest


def diagnose_checkpoints(api, token, download, input_root, output_root, sources=None):
    report = {'repository': REPO_ID, 'repo_type':'dataset', 'training':'NOT RUN',
              'latest_json':'NOT MODIFIED', 'experiments':{}, 'status':'FAIL'}
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    sources = SOURCES if sources is None else sources
    try:
        api.whoami()
        api.repo_info(repo_id=REPO_ID, repo_type='dataset')
    except Exception as error:
        report['stage']='authentication_or_repository'
        report.update(error_detail(error, token))
    else:
        for experiment, expected in sources.items():
            item = {'status':'FAIL', 'stage':'prepare_source', 'epoch':13}
            report['experiments'][experiment] = item
            run_id = uuid.uuid4().hex
            stage = output/'staging'/experiment/run_id
            remote = 'upload_diagnostics/checkpoints/'+experiment+'/'+run_id
            item['remote_folder'] = remote
            try:
                print(experiment, 'checking epoch-13 source...', flush=True)
                manifest = prepare_source(input_root, stage, experiment, expected)
                expected_hashes = dict(manifest['sha256'], **{'manifest.json':sha256_file(stage/'manifest.json')})
                item.update(source_sha256='PASS', stage='upload_folder',
                            total_bytes=sum(p.stat().st_size for p in stage.iterdir()))
                print(experiment, 'upload_folder:',remote,'bytes=',item['total_bytes'], flush=True)
                commit = api.upload_folder(folder_path=str(stage), path_in_repo=remote,
                    repo_id=REPO_ID, repo_type='dataset', token=token,
                    commit_message=f'ROD diagnostic: {experiment} audited epoch 13, no pointer change')
                item.update(upload_folder='PASS',stage='download_sha256')
                revision = getattr(commit,'oid',None)
                item['commit']=revision
                for artifact, expected_hash in expected_hashes.items():
                    kwargs = dict(repo_id=REPO_ID,repo_type='dataset',filename=remote+'/'+artifact,
                                  token=token,local_dir=str(output/'download'/experiment/run_id))
                    if revision:
                        kwargs['revision']=revision
                    downloaded=download(**kwargs)
                    if sha256_file(downloaded)!=expected_hash:
                        raise RuntimeError('Remote artifact SHA256 mismatch: '+artifact)
                    print(experiment,'verified:',artifact,flush=True)
                item.update(status='PASS',stage='complete',download_all_sha256='PASS',
                            recovery_sha256=expected_hashes['recovery_full.pt'])
            except Exception as error:
                item.update(error_detail(error, token))
            (output/'checkpoint_upload_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            print(json.dumps({experiment:item},indent=2,ensure_ascii=False),flush=True)
        if report['experiments'] and all(r['status']=='PASS' for r in report['experiments'].values()):
            report['status']='PASS'
    report['scope']='Upload diagnostics only. No training latest.json was advanced; do not resume from epoch 12 until recovery is promoted.'
    (output/'checkpoint_upload_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    print('Send checkpoint_upload_report.json or the JSON above; never send the token.',flush=True)
    return report


def main():
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import HfApi, hf_hub_download
    token=UserSecretsClient().get_secret(SECRET_NAME)
    if not token or not token.strip():
        raise RuntimeError('Missing Kaggle Secret HF_TOKEN2')
    return diagnose_checkpoints(HfApi(token=token),token,hf_hub_download,
                                '/kaggle/input','/kaggle/working/hf_checkpoint_upload_diagnostic')


if __name__=='__main__':main()
