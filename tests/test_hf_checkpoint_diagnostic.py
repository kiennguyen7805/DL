import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diagnose_hf_checkpoint_upload import DATASET_SHA256, REQUIRED, diagnose_checkpoints


def fixture_input(root, extracted=False, corrupt=False):
    data={name:('fixture '+name).encode() for name in REQUIRED}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in data.items()}
    expected={'archive':'results_nano.zip','model':'yolo11n.pt','folder':'YOLO 11n',
              'config':'config','checkpoint_sha256':hashes['recovery_full.pt']}
    manifest={'schema':2,'completed_epoch':13,'model':'yolo11n.pt','folder':'YOLO 11n',
              'config_sha256':'config','dataset_sha256':DATASET_SHA256,'sha256':hashes}
    prefix='run/hf_backup/snapshots/snapshot'
    root.mkdir()
    if corrupt:data['recovery_full.pt']=b'corrupted'
    if extracted:
        folder=root/prefix;folder.mkdir(parents=True)
        for name,value in data.items():(folder/name).write_bytes(value)
        (folder/'manifest.json').write_text(json.dumps(manifest))
    else:
        with zipfile.ZipFile(root/'results_nano.zip','w') as archive:
            for name,value in data.items():archive.writestr(prefix+'/'+name,value)
            archive.writestr(prefix+'/manifest.json',json.dumps(manifest))
    return {'E3':expected}


class FakeApi:
    def __init__(self,remote,error=None):self.remote=remote;self.error=error;self.uploads=[]
    def whoami(self):return {}
    def repo_info(self,**kwargs):return {}
    def upload_folder(self,folder_path,path_in_repo,**kwargs):
        if self.error:raise self.error
        assert path_in_repo.startswith('upload_diagnostics/checkpoints/E3/')
        assert 'latest.json' not in {p.name for p in Path(folder_path).iterdir()}
        shutil.copytree(folder_path,self.remote/path_in_repo)
        self.uploads.append(path_in_repo)
        return SimpleNamespace(oid='test_commit')
    def download(self,filename,revision,**kwargs):
        assert revision=='test_commit'
        return self.remote/filename


@pytest.mark.parametrize('extracted',[False,True])
def test_upload_folder_and_every_remote_hash(tmp_path,extracted):
    sources=fixture_input(tmp_path/'input',extracted=extracted)
    api=FakeApi(tmp_path/'remote')
    report=diagnose_checkpoints(api,'hf_test_secret',api.download,tmp_path/'input',tmp_path/'output',sources)
    assert report['status']=='PASS'
    assert report['experiments']['E3']['download_all_sha256']=='PASS'
    assert len(api.uploads)==1
    assert report['latest_json']=='NOT MODIFIED'
    assert report['training']=='NOT RUN'


def test_corrupt_source_stops_before_upload(tmp_path):
    sources=fixture_input(tmp_path/'input',corrupt=True)
    api=FakeApi(tmp_path/'remote')
    report=diagnose_checkpoints(api,'secret',api.download,tmp_path/'input',tmp_path/'output',sources)
    assert report['status']=='FAIL' and not api.uploads
    assert 'Source SHA256 mismatch' in report['experiments']['E3']['server_message']


def test_badrequest_response_captured_without_token(tmp_path,capsys):
    sources=fixture_input(tmp_path/'input')
    class BadRequestError(Exception):
        response=SimpleNamespace(status_code=400)
        server_message='Upload rejected hf_test_secret https://example.test/?secret=private'
    api=FakeApi(tmp_path/'remote',BadRequestError('rejected'))
    report=diagnose_checkpoints(api,'hf_test_secret',api.download,tmp_path/'input',tmp_path/'output',sources)
    result=report['experiments']['E3']
    assert result['status']=='FAIL' and result['stage']=='upload_folder'
    assert result['http_status']==400 and 'Upload rejected' in result['server_message']
    text=capsys.readouterr().out+json.dumps(report)
    assert 'hf_test_secret' not in text and 'secret=private' not in text


def test_unsafe_zip_does_not_write_outside_staging(tmp_path):
    sources=fixture_input(tmp_path/'input')
    with zipfile.ZipFile(tmp_path/'input/results_nano.zip','a') as archive:
        archive.writestr('../escape.txt',b'no')
    api=FakeApi(tmp_path/'remote')
    report=diagnose_checkpoints(api,'secret',api.download,tmp_path/'input',tmp_path/'output',sources)
    assert report['status']=='FAIL' and not api.uploads
    assert not (tmp_path/'escape.txt').exists()
