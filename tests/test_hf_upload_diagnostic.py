import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diagnose_hf_upload import diagnose


class FakeApi:
    def __init__(self, failure=None):
        self.failure=failure;self.uploads=[]
    def whoami(self):
        return {'name':'not printed'}
    def repo_info(self,repo_type,**kwargs):
        if repo_type=='model':raise RuntimeError('Not found')
    def upload_file(self,**kwargs):
        if self.failure:raise self.failure
        self.uploads.append(kwargs)


def test_tiny_probe_does_not_touch_model_folders(tmp_path):
    api=FakeApi()
    def download(**kwargs):
        path=tmp_path/'downloaded.txt'
        assert kwargs['filename'].startswith('upload_diagnostics/probe_')
        path.write_bytes(api.uploads[-1]['path_or_fileobj'])
        return path
    report=diagnose(api,'hf_private_secret',download,tmp_path)
    assert report['status']=='PASS'
    assert report['training']=='NOT RUN'
    assert len(api.uploads)==1 and len(api.uploads[0]['path_or_fileobj'])<100
    assert 'hf_private_secret' not in (tmp_path/'hf_upload_diagnostic.json').read_text()


def test_badrequest_reason_and_status_are_visible_but_secret_is_hidden(tmp_path,capsys):
    class BadRequestError(Exception):
        response=SimpleNamespace(status_code=400)
        server_message='Storage limit hf_private_secret https://example.test/?token=private'
    api=FakeApi(BadRequestError('rejected'))
    def download(**kwargs):raise AssertionError('Upload failed, must not download')
    result=diagnose(api,'hf_private_secret',download,tmp_path)
    assert result['status']=='FAIL' and result['http_status']==400
    assert result['stage']=='upload_small_file'
    assert 'Storage limit' in result['server_message']
    text=capsys.readouterr().out+(tmp_path/'hf_upload_diagnostic.json').read_text()
    assert 'hf_private_secret' not in text and 'token=private' not in text


def test_download_mismatch_cannot_claim_pass(tmp_path):
    api=FakeApi()
    def download(**kwargs):
        path=tmp_path/'corrupted.txt';path.write_bytes(b'wrong');return path
    result=diagnose(api,'secret',download,tmp_path)
    assert result['status']=='FAIL' and result['upload_small_file']=='PASS'
    assert result['stage']=='download_checksum'
