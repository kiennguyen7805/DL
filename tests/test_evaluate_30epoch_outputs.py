import json
import zipfile

import pytest

from scripts.download_hf_checkpoints import DATASET_HASH, MODELS, REQUIRED
from scripts.evaluate_30epoch_outputs import prepare_checkpoint, read_cache
import hashlib


def make_zip(tmp_path, corrupt=False, completed=30):
    folder, model, config = MODELS['E3']
    snap = folder + '/snapshots/example123'
    data = {name: name.encode() for name in REQUIRED - {'best.pt'}}
    data['results.csv'] = ('epoch,metric\n' + ''.join(f'{i},0.5\n' for i in range(1, 31))).encode()
    manifest = {'schema': 2, 'folder': folder, 'model': model, 'config_sha256': config,
                'dataset_sha256': DATASET_HASH, 'completed_epoch': completed,
                'is_complete': True, 'target_epochs': 30, 'snapshot': snap,
                'sha256': {name: hashlib.sha256(value).hexdigest() for name, value in data.items()},
                'artifact_aliases': {'best.pt': 'recovery_full.pt'}}
    if corrupt:
        data['recovery_full.pt'] = b'bad'
    path = tmp_path / 'output.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('run/last_hf_manifest.json', json.dumps(manifest))
        for name, value in data.items():
            archive.writestr('run/hf_backup/download/' + snap + '/' + name, value)
        archive.writestr('run/train/weights/best.pt', b'unverified native best')
    return path


def test_zip_uses_verified_alias_instead_of_native_best(tmp_path):
    checkpoint, audit = prepare_checkpoint(make_zip(tmp_path), 'E3', tmp_path / 'results')
    assert checkpoint.read_bytes() == b'recovery_full.pt'
    assert audit['completed_epoch'] == 30
    assert audit['best_role'] == 'recovery_full.pt alias'


@pytest.mark.parametrize('kwargs', [{'corrupt': True}, {'completed': 29}])
def test_bad_zip_rejected(tmp_path, kwargs):
    with pytest.raises(RuntimeError):
        prepare_checkpoint(make_zip(tmp_path, **kwargs), 'E3', tmp_path / 'results')


def test_torn_prediction_line_recovered(tmp_path):
    path = tmp_path / 'cache.jsonl'
    path.write_bytes(b'{"image":"one.jpg"}\n{"image":')
    assert read_cache(path) == [{'image': 'one.jpg'}]
    assert path.read_bytes() == b'{"image":"one.jpg"}\n'


def test_corruption_before_final_line_is_rejected(tmp_path):
    path = tmp_path / 'cache.jsonl'
    path.write_bytes(b'invalid\n{"image":"one.jpg"}\n')
    with pytest.raises(RuntimeError):
        read_cache(path)


def test_complete_final_json_without_newline_can_resume(tmp_path):
    path = tmp_path / 'cache.jsonl'
    path.write_bytes(b'{"image":"one.jpg"}')
    assert read_cache(path) == [{'image': 'one.jpg'}]
    assert path.read_bytes().endswith(b'\n')
