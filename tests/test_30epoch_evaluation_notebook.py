import ast
import json
import sys
from types import ModuleType

from src.build_30epoch_evaluation_notebook import ROOT, DISCOVER, bundled_sources


def test_notebook_embedded_code_is_current_and_compiles():
    book = json.loads((ROOT / 'notebooks/Kaggle_E3_E4_30epochs_Valid_Test.ipynb').read_text(encoding='utf-8'))
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
            assert not cell['outputs'] and cell['execution_count'] is None
    tree = ast.parse(''.join(book['cells'][2]['source']))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'SOURCES' for target in node.targets))
    assert ast.literal_eval(assignment.value) == bundled_sources()
    for path, code in bundled_sources().items():
        compile(code, path, 'exec')


def test_standalone_calibration_helpers(monkeypatch):
    module = ModuleType('kaggle_calibration_test')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(bundled_sources()['app/calibrate_confidence.py'], module.__dict__)
    records = [{'gt': [[0, 0, 0, 10, 10]], 'pred': [[0, 0.7, 0, 0, 10, 10]]}]
    selected = module.choose_threshold(module.score_predictions(records))
    assert selected.threshold == 0.7
    assert selected.f2 == 1.0


def test_kaggle_input_discovery_accepts_original_zip(tmp_path):
    # Execute the function definitions only, using a local input directory.
    source = DISCOVER[:DISCOVER.index('if DATASET_ROOT:')]
    env = {}
    exec(source, env)
    env['INPUT'] = tmp_path
    path = tmp_path / 'results_e3.zip'
    path.write_bytes(b'input')
    assert env['find_output']('E3', None) == path


def test_kaggle_input_discovery_repackages_extracted_output(tmp_path):
    import zipfile
    from tests.test_evaluate_30epoch_outputs import make_zip
    original = make_zip(tmp_path)
    inputs = tmp_path / 'input'
    with zipfile.ZipFile(original) as archive:
        archive.extractall(inputs)
    source = DISCOVER[:DISCOVER.index('if DATASET_ROOT:')]
    source = source.replace("Path('/kaggle/working/input_staging')", "INPUT / 'staging'")
    env = {}
    exec(source, env)
    env['INPUT'] = inputs
    repacked = env['find_output']('E3', None)
    from scripts.evaluate_30epoch_outputs import prepare_checkpoint
    checkpoint, audit = prepare_checkpoint(repacked, 'E3', tmp_path / 'out')
    assert audit['completed_epoch'] == 30
    assert checkpoint.read_bytes() == b'recovery_full.pt'
