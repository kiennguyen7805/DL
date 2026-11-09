"""Checkpoint safety and VALID/TEST sequencing without GPU or real TEST images."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/evaluate_e6_30epochs_valid_test_kaggle.py'


def load_namespace():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    # The tested logic uses only stdlib; inference itself requires Kaggle CUDA.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(a.name in {'torch', 'torchvision', 'yaml'} for a in node.names)
        or isinstance(node, ast.ImportFrom) and node.module.startswith(('PIL', 'torchvision'))
        or isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
    )]
    namespace = {}
    exec(compile(tree, str(SCRIPT), 'exec'), namespace)
    return namespace


class CompleteE6Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns = load_namespace()
        self.ns['INPUT_ROOT'] = self.root / 'input'
        self.ns['INPUT_ROOT'].mkdir()
        self.ns['OUTPUT_ROOT'] = self.root / 'output'
        self.weights = b'new synthetic checkpoint'
        self.ns['EXPECTED_BEST_SHA256'] = hashlib.sha256(self.weights).hexdigest()
        self.manifest = {
            'model': 'fasterrcnn_resnet50_fpn_v2', 'class_count': 25,
            'target_epochs': 30, 'completed_epoch': 30, 'is_complete': True,
            'sha256': {'best_weights.pth': self.ns['EXPECTED_BEST_SHA256']},
        }

    def make_zip(self, weights=None, manifest=None):
        path = self.ns['INPUT_ROOT'] / 'results_e6_282930.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('rod_e6_faster_rcnn_hf/last_hf_manifest.json', json.dumps(manifest or self.manifest))
            archive.writestr('rod_e6_faster_rcnn_hf/best_weights.pth', weights or self.weights)
        return path

    def test_complete_zip_and_hash(self):
        self.make_zip()
        path, manifest, source = self.ns['find_checkpoint']()
        self.assertEqual(path.read_bytes(), self.weights)
        self.assertEqual(manifest['completed_epoch'], 30)
        self.assertEqual(source.name, 'results_e6_282930.zip')

    def test_rejects_28_epoch_manifest(self):
        self.make_zip(manifest={**self.manifest, 'completed_epoch': 28, 'is_complete': False})
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE 30/30'):
            self.ns['find_checkpoint']()

    def test_rejects_old_best_even_if_marked_complete(self):
        manifest = {**self.manifest, 'sha256': {'best_weights.pth': '39670d3a401bfd3e4bda7053878b9bb89cca179e618ad44a1751d75c33352af7'}}
        self.make_zip(manifest=manifest)
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE 30/30'):
            self.ns['find_checkpoint']()

    def test_rejects_corrupt_weights(self):
        self.make_zip(weights=b'wrong checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            self.ns['find_checkpoint']()

    def test_accepts_kaggle_extracted_input(self):
        folder = self.ns['INPUT_ROOT'] / 'rod_e6_faster_rcnn_hf'
        folder.mkdir()
        (folder / 'last_hf_manifest.json').write_text(json.dumps(self.manifest))
        (folder / 'best_weights.pth').write_bytes(self.weights)
        path, _, source = self.ns['find_checkpoint']()
        self.assertEqual(path.read_bytes(), self.weights)
        self.assertIsNone(source)

    def test_reselects_threshold_and_locks_before_test(self):
        ns = self.ns
        checkpoint = self.root / 'weights.pth'
        checkpoint.write_bytes(self.weights)
        dataset = self.root / 'data.yaml'
        dataset.write_text('synthetic dataset')
        manifest = {**self.manifest, 'dataset_sha256': ns['sha256_file'](dataset)}
        (ns['OUTPUT_ROOT'] / 'E6/checkpoint').mkdir(parents=True)
        ns['torch'] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'fake'), device=lambda _: None, __version__='fake')
        ns['find_dataset'] = lambda: (self.root, dataset, {'names': [str(i) for i in range(25)]})
        ns['find_checkpoint'] = lambda: (checkpoint, manifest, None)
        ns['build_model'] = lambda _: None
        sequence = []
        def collect(model, device, split, *args):
            sequence.append(split)
            lock_path = ns['OUTPUT_ROOT'] / 'E6/evaluation_lock.json'
            if split == 'test':
                self.assertEqual(sequence, ['valid', 'test'])
                self.assertTrue(lock_path.exists())
                lock = json.loads(lock_path.read_text())
                self.assertEqual(lock['threshold'], 0.8)
                self.assertTrue(lock['training_complete'])
                self.assertEqual(lock['checkpoint_sha256'], ns['EXPECTED_BEST_SHA256'])
            out = ns['OUTPUT_ROOT'] / 'E6' / split
            out.mkdir(parents=True)
            cache = out / 'predictions.jsonl'
            record = {'image': 'synthetic.jpg', 'gt': [[0, 0, 0, 10, 10]],
                      'pred': [[0, 0.8, 0, 0, 10, 10], [0, 0.3, 20, 20, 30, 30]]}
            cache.write_text(json.dumps(record) + '\n')
            return [record], cache, {}
        ns['collect_split'] = collect
        ns['main']()
        test = json.loads((ns['OUTPUT_ROOT'] / 'E6/test/metrics.json').read_text())
        self.assertEqual(test['operating_threshold'], 0.8)
        self.assertEqual(test['status'], 'FINAL_AFTER_VALID_LOCK')
        self.assertEqual((test['tp'], test['fp'], test['fn']), (1, 0, 0))
        self.assertEqual(test['lock_sha256'], ns['sha256_file'](ns['OUTPUT_ROOT'] / 'E6/evaluation_lock.json'))
        output_zip = self.root / 'evaluation_E6_30epochs_complete_results.zip'
        with zipfile.ZipFile(output_zip) as archive:
            self.assertIsNone(archive.testzip())

    def test_notebook_embeds_current_script(self):
        notebook = json.loads((ROOT / 'notebooks/Kaggle_E6_30epochs_Valid_Test.ipynb').read_text(encoding='utf-8'))
        code = ''.join(notebook['cells'][-1]['source'])
        self.assertEqual(code, SCRIPT.read_text(encoding='utf-8'))
        compile(code, 'notebook', 'exec')


if __name__ == '__main__':
    unittest.main()
