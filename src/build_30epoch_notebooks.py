"""Build three standalone Kaggle notebooks and audited epoch-10 source bundles.

Run with --prepare-bundles to read trusted project checkpoints. Never trains.
"""
import argparse
import ast
import csv
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

from src.kaggle_hf_shared import atomic_json, sha256_file
from src.rod_train_30epochs import check_yolo, check_e6, validate_metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'E3': {'recovery_full.pt': 'results/E3_YOLO11n/weights/recovery_full.pt',
           'best.pt': 'results/E3_YOLO11n/weights/best.pt'},
    'E4': {'recovery_full.pt': 'results/E4_YOLO11s/checkpoints/recovery_full.zip',
           'best.pt': 'results/E4_YOLO11s/checkpoints/best.pt'},
    'E6': {'checkpoint.pt': 'results/E6_Faster_RCNN/checkpoints/checkpoint.zip',
           'best_weights.pth': 'results/E6_Faster_RCNN/checkpoints/best_weights.zip'},
}
FROZEN = {'E3': 'configs/e3_pretrained.yaml', 'E4': 'configs/e4_yolo11s_pretrained.yaml'}


def make_cell(source, kind='code'):
    cell = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebooks():
    # Use the original successful notebooks; the custom extension runner is retired.
    from src.build_simple_30epoch_notebooks import build
    build()


def prepare_bundles():
    import torch
    import yaml
    names = yaml.safe_load((ROOT / 'data/processed/rod_detection_final/data.yaml').read_text(encoding='utf-8'))['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(25)]
    out = ROOT / 'artifacts/training_30epochs'
    out.mkdir(parents=True, exist_ok=True)
    audits = {}
    for experiment, source_paths in SOURCES.items():
        target = out / (experiment.lower() + '_resume_epoch010.zip')
        if target.exists():
            raise FileExistsError('Refusing to overwrite: ' + str(target))
        payload = torch.load(ROOT / next(iter(source_paths.values())), map_location='cpu', weights_only=False)
        manifest = {'schema': 1, 'experiment': experiment, 'completed_epoch': 10, 'class_names': names,
                    'dataset_yaml_sha256': sha256_file(ROOT / 'data/processed/rod_detection_final/data.yaml'),
                    'original_paths': source_paths, 'sha256': {}, 'metadata_origin': 'real local checkpoint'}
        with tempfile.TemporaryDirectory() as tmp:
            derived = {}
            if experiment in FROZEN:
                frozen = yaml.safe_load((ROOT / FROZEN[experiment]).read_text(encoding='utf-8'))
                check_yolo(payload, experiment, names, frozen, 10)
                data = payload['train_results']
                rows = [dict(zip(data, values)) for values in zip(*data.values(), strict=True)]
                validate_metrics(rows, 10)
                csv_path = Path(tmp) / 'results.csv'
                with csv_path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(data))
                    writer.writeheader(); writer.writerows(rows)
                args_path = Path(tmp) / 'args.yaml'
                args_path.write_text(yaml.safe_dump(payload['train_args'], allow_unicode=True), encoding='utf-8')
                derived = {'results.csv': csv_path, 'args.yaml': args_path}
                manifest.update(original_schedule_epochs=30, ultralytics_version=payload['version'],
                                optimizer_states=len(payload['optimizer']['state']))
                best = torch.load(ROOT / source_paths['best.pt'], map_location='cpu', weights_only=False)
                best_model = best.get('ema') or best.get('model')
                if best_model.yaml.get('scale') != ('n' if experiment == 'E3' else 's') or len(best_model.names) != 25:
                    raise RuntimeError('Wrong source best model')
                del best
            else:
                expected = '29e51f0af22227ab3722acb9c5378e0aab2adf239959af3be2b9e5ad7a6be2da'
                done, offset = check_e6(payload, expected, manifest['dataset_yaml_sha256'])
                if (done, offset) != (10, 0):
                    raise RuntimeError('Expected complete epoch 10 E6')
                manifest.update(original_config_hash=expected, scheduler=payload['scheduler'],
                                original_framework_version='MISSING in checkpoint; do not invent',
                                lr_at_epoch10=payload['scheduler']['_last_lr'])
                best = torch.load(ROOT / source_paths['best_weights.pth'], map_location='cpu', weights_only=False)
                if best['roi_heads.box_predictor.cls_score.weight'].shape[0] != 26:
                    raise RuntimeError('Wrong E6 best head')
                del best
                csv_path = Path(tmp) / 'metrics.csv'
                with csv_path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(payload['metrics_rows'][0]))
                    writer.writeheader(); writer.writerows(payload['metrics_rows'])
                derived = {'metrics.csv': csv_path}
            files = {name: ROOT / rel for name, rel in source_paths.items()} | derived
            manifest['sha256'] = {name: sha256_file(path) for name, path in files.items()}
            temp_target = target.with_suffix('.tmp')
            with zipfile.ZipFile(temp_target, 'w', compression=zipfile.ZIP_STORED) as archive:
                for name, path in files.items():
                    archive.write(path, name)
                archive.writestr('source_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            with zipfile.ZipFile(temp_target) as archive:
                if archive.testzip():
                    raise RuntimeError('Bundle CRC failed')
                import hashlib
                for name, expected in manifest['sha256'].items():
                    h = hashlib.sha256()
                    with archive.open(name) as stream:
                        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                            h.update(chunk)
                    if h.hexdigest() != expected:
                        raise RuntimeError('Bundle SHA256 differs: ' + name)
            os.replace(temp_target, target)
            audits[experiment] = {'bundle': str(target.relative_to(ROOT)), 'sha256': sha256_file(target),
                                  'source': manifest, 'checkpoint_schema': 'PASS', 'zip_crc_sha256': 'PASS'}
            print('AUDITED BUNDLE', experiment, target.stat().st_size, flush=True)
        del payload
    atomic_json(out / 'source_audit.json', audits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-bundles', action='store_true')
    args = parser.parse_args()
    if args.prepare_bundles:
        prepare_bundles()
    build_notebooks()
