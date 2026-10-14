import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "archive/e3_kaggle_evidence/rod_e3_pretrained_epoch010.zip"


class E3FinalizationTests(unittest.TestCase):
    def test_notebook_is_standalone_and_audit_only(self):
        notebook = ROOT / "notebooks/kaggle_rod_e3_finalize_10epochs.ipynb"
        obj = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(obj["nbformat"], 4)
        source = notebook.read_text(encoding="utf-8")
        self.assertNotIn("E5", source)
        self.assertNotIn(".train(", source)
        self.assertNotIn("/test/", source)
        for cell in obj["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(notebook), "exec")

    def test_artifact_metrics_checksums_and_original_schedule(self):
        self.assertTrue(ARTIFACT.is_file())
        sidecar = ARTIFACT.with_suffix(ARTIFACT.suffix + ".sha256")
        self.assertEqual(sidecar.read_text(encoding="ascii").split()[0],
                         hashlib.sha256(ARTIFACT.read_bytes()).hexdigest())
        with zipfile.ZipFile(ARTIFACT) as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
            self.assertNotIn("E5_scratch/weights/last.pt", names)
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["completed_epochs"], 10)
            self.assertEqual(manifest["original_schedule_epochs"], 30)
            self.assertEqual(manifest["current_main_budget_epochs"], 10)
            self.assertEqual(manifest["source_archive"]["sha256"],
                             "e60c3f490e25636c1a03fbef6e55bc7add534d760ffced84919b95c307646983")
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            rows = list(csv.DictReader(archive.read("E3_pretrained/results.csv").decode().splitlines()))
            self.assertEqual([int(row["epoch"]) for row in rows], list(range(1, 11)))
            self.assertAlmostEqual(float(rows[-1]["metrics/mAP50-95(B)"]), 0.60605)
            weights = manifest["checkpoints"]
            self.assertFalse(weights["best.pt"]["optimizer_present"])
            self.assertFalse(weights["last.pt"]["optimizer_present"])
            self.assertTrue(weights["resume_epoch010.pt"]["optimizer_present"])
            self.assertEqual(weights["best.pt"]["load"], "PASS")

    def test_packaged_best_checkpoint_runs_synthetic_cpu_inference(self):
        import numpy as np
        from ultralytics import YOLO
        with zipfile.ZipFile(ARTIFACT) as archive, tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "best.pt"
            target.write_bytes(archive.read("E3_pretrained/weights/best.pt"))
            model = YOLO(str(target))
            result = model.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                                   imgsz=64, device="cpu", verbose=False)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(model.names), 25)


if __name__ == "__main__":
    unittest.main()
