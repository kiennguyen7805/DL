"""Unit tests for the independent ROD audit; no external dataset required."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_rod_conversion import raw_to_box, processed_to_box, compare_boxes, audit


class TestConversion(unittest.TestCase):
    def test_valid_box(self):
        cls, box, kind, clipped = raw_to_box("2 0.5 0.5 0.4 0.2", 4)
        self.assertEqual((cls, kind, clipped), (2, "box", False))
        self.assertAlmostEqual(box[0], 0.3)
        self.assertAlmostEqual(box[3], 0.6)

    def test_polygon_enclosing_bbox(self):
        cls, box, kind, clipped = raw_to_box("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6", 4)
        self.assertEqual((cls, kind, clipped), (1, "polygon", False))
        self.assertEqual(box, (0.1, 0.2, 0.5, 0.6))

    def test_clip_small_out_of_bounds(self):
        cls, box, kind, clipped = raw_to_box("0 0.995 0.5 0.03 0.4", 4)
        self.assertTrue(clipped)
        self.assertAlmostEqual(box[2], 1.0)
        self.assertAlmostEqual(box[0], 0.98)

    def test_invalid_raw(self):
        cases = ["4 0.5 0.5 0.2 0.2", "x 0.5 0.5 0.2 0.2",
                 "0 0.5 0.5 -0.1 0.1", "0 0.1 0.2 0.3", "0 NaN 0.5 0.1 0.1",
                 "0 0.1 0.1 0.2 0.2 0.3", "0 0.5 0.5 2.0 0.1"]
        for line in cases:
            with self.subTest(line=line), self.assertRaises(ValueError):
                raw_to_box(line, 4)

    def test_processed_only_boxes(self):
        with self.assertRaises(ValueError):
            processed_to_box("1 0.1 0.2 0.4 0.2 0.4 0.6", 4)
        self.assertEqual(processed_to_box("0 0.5 0.5 0.4 0.4", 4)[0], 0)

    def test_order_independent_match(self):
        raw = [raw_to_box("0 0.25 0.25 0.2 0.2", 2), raw_to_box("0 0.75 0.75 0.2 0.2", 2)]
        proc = [processed_to_box("0 0.75 0.75 0.2 0.2", 2), processed_to_box("0 0.25 0.25 0.2 0.2", 2)]
        matches, errors = compare_boxes(raw, proc, 0.0001)
        self.assertEqual(matches, 2)
        self.assertFalse(errors)

    def test_small_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for kind in ("raw", "proc"):
                root = base / kind
                root.mkdir()
                (root / "data.yaml").write_text("names: [person, car]\nnc: 2\n", encoding="utf-8")
                for split in ("train", "valid", "test"):
                    for part in ("images", "labels"):
                        (root / split / part).mkdir(parents=True)
                    (root / split / "images" / "frame.jpg").write_bytes(b"fake image bytes; this task checks filenames")
            for split in ("train", "valid", "test"):
                (base / "raw" / split / "labels" / "frame.txt").write_text(
                    "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n", encoding="utf-8")
                (base / "proc" / split / "labels" / "frame.txt").write_text(
                    "0 0.3 0.4 0.4 0.4\n", encoding="utf-8")
            result = audit(base / "raw", base / "proc", base / "report", 0.0001, 0.02, 5, 42)
            self.assertEqual(result["status"], "PASS")
            (base / "proc" / "train" / "labels" / "frame.txt").write_text(
                "0 0.9 0.9 0.1 0.1\n", encoding="utf-8")
            result = audit(base / "raw", base / "proc", base / "report", 0.0001, 0.02, 5, 42)
            self.assertEqual(result["status"], "FAIL")
            self.assertGreater(result["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()

