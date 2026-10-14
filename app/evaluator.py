"""Shared ROD detection metrics for all three model adapters.

Protocol: class-aware, one-to-one IoU matching; 101-point interpolated AP at
IoU .50:.05:.95; predictions collected at score >= .01, then max 100 scored
detections per image for AP and max 300 for operating Precision/Recall (the
same collection cap used during threshold calibration). ROD has no crowd/ignore
annotations, so these are COCO-style all-area metrics, not a pycocotools run.
"""

from __future__ import annotations

from dataclasses import dataclass

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
MAX_DET = 100
OPERATING_MAX_DET = 300
METRIC_PROTOCOL = {
    "iou_thresholds": list(IOU_THRESHOLDS), "ap_recall_points": 101,
    "max_detections_per_image": MAX_DET,
    "operating_max_detections_per_image": OPERATING_MAX_DET,
    "match": "class-aware greedy, score descending",
    "area": "all", "crowd_ignore": False,
    "note": "COCO-style interpolation; ROD has no crowd/ignore labels",
}


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class ClassMetrics:
    class_id: int
    gt_count: int
    ap50: float | None
    ap50_95: float | None
    tp: int
    fp: int
    fn: int


def _validate(records: list[dict]) -> None:
    if len({row["image"] for row in records}) != len(records):
        raise ValueError("Duplicate image in evaluation records")
    for row in records:
        width, height = row["width"], row["height"]
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")
        for kind in ("gt", "pred"):
            for item in row[kind]:
                if len(item) != (5 if kind == "gt" else 6):
                    raise ValueError("Invalid annotation/prediction row")
                cls = item[0]
                if isinstance(cls, bool) or int(cls) != cls or not 0 <= cls < 25:
                    raise ValueError("Class ID outside ROD 0..24")
                box = item[1:] if kind == "gt" else item[2:]
                # Float32 model/export rounding can put an edge microscopically
                # outside the image (for example 999.000061 on a 999 px image).
                # Keep the tolerance far below one pixel; genuinely out-of-bounds
                # boxes are still rejected and zero-area predictions remain FP.
                tolerance = 1e-3
                order = (box[0] < box[2] and box[1] < box[3]) if kind == "gt" else (
                    box[0] <= box[2] and box[1] <= box[3])
                if not (order and -tolerance <= box[0] <= width + tolerance and
                        -tolerance <= box[1] <= height + tolerance and
                        -tolerance <= box[2] <= width + tolerance and
                        -tolerance <= box[3] <= height + tolerance):
                    raise ValueError("Invalid pixel xyxy box")
                if kind == "pred" and not 0 <= item[1] <= 1:
                    raise ValueError("Invalid confidence")


def _class_outcomes(records: list[dict], class_id: int, threshold: float,
                    score_cutoff: float = 0.0,
                    max_det: int = MAX_DET) -> tuple[list[tuple[float, bool]], int]:
    ground_truth: dict[str, list[list[float]]] = {}
    predictions: list[tuple[float, str, list[float]]] = []
    for row in records:
        image = row["image"]
        ground_truth[image] = [item[1:] for item in row["gt"] if item[0] == class_id]
        selected = sorted(row["pred"], key=lambda item: -item[1])[:max_det]
        predictions.extend((float(item[1]), image, item[2:]) for item in selected
                           if item[0] == class_id and item[1] >= score_cutoff)
    predictions.sort(key=lambda item: -item[0])
    used: dict[str, set[int]] = {image: set() for image in ground_truth}
    outcomes = []
    for score, image, box in predictions:
        candidates = [(iou(box, gt), index) for index, gt in enumerate(ground_truth[image])
                      if index not in used[image]]
        best, index = max(candidates, default=(0.0, -1))
        hit = best >= threshold
        if hit:
            used[image].add(index)
        outcomes.append((score, hit))
    return outcomes, sum(map(len, ground_truth.values()))


def _average_precision(outcomes: list[tuple[float, bool]], gt_count: int) -> float | None:
    if gt_count == 0:
        return None
    true, false = 0, 0
    curve: list[tuple[float, float]] = []
    for _, hit in outcomes:
        true += int(hit)
        false += int(not hit)
        curve.append((true / gt_count, true / (true + false)))
    return sum(max((precision for recall, precision in curve if recall >= r), default=0.0)
               for r in (i / 100 for i in range(101))) / 101


def evaluate_records(records: list[dict], operating_threshold: float) -> dict:
    """Return identical metric definitions for E3, E4 and E6 records."""
    if not 0 < operating_threshold < 1:
        raise ValueError("Operating threshold must be in (0,1)")
    _validate(records)
    per_class = []
    for class_id in range(25):
        aps = []
        for match_iou in IOU_THRESHOLDS:
            outcomes, count = _class_outcomes(records, class_id, match_iou)
            aps.append(_average_precision(outcomes, count))
        selected, count = _class_outcomes(records, class_id, 0.5, operating_threshold,
                                           max_det=OPERATING_MAX_DET)
        tp = sum(hit for _, hit in selected)
        fp = len(selected) - tp
        per_class.append(ClassMetrics(
            class_id, count, aps[0],
            sum(aps) / len(aps) if count else None,
            tp, fp, count - tp,
        ))
    tp = sum(item.tp for item in per_class)
    fp = sum(item.fp for item in per_class)
    fn = sum(item.fn for item in per_class)
    active = [item for item in per_class if item.gt_count]
    return {
        "images": len(records), "ground_truth_objects": sum(item.gt_count for item in per_class),
        "operating_threshold": operating_threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "map50": sum(item.ap50 for item in active) / len(active) if active else None,
        "map50_95": sum(item.ap50_95 for item in active) / len(active) if active else None,
        "per_class": [item.__dict__ for item in per_class],
        "protocol": METRIC_PROTOCOL,
    }
