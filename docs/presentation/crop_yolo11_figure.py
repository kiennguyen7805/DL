"""Render and crop Figure 4 from arXiv:2501.13400 for presentation use."""

from pathlib import Path

import pymupdf
from PIL import Image


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "yolo11_paper_2501.13400.pdf"
OUT = ROOT / "assets" / "yolo11"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(PAPER)
    page = document[7]
    rendered = OUT / "paper_page_08.png"
    page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False).save(rendered)

    image = Image.open(rendered)
    # Coordinates refer to the 1836 x 2376 render of page 8. Crops retain
    # labels and connectors from the published figure; no diagram is redrawn.
    crops = {
        "yolo11_overall.png": (235, 145, 1515, 1515),
        "backbone_early.png": (250, 165, 555, 605),
        "backbone_middle.png": (250, 530, 555, 970),
        "backbone_deep.png": (250, 890, 555, 1285),
        "neck_context.png": (250, 1180, 570, 1510),
        "neck_topdown.png": (575, 520, 845, 1260),
        "neck_bottomup.png": (840, 520, 1305, 1510),
        "head_p3.png": (1230, 450, 1515, 690),
        "head_p4.png": (1230, 760, 1515, 1010),
        "head_p5.png": (1230, 1220, 1515, 1510),
    }
    for name, box in crops.items():
        image.crop(box).save(OUT / name, optimize=True)


if __name__ == "__main__":
    main()
