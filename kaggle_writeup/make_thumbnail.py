"""Thumbnail cards of the Kaggle writeup: a test photo's solar disk with our submitted masks (reference/submission.csv),
next to the title and the scores. Writes kaggle_writeup/thumbnail.png (1200 x 630) and thumbnail_square.png (800 x 800).

    python kaggle_writeup/make_thumbnail.py          (needs the competition photos in data/, as the notebook downloads them)
"""
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import paths  # noqa: E402

PHOTO = "20140420195854Bh"                         # a test photo with 15 submitted filaments
BG, INK, MUTED, ACCENT = "#0d1117", "#f0f3f6", "#9aa4b2", "#ff8a3d"


def disk_with_masks(size):
    img = cv2.imread(str(paths.TEST_IMAGES / f"{PHOTO}.jpeg"), cv2.IMREAD_GRAYSCALE)
    csv.field_size_limit(10 ** 9)
    rles = [r[1] for r in csv.reader(open(REPO / "reference/submission.csv")) if r[0].rsplit("_", 1)[0] == PHOTO]
    m = np.zeros(img.shape, bool)
    for c in rles:
        m |= mu.decode(dict(size=[2048, 2048], counts=c.encode())).astype(bool)
    rgb = np.dstack([img] * 3).astype(np.float32) / 255
    rgb = np.clip((rgb - 0.08) * 1.25, 0, 1)                                   # a little more contrast on the disk
    acc = np.array([int(ACCENT[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    rgb[m] = 0.25 * rgb[m] + 0.75 * acc
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
    rgb[edge] = acc
    c, r = 1024, 960                                                           # disk centre and a margin around the limb
    crop = rgb[c - r:c + r, c - r:c + r]
    yy, xx = np.mgrid[-r:r, -r:r]
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    crop[(yy ** 2 + xx ** 2) > (r - 2) ** 2] = bg                              # round card, no square frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), len(rles)


def card(path, w, h, square=False):
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=BG)
    disk, n = disk_with_masks(560)
    if square:
        ax = fig.add_axes([0.16, 0.30, 0.68, 0.68]); text_x, top = 0.5, 0.25
    else:
        ax = fig.add_axes([0.02, 0.06, 0.88 * h / w, 0.88]); text_x, top = 0.53, 0.86
    ax.imshow(disk); ax.axis("off")
    ha = "center" if square else "left"
    t = fig.text
    if square:
        t(text_x, top, "Choosing, not producing", color=INK, fontsize=34, fontweight="bold", ha=ha, va="top")
        t(text_x, top - 0.085, "Mask2Former candidates, a learned PQ scorer, exact selection", color=MUTED, fontsize=16, ha=ha, va="top")
        t(text_x, top - 0.15, "OOF PQ 0.4801   |   Public 0.42   |   Private 0.39 (117th / 1,567)", color=ACCENT, fontsize=15,
          fontweight="bold", ha=ha, va="top")
    else:
        t(text_x, top, "KAGGLE  ·  SOLAR FILAMENT SEGMENTATION 2026", color=MUTED, fontsize=13, ha=ha, va="top")
        t(text_x, top - 0.08, "Choosing,", color=INK, fontsize=46, fontweight="bold", ha=ha, va="top")
        t(text_x, top - 0.19, "not producing", color=INK, fontsize=46, fontweight="bold", ha=ha, va="top")
        lines = ["Two Mask2Former families propose filaments;", "a LightGBM scorer predicts each mask's IoU",
                 "with the annotators; an integer program", "keeps the best non-overlapping set."]
        for i, s in enumerate(lines):
            t(text_x, top - 0.34 - 0.055 * i, s, color=MUTED, fontsize=16, ha=ha, va="top")
        y = top - 0.62
        for label, value in (("OOF PQ", "0.4801"), ("Public LB", "0.42"), ("Private LB", "0.39")):
            t(text_x, y, value, color=ACCENT, fontsize=28, fontweight="bold", ha=ha, va="top")
            t(text_x, y - 0.075, label, color=MUTED, fontsize=12, ha=ha, va="top")
            text_x += 0.165
        t(0.53, 0.06, f"117th of 1,567 teams  ·  orange: our {n} masks on a test photo", color=MUTED, fontsize=11,
          ha="left", va="bottom")
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(path, f"{w}x{h}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    card(out / "thumbnail.png", 1200, 630)
    card(out / "thumbnail_square.png", 800, 800, square=True)
