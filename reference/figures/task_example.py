"""reference/figures/task_example.png: one training photo and its two annotators' filaments (README, task presentation).

    python reference/figures/task_example.py [photo file name]      (reads the competition data through src/paths.py)
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import paths  # noqa: E402

PHOTO = sys.argv[1] if len(sys.argv) > 1 else "20130425135954Bh.jpeg"
# Fixed-order categorical colours, one per filament of an annotator.
COLOURS = ["#2a9d8f", "#e76f51", "#e9c46a", "#8ab17d", "#f4a261", "#6d9dc5", "#c77dff", "#ff8fab", "#48cae4", "#b5838d",
           "#90be6d", "#f9844a"]


def filaments(coco, record_id, h, w):
    out = []
    for a in coco["annotations"]:
        if a["image_id"] != record_id:
            continue
        seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
        polygons = [p for p in seg if len(p) >= 6]
        if polygons:
            out.append(mu.decode(mu.merge(mu.frPyObjects(polygons, h, w))))
    return out


def main():
    coco = json.loads(paths.ANNOTATIONS.read_text())
    records = sorted(r["id"] for r in coco["images"] if r["file_name"] == PHOTO)
    gray = cv2.imread(str(paths.TRAIN_IMAGES / PHOTO), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255
    panels = [("H-alpha photograph (2048 x 2048)", rgb)]
    for k, rid in enumerate(records[:2], 1):
        img = rgb.copy()
        masks = filaments(coco, rid, h, w)
        for i, m in enumerate(masks):
            colour = np.array(matplotlib.colors.to_rgb(COLOURS[i % len(COLOURS)]))
            img[m > 0] = 0.25 * img[m > 0] + 0.75 * colour
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, tuple(float(c) for c in colour), 6)
        panels.append((f"annotator {k}: {len(masks)} filaments, one mask each", img))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6), dpi=100)
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(cv2.resize(img, (900, 900), interpolation=cv2.INTER_AREA))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.suptitle(f"Training photo {PHOTO}: each annotator's drawing is scored separately", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=100, facecolor="white")
    print(out, f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
