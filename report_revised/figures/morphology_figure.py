"""Fig. "morphology" of the report: submitted test masks drawn on their H-alpha photographs.

One full disk (each mask outlined in its own colour) and zoomed crops chosen to show the typical shapes.

    python morphology_figure.py SUBMISSION.csv TEST_IMAGES/ OUT.png DISK_STEM CROP_SPEC [CROP_SPEC ...]

CROP_SPEC is STEM:X:Y:SIZE:TITLE (crop centre and side in native pixels). Without CROP_SPEC, writes a preview of
DISK_STEM with mask numbers instead.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

SIZE = 2048
COLOURS = plt.get_cmap("tab10").colors


def load(submission: Path) -> dict[str, list[np.ndarray]]:
    masks: dict[str, list[np.ndarray]] = defaultdict(list)
    with open(submission, newline="") as handle:
        for row in csv.DictReader(handle):
            stem = row["filament_id"].rsplit("_", 1)[0]
            masks[stem].append(mu.decode(dict(size=[SIZE, SIZE], counts=row["segmentation_rle"].encode("ascii"))))
    return masks


def overlay(gray: np.ndarray, masks: list[np.ndarray], thickness: int) -> np.ndarray:
    """Each mask filled and outlined in the colour of its index, so a mask keeps its colour in every panel."""
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32) / 255.0
    for i, binary in enumerate(masks):
        if not binary.any():
            continue
        colour = np.array(COLOURS[i % len(COLOURS)])
        rgb[binary > 0] = 0.65 * rgb[binary > 0] + 0.35 * colour
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, tuple(float(c) for c in colour), thickness)
    return rgb


def main() -> None:
    submission, images, out, disk_stem, *specs = sys.argv[1:]
    masks = load(Path(submission))
    gray = lambda stem: cv2.imread(str(Path(images) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)  # noqa: E731
    if not specs:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(overlay(gray(disk_stem), masks[disk_stem], 3))
        for i, binary in enumerate(masks[disk_stem]):
            ys, xs = np.nonzero(binary)
            ax.text(xs.mean(), ys.min() - 10, str(i), color="yellow", fontsize=12)
        ax.set_title(f"{disk_stem}: {len(masks[disk_stem])} masks")
        fig.savefig(out, dpi=80, bbox_inches="tight")
        return

    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(2, 1 + (len(specs) + 1) // 2, width_ratios=[2] + [1] * ((len(specs) + 1) // 2),
                            wspace=0.03, hspace=0.12)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(overlay(gray(disk_stem), masks[disk_stem], 5))
    ax.set_title(f"(a) test photograph {disk_stem}: {len(masks[disk_stem])} masks", fontsize=10)
    ax.axis("off")
    for k, spec in enumerate(specs):
        stem, x, y, side, title = spec.split(":", 4)
        x, y, half = int(x), int(y), int(side) // 2
        y0, x0 = max(0, y - half), max(0, x - half)
        image = gray(stem)[y0:y + half, x0:x + half]
        crop = [binary[y0:y + half, x0:x + half] for binary in masks[stem]]
        sub = fig.add_subplot(grid[k % 2, 1 + k // 2])
        sub.imshow(overlay(image, crop, 1))
        sub.set_title(f"({chr(98 + k)}) {title}", fontsize=9)
        sub.axis("off")
        if stem == disk_stem:
            ax.add_patch(plt.Rectangle((x0, y0), image.shape[1], image.shape[0], fill=False, color="white", lw=1.2))
            ax.text(x0 + 15, y0 + 70, chr(98 + k), color="white", fontsize=10, weight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
