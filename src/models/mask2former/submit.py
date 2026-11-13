"""Emission of the selected masks: candidates above a score are painted in descending score (each pixel goes to the best
mask), optionally eroded first, masks under min_area pixels dropped, RLE encoded. The same emission is used for
validation sweeps and for the final selection, so a validated rule deploys unchanged."""
from __future__ import annotations

import cv2
from pycocotools import mask as mu

from src.evaluation.sweep import encode
from src.utils.masks import paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    """The emitted masks of one photo's candidates {score, rle} at score threshold conf (grow < 0: erosion in px)."""
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow != 0:
            raise ValueError(f"only unaltered and eroded masks are screened: {grow}")
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]
