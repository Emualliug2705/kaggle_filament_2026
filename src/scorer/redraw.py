"""Full-resolution boundaries for a submission: each mask keeps its extent, its boundary within r px is redrawn from P,
the mean of the cnxL and r34 U-Net probability maps (test maps: mean of the 5 fold models), then the masks are
repainted in the file's order (= selection priority). With core = erode(M, r), outer = dilate(M, r), band = outer - core:

  1. P is moved 0.25 px towards the top of the photo (bilinear), where the annotators' outlines sit on average.
  2. The area of M' is fixed before its pixels are chosen, as the geometric mean of the Mask2Former area and the
     U-Net area:   area(M') = area(M) x clip(A_u / area(M), 0.8, 1.25) ^ 0.5,   A_u = area(core + (P > 0.5) in outer)
  3. M' = core + the band pixels with the highest P, up to that area.

    python -m src.scorer.redraw in.csv out.csv [r=2]
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src import paths
from src.scorer.paint import paint

ARCHES = tuple(os.environ.get("FILAMENT_REDRAW_ARCHES", "cnxL,r34").split(","))   # maps averaged for the redraw
MAPS = Path(os.environ.get("FILAMENT_REDRAW_TEST_MAPS", str(paths.SCRATCH / "maps" / "test")))
SHIFT_DY = -0.25          # px; negative = towards row 0 (the top of the photo)
AREA_WEIGHT = 0.5         # 0: the Mask2Former area is kept; 1: the U-Net area (inside outer) is taken
AREA_CLIP = (0.8, 1.25)   # limits of the U-Net / Mask2Former area ratio before the blend
AREA_LEVEL = 0.5          # probability level that defines the U-Net area


def shifted(p, dy=SHIFT_DY, dx=0.0):
    """p translated by (dx, dy) px, bilinear, edge values repeated."""
    import cv2
    if not dx and not dy:
        return p
    return cv2.warpAffine(p, np.float32([[1, 0, dx], [0, 1, dy]]), (p.shape[1], p.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def redraw(mask, P, r=2, dy=SHIFT_DY, weight=AREA_WEIGHT):
    """mask (2048 x 2048, 0/1) redrawn within r px from P (2048 x 2048, 0-255): see the module text."""
    import cv2
    ys, xs = np.nonzero(mask)
    pad = r + 8
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, 2048), max(xs.min() - pad, 0), min(xs.max() + pad + 1, 2048)
    m = np.ascontiguousarray(mask[y0:y1, x0:x1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core, outer = cv2.erode(m, k).astype(bool), cv2.dilate(m, k).astype(bool)
    band = outer & ~core
    p = shifted(np.ascontiguousarray(P[y0:y1, x0:x1], np.float32) / 255, dy)
    area = float(m.sum())
    area_u = float((((p > AREA_LEVEL) & outer) | core).sum())
    target = int(round(float(np.clip(area_u / area, *AREA_CLIP)) ** weight * area))
    need = target - int(core.sum())
    new = core.copy()
    v = p[band]
    if need >= len(v):
        new = outer
    elif need > 0:
        new[band] = v >= np.partition(v, len(v) - need)[len(v) - need]
    out = mask.copy()
    out[y0:y1, x0:x1] = new.astype(np.uint8)
    return out


def photo(job):
    from pycocotools import mask as mu
    stem, rles, j, r = job
    U = np.mean([np.asarray(np.load(MAPS / f"test{a}.npy", mmap_mode="r")[j, 0]).astype(np.float32) for a in ARCHES], 0)
    masks = [mu.decode(dict(size=[2048, 2048], counts=c.encode())) for c in rles]
    return stem, [o["counts"].decode() for o in paint([redraw(m, U, r) for m in masks])]


def main(src, dst, r=2):
    csv.field_size_limit(10 ** 9)
    files = json.loads((MAPS / "testcnxL.npy.json").read_text())["files"]
    assert files == json.loads((MAPS / "testr34.npy.json").read_text())["files"]
    pos = {Path(n).stem: j for j, n in enumerate(files)}
    by = {}
    for fid, c in list(csv.reader(open(src)))[1:]:
        stem, i = fid.rsplit("_", 1)
        by.setdefault(stem, []).append((int(i), c))
    jobs = [(s, [c for _, c in sorted(v)], pos[s], r) for s, v in sorted(by.items())]
    with ProcessPoolExecutor(int(os.environ.get("FILAMENT_CPUS", "24"))) as ex:
        res = dict(ex.map(photo, jobs, chunksize=2))
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["filament_id", "segmentation_rle"])
        for s in sorted(res):
            for i, c in enumerate(res[s], start=1):
                w.writerow([f"{s}_{i}", c])
    print(f"{sum(len(v) for v in by.values())} masks -> {sum(map(len, res.values()))} masks (boundary within {r} px) -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
