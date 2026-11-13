"""Full-resolution boundaries for a submission: each mask keeps its extent, its boundary within r px is redrawn from P,
the mean of the cnxL and r34 U-Net probability maps (test maps: mean of the 5 fold models), then the masks are
repainted in the file's order (= selection priority):

    M' = erode(M, r)  +  (P > t) inside dilate(M, r),   t = the threshold at which this best reproduces M (0.10-0.90)

    python -m src.scorer.redraw in.csv out.csv [r=3]
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


def redraw(mask, P, r, fixed=None):
    import cv2
    ys, xs = np.nonzero(mask)
    pad = r + 2
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, 2048), max(xs.min() - pad, 0), min(xs.max() + pad + 1, 2048)
    m = mask[y0:y1, x0:x1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core, outer = cv2.erode(m, k).astype(bool), cv2.dilate(m, k).astype(bool)
    p = P[y0:y1, x0:x1].astype(np.float32) / 255
    mb = m.astype(bool)
    if fixed is None:
        best, best_iou = 0.5, -1.0
        for t in np.arange(0.10, 0.91, 0.05):
            c = ((p > t) & outer) | core
            iou = (c & mb).sum() / max((c | mb).sum(), 1)
            if iou > best_iou:
                best, best_iou = t, iou
    else:
        best = fixed
    out = mask.copy()
    out[y0:y1, x0:x1] = (((p > best) & outer) | core).astype(np.uint8)
    return out


def photo(job):
    from pycocotools import mask as mu
    stem, rles, j, r = job
    U = np.mean([np.asarray(np.load(MAPS / f"test{a}.npy", mmap_mode="r")[j, 0]).astype(np.float32) for a in ARCHES], 0)
    masks = [mu.decode(dict(size=[2048, 2048], counts=c.encode())) for c in rles]
    return stem, [o["counts"].decode() for o in paint([redraw(m, U, r) for m in masks])]


def main(src, dst, r=3):
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
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
