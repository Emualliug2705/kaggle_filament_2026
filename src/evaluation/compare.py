"""Mask-level comparison of two submission files (same photos, 2048 x 2048 COCO RLE rows): how many masks of one file
have a counterpart in the other at IoU > 0.5 (the competition's match rule), and how close the matched pairs are."""
import csv

import numpy as np


def _by_photo(path):
    csv.field_size_limit(10 ** 9)
    out = {}
    with open(path, newline="") as fh:
        for fid, counts in list(csv.reader(fh))[1:]:
            out.setdefault(fid.rsplit("_", 1)[0], []).append(counts)
    return out


def compare_csv(new, ref) -> dict:
    """masks_new, masks_ref, identical (rows with the same RLE in the same photo), matched (pairs at IoU > 0.5),
    mean_iou and min_iou of the matched pairs, photos_differing (photos with a mask that has no counterpart)."""
    from pycocotools import mask as mu
    a, b = _by_photo(new), _by_photo(ref)
    rle = lambda c: dict(size=[2048, 2048], counts=c.encode())                # noqa: E731
    ious, identical, differing = [], 0, 0
    for photo in sorted(set(a) | set(b)):
        x, y = a.get(photo, []), b.get(photo, [])
        identical += len(set(x) & set(y))
        hit = np.zeros((len(x), len(y)), bool)
        if x and y:
            iou = np.asarray(mu.iou([rle(c) for c in x], [rle(c) for c in y], [0] * len(y))).reshape(len(x), len(y))
            hit = iou > 0.5
            ious += iou[hit].tolist()
        differing += int(len(x) + len(y) - 2 * hit.sum() > 0)
    return dict(masks_new=sum(map(len, a.values())), masks_ref=sum(map(len, b.values())), identical=identical,
                matched=len(ious), mean_iou=float(np.mean(ious)) if ious else 0.0, min_iou=float(np.min(ious)) if ious else 0.0,
                photos_differing=differing)
