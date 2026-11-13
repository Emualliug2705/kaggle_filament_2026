"""Out-of-fold helpers: the annotations path, RLE conversion, the organisers' matching rule per (prediction set, annotator
record) and PQ from pooled counts."""
import numpy as np

from src import paths

ANN = str(paths.ANNOTATIONS)


def native(rle):
    from pycocotools import mask as mu
    c = rle["counts"]
    if isinstance(c, list):
        return mu.frPyObjects(rle, *rle["size"])
    return dict(size=rle["size"], counts=c.encode() if isinstance(c, str) else c)


def counts(pred, gt):
    """(tp, fp, fn, sum of matched IoU) and the IoU matrix gt x pred, the organisers' rule."""
    from pycocotools import mask as mu
    if not gt or not pred:
        return (0, len(pred), len(gt), 0.0), np.zeros((len(gt), len(pred)))
    iou = np.asarray(mu.iou(pred, gt, [0] * len(gt))).T
    hits = iou > 0.5
    tp = int(hits.sum())
    return (tp, len(pred) - tp, len(gt) - tp, float(iou[hits].sum())), iou


def pq(t):
    t = np.asarray(t, float)
    d = t[0] + t[1] / 2 + t[2] / 2
    return t[3] / d if d else 0.0
