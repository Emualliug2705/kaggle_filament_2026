"""Disjoint painting of redrawn masks: each mask keeps the pixels not already claimed, in order; minimum 100 px."""
import numpy as np


def paint(masks):
    from pycocotools import mask as mu
    claimed = np.zeros((2048, 2048), np.uint8)
    out = []
    for m in masks:
        rem = m & (1 - claimed)
        if int(rem.sum()) >= 100:
            claimed |= rem
            out.append(mu.encode(np.asfortranarray(rem)))
    return out
