"""Validation records (each annotator's masks per photo, held-out fold only) and the PQ sweep of an emission rule."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from src.data.split import make_split, assert_disjoint
from src.utils.masks import aggregate_pq, paint_panoptic

BASELINE_PQ = 0.4403668270817509


def encode(mask):
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def pq_rle(predicted, truth):
    """Same greedy strict->0.5 matching as the competition metric, without decoding."""
    matched_p, matched_g, total = set(), set(), 0.
    if predicted and truth:
        ious = np.asarray(mu.iou(predicted, truth, [0] * len(truth)))
        pairs = np.argwhere(ious > .5)
        for p, g in sorted(pairs, key=lambda pair: -ious[pair[0], pair[1]]):
            if p not in matched_p and g not in matched_g:
                matched_p.add(int(p)); matched_g.add(int(g)); total += float(ious[p, g])
    tp = len(matched_p)
    return dict(tp=tp, fp=len(predicted)-tp, fn=len(truth)-tp, iou_sum=total)


def records_for(annotations):
    coco = json.loads(Path(annotations).read_text())
    split = make_split(str(annotations)); assert_disjoint(split)
    record = {r["id"]: r for r in coco["images"]}
    by_id = defaultdict(list)
    val_ids = set(split.val_image_ids)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in val_ids:
            continue
        r = record[annotation["image_id"]]
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mu.merge(mu.frPyObjects(segmentation, r["height"], r["width"]))
        elif isinstance(segmentation["counts"], list):
            rle = mu.frPyObjects(segmentation, r["height"], r["width"])
        else:
            rle = segmentation
        by_id[annotation["image_id"]].append(rle)
    by_photo = defaultdict(list)
    for image_id in split.val_image_ids:
        by_photo[record[image_id]["file_name"]].append((image_id, by_id[image_id]))
    return dict(by_photo), split


def evaluate(cache, records, conf=.35, grow=-1, min_area=300):
    rows = []
    per_record = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for name, passes in records.items():
        candidates = []
        for entry in cache[name]:
            if entry["score"] < conf:
                continue
            mask = mu.decode(entry["rle"])
            if grow < 0:
                mask = cv2.erode(mask, kernel, iterations=-grow)
            elif grow != 0:
                raise ValueError("Only unaltered and eroded masks are screened")
            candidates.append((entry["score"], mask))
        predicted = [encode(m) for _, m, _ in paint_panoptic(candidates, min_area=min_area)]
        for image_id, truth in passes:
            score = pq_rle(predicted, truth)
            rows.append(score)
            per_record[image_id] = dict(score, photograph=name)
    return dict(aggregate_pq(rows), conf=conf, grow=grow, min_area=min_area), per_record


def evaluate_thresholds(cache, records, thresholds, grow=-1, min_area=300):
    """Paint once: every higher confidence threshold selects an identical prefix."""
    thresholds = sorted(set(thresholds))
    totals = {c: [] for c in thresholds}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for name, passes in records.items():
        candidates = []
        for entry in cache[name]:
            if entry['score'] < thresholds[0]:
                continue
            mask = mu.decode(entry['rle'])
            if grow < 0:
                mask = cv2.erode(mask, kernel, iterations=-grow)
            elif grow != 0:
                raise ValueError(grow)
            candidates.append((entry['score'], mask))
        painted = [(score, encode(mask)) for score, mask, _ in
                   paint_panoptic(candidates, min_area=min_area)]
        for conf in thresholds:
            predicted = [rle for score, rle in painted if score >= conf]
            for _, truth in passes:
                totals[conf].append(pq_rle(predicted, truth))
    return [dict(aggregate_pq(totals[c]), conf=c, grow=grow, min_area=min_area)
            for c in thresholds]


