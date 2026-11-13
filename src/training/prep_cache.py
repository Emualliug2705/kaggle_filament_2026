"""Training cache of the U-Nets and Mask R-CNN: the training photos as uint8 arrays, soft filament targets (mean over the
annotators), spine targets, crop centres and a meta.json with each photo's fold and disk statistics.

    python -m src.training.prep_cache --cache DIR --data DATA --split configs/base_split.json --workers N"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from src import paths
from src.data.photos import disk_stats

H = W = 2048


def record_masks(anns):
    """Union mask and spine raster for one annotator record."""
    full = np.zeros((H, W), np.uint8)
    spine = np.zeros((H, W), np.uint8)
    for a in anns:
        segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
        rles = mu.frPyObjects([s for s in segs if len(s) >= 6], H, W)
        full |= mu.decode(mu.merge(rles)).astype(np.uint8)
        sp = json.loads(a['spine']) if isinstance(a['spine'], str) else a['spine']
        if sp and len(sp) >= 4:
            pts = np.round(np.asarray(sp, np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(spine, [pts], False, 1, thickness=3)
    return full, spine


def build_one(args):
    path, recs = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img.shape == (H, W), path
    soft = np.zeros((H, W), np.float32)
    sp = np.zeros((H, W), np.float32)
    for anns in recs:
        f, s = record_masks(anns)
        soft += f
        sp += s
    n = max(len(recs), 1)
    return img, np.round(soft / n * 255).astype(np.uint8), np.round(sp / n * 255).astype(np.uint8), disk_stats(img)


def main(cache, data, split_path, workers=8):
    with open(data / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json', encoding='utf-8') as fh:
        ann = json.load(fh)
    with open(split_path, encoding='utf-8') as fh:
        split = json.load(fh)
    by_rec = {}
    for a in ann['annotations']:
        by_rec.setdefault(a['image_id'], []).append(a)
    obs = sorted(split['observations'], key=lambda o: o['file_name'])
    os.makedirs(cache, exist_ok=True)
    N = len(obs)
    mm = {k: np.lib.format.open_memmap(os.path.join(cache, f'train_{k}.npy'), 'w+', np.uint8, (N, H, W))
          for k in ('img', 'soft', 'spine')}
    jobs = [(os.path.join(data, 'train', 'train_images', o['file_name']), [by_rec[r] for r in o['annotator_image_ids']])
            for o in obs]
    stats = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (img, soft, sp, st) in enumerate(ex.map(build_one, jobs, chunksize=4)):
            mm['img'][i], mm['soft'][i], mm['spine'][i] = img, soft, sp
            stats.append(st)
    for m in mm.values():
        m.flush()
    meta = {'train': [dict(file_name=o['file_name'], fold=o['fold'], group=o['group_id'], records=o['annotator_image_ids'],
                           disk_mean=s[0], disk_std=s[1], disk_thr=s[2]) for o, s in zip(obs, stats)]}
    with open(os.path.join(cache, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    centers = {}
    for i, o in enumerate(obs):
        cs = []
        for r in o['annotator_image_ids']:
            for a in by_rec[r]:
                x, y, w, h = json.loads(a['bbox']) if isinstance(a['bbox'], str) else a['bbox']
                cs.append((x + w / 2, y + h / 2))
        centers[str(i)] = cs
    with open(os.path.join(cache, 'centers.json'), 'w', encoding='utf-8') as fh:
        json.dump(centers, fh)
    print('train', N)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=paths.WORK / 'train_cache')
    p.add_argument('--data', type=Path, default=paths.DATA)
    p.add_argument('--split', type=Path, default=paths.CONFIGS / 'base_split.json')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    main(str(a.cache), a.data, a.split, a.workers)
