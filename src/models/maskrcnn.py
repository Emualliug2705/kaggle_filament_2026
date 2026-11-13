"""Mask R-CNN R50-FPN v2 (torchvision) with 2-class heads: construction, loading of a trained fold model and its
candidates (masks, scores, logit features) on 2048 px photos, one JSON per photo plus a manifest."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
from pycocotools import mask as mu

from src import paths

H = W = 2048


def build(pretrained=True, mask56=False):
    """Mask R-CNN R50-FPN v2 with 2-class heads; ``pretrained`` starts from torchvision's COCO weights (training).

    mask56: mask RoIAlign on P2 only (stride 4) at 28x28, so the mask head outputs 56x56 masks (same parameters).
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                 min_size=1024, max_size=1024, box_detections_per_img=100, box_score_thresh=0.03)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    if mask56:
        m.roi_heads.mask_roi_pool = MultiScaleRoIAlign(['0'], 28, sampling_ratio=2)
    return m


def _rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {'size': [H, W], 'counts': r['counts'].decode()}


def build_record(masks, sc):
    """masks: iterable of (binary mask HxW, mean probability inside it) per detection in score order; sc: scores.
    Returns (rows, selected) in the candidate-bank JSON layout read by the scorer's evidence features."""
    lg = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))  # noqa: E731
    rows, kept = [], []
    for j, (bm, mp) in enumerate(masks):
        area = int(bm.sum())
        if area < 20:
            continue
        ys, xs = np.nonzero(bm)
        rho = float(np.hypot(xs.mean() - 1024, ys.mean() - 1024) / 965)
        rows.append({'rle': _rle(bm), 'score': float(sc[j]), 'features': [0.0, lg(float(sc[j])), lg(mp)], 'area': area, 'rho': rho})
        kept.append((j, bm))
    # disjoint "selected" set: greedy by score above 0.5, dropping overlaps with pixels already taken
    taken = np.zeros((H, W), bool); sel = []
    for j, bm in kept:
        if sc[j] < 0.5:
            break
        mm = bm & ~taken
        if mm.sum() >= 0.7 * bm.sum() and mm.sum() >= 20:
            taken |= mm; sel.append(_rle(mm))
    return rows, sel


def load_model(name: str, device='cuda'):
    """Trained model ``name`` (for example ``'maskrcnn-f0'``) from models/<name>/model.pt, in eval mode at native 2048 px."""
    model = build(pretrained=False, mask56=False)
    model.load_state_dict(torch.load(paths.MODELS / name / 'model.pt', map_location='cpu'))
    model.transform.min_size, model.transform.max_size = (H,), H  # full native resolution
    return model.to(device).eval()


@torch.no_grad()
def infer(name, images, recs, out, device='cuda', dump=False, log=print):
    """Write the candidate root of fold model ``name`` for the images into folder ``out``.

    images: uint8 (n, 2048, 2048) array or memmap; recs: per image dicts with ``file_name``;
    out: output folder (files overwritten). Returns the manifest dict.
    """
    out = str(out)
    dev = torch.device(device)
    model = load_model(name, dev)
    ids = list(range(len(recs)))
    os.makedirs(out, exist_ok=True)
    if dump:  # raw 28x28 mask probabilities + boxes (before pasting)
        os.makedirs(os.path.join(out, 'dump'), exist_ok=True)
        raw = {}
        # copy at hook time: transform.postprocess later replaces the dict's masks with pasted full-size ones in place
        model.roi_heads.register_forward_hook(lambda mod, inp, o: raw.update(r={k: v.detach().clone() for k, v in o[0][0].items()}))
    man = []; t0 = time.time()
    for n, i in enumerate(ids):
        x = torch.from_numpy(np.asarray(images[i]).astype(np.float32) / 255)[None].expand(3, -1, -1).to(dev)
        o = model([x])[0]  # fp32: under bf16 autocast torchvision decodes boxes on an 8 px grid at 2048 and clips masks
        sc = o['scores'].float().cpu().numpy(); pm = o['masks'][:, 0].float()
        binm = (pm > 0.5).cpu().numpy(); mprob = [float(pm[j][pm[j] > 0.5].mean()) if binm[j].any() else 0.5 for j in range(len(sc))]
        rows, sel = build_record(zip(binm, mprob), sc)
        if dump:
            r = raw['r']
            np.savez_compressed(os.path.join(out, 'dump', f'{n:04d}.npz'), boxes=r['boxes'].float().cpu().numpy(),
                                scores=r['scores'].float().cpu().numpy(), masks=r['masks'][:, 0].half().cpu().numpy())
        fn = f'{n:04d}.json'
        with open(os.path.join(out, fn), 'w', encoding='utf-8') as fh:
            json.dump({'views': {'0.5': rows}, 'predictions': {'set_selector_pq': sel}}, fh)
        man.append({'file': fn, 'file_name': recs[i]['file_name'], 'kind': 'clean'})
        if log and (n % 20 == 0 or n == len(ids) - 1):
            log(f'det {name}: image {n + 1}/{len(ids)} {time.time() - t0:.0f}s')
    manifest = {'records': man, 'model': name, 'split': 'test', 'sec': time.time() - t0, 'device': str(dev),
                'torch': torch.__version__}
    with open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh)
    return manifest
