"""Mask R-CNN training: 1024 px crops, COCO start, 12,000 iterations, batch 4, SGD lr 0.01 with warm-up and cosine decay.
The photos whose meta fold equals --fold are held out.

    python -m src.training.train_maskrcnn --name maskrcnn-f0 --fold 0 --cache DIR --runs models --ann ANNOTATIONS"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mu

from src import paths
from src.models.maskrcnn import build

H = W = 2048
CACHE = str(paths.WORK / 'train_cache')


def record_rles(ann_path):
    """{record id: [RLE per instance]} at full resolution."""
    with open(ann_path, encoding='utf-8') as fh:
        ann = json.load(fh)
    out = {}
    for a in ann['annotations']:
        segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
        segs = [s for s in segs if len(s) >= 6]
        if segs:
            out.setdefault(a['image_id'], []).append(mu.merge(mu.frPyObjects(segs, H, W)))
    return out


class Crops(torch.utils.data.Dataset):
    def __init__(self, idx, meta, rles, crop, length, seed, target='random', cache=None):
        self.idx, self.meta, self.rles, self.crop, self.length, self.seed = list(idx), meta, rles, crop, length, seed
        self.target = target  # 'random': one random annotator per sample; 'union': every annotator's instances
        self.cache = cache or CACHE
        self.img = None

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        if self.img is None:
            self.img = np.load(os.path.join(self.cache, 'train_img.npy'), mmap_mode='r')
        rng = np.random.default_rng((self.seed, k))
        i = self.idx[rng.integers(len(self.idx))]
        m = self.meta[i]
        rec = m['records'][rng.integers(len(m['records']))]  # random annotator: scores estimate P(an annotator draws it)
        rl = self.rles.get(rec, []) if self.target == 'random' else [r_ for rr in m['records'] for r_ in self.rles.get(rr, [])]
        C = self.crop
        if rl and rng.random() < 0.7:
            x, y, w, h = mu.toBbox(rl[rng.integers(len(rl))])
            cx, cy = x + w / 2 + rng.normal(0, C * 0.2), y + h / 2 + rng.normal(0, C * 0.2)
        else:
            r = 965 * math.sqrt(rng.random()); t = rng.random() * 2 * math.pi
            cx, cy = 1024 + r * math.cos(t), 1024 + r * math.sin(t)
        x0 = int(np.clip(round(cx - C / 2), 0, W - C)); y0 = int(np.clip(round(cy - C / 2), 0, H - C))
        img = np.asarray(self.img[i, y0:y0 + C, x0:x0 + C]).astype(np.float32) / 255
        masks = []
        for r_ in rl:
            bx, by, bw, bh = mu.toBbox(r_)
            if bx > x0 + C or by > y0 + C or bx + bw < x0 or by + bh < y0:
                continue
            mm = mu.decode(r_)[y0:y0 + C, x0:x0 + C]
            if mm.sum() >= 30:
                masks.append(mm)
        k_rot = int(rng.integers(4)); flip = rng.random() < 0.5
        tf = lambda a: (np.rot90(a, k_rot)[:, ::-1] if flip else np.rot90(a, k_rot))  # noqa: E731
        img = tf(img)
        img = np.clip(img ** rng.uniform(0.8, 1.25) * rng.uniform(0.9, 1.1), 0, 1)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), rng.uniform(0.3, 1.2))
        masks = [np.ascontiguousarray(tf(mm)) for mm in masks]
        boxes = []
        for mm in masks:
            ys, xs = np.nonzero(mm)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        x = torch.from_numpy(np.ascontiguousarray(img))[None].expand(3, -1, -1).contiguous()
        tgt = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
               'labels': torch.ones(len(masks), dtype=torch.int64),
               'masks': torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, C, C), dtype=torch.uint8)}
        return x, tgt


def collate(b):
    return tuple(zip(*b))


def train(a):
    with open(os.path.join(a.cache, 'meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)['train']
    idx = [i for i, m in enumerate(meta) if m['fold'] != 4 and m['fold'] != a.fold]
    out = os.path.join(a.runs, a.name); os.makedirs(out, exist_ok=True)
    rec = {k: v for k, v in vars(a).items() if k not in ('cache', 'runs', 'ann', 'config')}
    with open(os.path.join(out, 'args.json'), 'w', encoding='utf-8') as fh:
        json.dump(rec | {'n_train_images': len(idx)}, fh, indent=1)
    torch.manual_seed(a.seed)
    model = build(pretrained=not a.init, mask56=a.mask56)
    if a.init:  # continue from another run's weights (e.g. det-f0 for a mask-branch fine-tune)
        model.load_state_dict(torch.load(os.path.join(a.runs, a.init, 'model.pt'), map_location='cpu'))
    if a.mask_only:  # train only the mask branch; boxes and scores stay exactly those of the init run
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.startswith(('roi_heads.mask_head.', 'roi_heads.mask_predictor.'))
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 256  # <= 64 positive mask ROIs per image
    model = model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    ds = Crops(idx, meta, record_rles(a.ann), a.crop, a.iters * a.bs, a.seed, a.target, a.cache)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, collate_fn=collate,
                                     persistent_workers=True, prefetch_factor=4)
    log = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8'); t0 = time.time()
    model.train()
    if a.mask_only:  # frozen BatchNorm statistics everywhere outside the mask head
        for n_, mod in model.named_modules():
            if isinstance(mod, torch.nn.BatchNorm2d) and not n_.startswith('roi_heads.mask_head'):
                mod.eval()
    for step, (xs, ts) in enumerate(dl, 1):
        f = step / a.warmup if step < a.warmup else 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / (a.iters - a.warmup)))
        for g in opt.param_groups:
            g['lr'] = a.lr * max(f, 0.01)
        xs = [x.cuda(non_blocking=True) for x in xs]
        ts = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in ts]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model(xs, ts)
        loss = losses['loss_mask'].float() if a.mask_only else sum(v.float() for v in losses.values())
        if not torch.isfinite(loss):
            print('non-finite loss, skipping', step, flush=True); opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step()
        if step % 100 == 0:
            msg = f'step {step} loss {loss.item():.4f} ' + ' '.join(f'{k[5:]}={v.item():.3f}' for k, v in losses.items()) + \
                  f' {(time.time()-t0)/step:.3f}s/it mem {torch.cuda.max_memory_allocated()/2**30:.2f}G'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if step % 3000 == 0 or step == a.iters:
            torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    log.close()
    with open(os.path.join(out, 'done.json'), 'w', encoding='utf-8') as fh:
        json.dump({'done': True, 'sec': time.time() - t0}, fh)


def parse(argv=None):
    t = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    t.add_argument('--config', default='', help='read every argument from configs/<config>.json (e.g. det-f0)')
    t.add_argument('--name'); t.add_argument('--fold', type=int)
    t.add_argument('--iters', type=int, default=12000); t.add_argument('--bs', type=int, default=4)
    t.add_argument('--crop', type=int, default=1024); t.add_argument('--lr', type=float, default=0.01)
    t.add_argument('--warmup', type=int, default=500); t.add_argument('--workers', type=int, default=5)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--init', default=''); t.add_argument('--mask-only', action='store_true')
    t.add_argument('--mask56', action='store_true'); t.add_argument('--target', default='random', choices=['random', 'union'])
    t.add_argument('--cache', default=CACHE); t.add_argument('--runs', default=str(paths.WORK / 'runs'))
    t.add_argument('--ann', default=str(paths.TRAIN_ANNOTATIONS))
    a = t.parse_args(argv)
    if a.config:
        with open(paths.CONFIGS / f'{a.config}.json', encoding='utf-8') as fh:
            rec = json.load(fh)
        for k in ('name', 'fold', 'iters', 'bs', 'crop', 'lr', 'warmup', 'workers', 'seed', 'init', 'mask_only', 'mask56', 'target'):
            if k in rec:
                setattr(a, k, rec[k])
    if a.name is None or a.fold is None:
        t.error('--name and --fold (or --config) are required')
    a.cmd = 'train'
    return a


if __name__ == '__main__':
    train(parse())
