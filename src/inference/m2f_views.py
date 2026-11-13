"""Raw 8-view candidates (D4: 4 rotations x mirror) of one Mask2Former fold model, kept per view (not fused), so that
the two families can be fused together (src.inference.pools). Same predict() as the caches (score floor 0.02, bf16,
deformable-attention kernel from the hub).

    python -m src.inference.m2f_views FAMILY K SPLIT      FAMILY mask2former|mask2former_limb, SPLIT val|test
    -> work/views/FAMILY/fold{K}_{SPLIT}.json  {photo: [[view, score, rle counts], ...]}
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src import paths

OUT = paths.WORK / "views"
TRAIN, TEST = paths.TRAIN_IMAGES, paths.TEST_IMAGES
PREP = {"mask2former": "raw", "mask2former_limb": "limb3"}


def main():
    import torch
    family, k, split = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    out = OUT / family / f"fold{k}_{split}.json"
    if out.is_file():
        print("done already", out); return
    os.environ["FILAMENT_PREP"] = PREP[family]                 # read when the model package is imported
    from src.models.mask2former import data, evaluate as ev, fast_ops as fo, model as model_mod
    dev = torch.device("cuda")
    fo.install_deformable_kernel("hub")
    model, _ = ev.load_checkpoint(paths.model_dir(family, k) / "model_e050.pt", dev)
    if split == "val":
        names = sorted(json.loads(paths.FOLDS_JSON.read_text())["folds"][k]); images = TRAIN
    else:
        names = sorted(p.name for p in TEST.glob("*.jpeg")); images = TEST

    def enc(preds, kk, mirror, v):
        res = []
        for score, binary in preds:
            r = np.rot90(binary, -kk)
            if mirror:
                r = np.fliplr(r)
            c = model_mod.encode(r)["counts"]
            res.append([v, float(score), c.decode() if isinstance(c, bytes) else c])
        return res
    cache, t0 = {}, time.time()
    with ThreadPoolExecutor(6) as pool, torch.no_grad():
        for i, n in enumerate(names, 1):
            gray = data.read_gray(images / n)
            futs = []
            for v, (kk, mirror) in enumerate(model_mod.D4):
                view = np.ascontiguousarray(np.rot90(np.fliplr(gray) if mirror else gray, kk))
                futs.append(pool.submit(enc, model_mod.predict(model, view, dev, True, 0.02), kk, mirror, v))
            cache[n] = [e for f in futs for e in f.result()]
            if i % 40 == 0:
                print(f"{family} f{k} {split}: {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial"); tmp.write_text(json.dumps(cache)); tmp.replace(out)
    print(f"{family} f{k} {split}: {len(cache)} photos, {sum(map(len, cache.values()))} view candidates, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
