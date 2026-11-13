"""Mask2Former candidate caches: {score, rle} per photo (score floor 0.02) of one checkpoint on the test photos or on one
fold's held-out photos, single view or 8 views (D4: 4 rotations x mirror, fused), sharded over processes sharing a GPU.
Also fuse_n, the view / model fusion used for the two-family pools.

    python -m src.inference.m2f_caches --checkpoint models/mask2former-f0/model_e050.pt --split test|val0 [--d4] [--bf16]
        --workers 1 --threads 16 --out work/mask2former/fold0_test_d4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src import paths

ANNOTATIONS = str(paths.ANNOTATIONS)
TRAIN_IMAGES = paths.TRAIN_IMAGES
TEST_IMAGES = paths.TEST_IMAGES
os.environ.setdefault("FILAMENT_FOLDS_JSON", str(paths.FOLDS_JSON))


def val_records(fold: int) -> dict:
    from src.evaluation.sweep import records_for
    os.environ["FILAMENT_VAL_FOLD"] = str(fold)
    records, _ = records_for(ANNOTATIONS)
    return records


# ----------------------------------------------------------------------------------------------- infer
def _encode_view(predictions: list, k: int, mirror: bool) -> list:
    """Restoration of one view to the native frame and RLE encoding (runs in a CPU thread)."""
    import numpy as np
    from src.models.mask2former.model import encode
    entries = []
    for score, binary in predictions:
        restored = np.rot90(binary, -k)
        if mirror:
            restored = np.fliplr(restored)
        entries.append((score, encode(restored)))
    return entries


def _finish_photo(view_futures: list, d4: bool) -> list[dict]:
    from src.inference.tta import fuse
    entries = [entry for future in view_futures for entry in future.result()]
    if not d4:
        return [dict(score=score, rle=rle) for score, rle in entries]
    return fuse(entries, 0.5, 0.5)


def _infer_shard(job: tuple) -> dict:
    """model.predict_photograph, pipelined: the GPU predicts the next view while CPU threads restore and
    encode the previous ones (same entries, same order)."""
    checkpoint, images, names, d4, bf16, threads, worker = job
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    from src.models.mask2former.data import NATIVE, read_gray
    from src.models.mask2former.evaluate import load_checkpoint
    from src.models.mask2former.fast_ops import install_deformable_kernel, verify_deformable_kernel
    from src.models.mask2former.model import D4, predict
    device = torch.device("cuda")
    install_deformable_kernel("hub")
    if not all(row["passed"] for row in verify_deformable_kernel(device)):
        install_deformable_kernel("pytorch")
        print("KERNEL_FALLBACK pytorch", flush=True)
    model, _ = load_checkpoint(Path(checkpoint), device)
    views = D4 if d4 else [(0, False)]
    out, pending, started = {}, [], time.time()
    with ThreadPoolExecutor(max_workers=threads) as encoders:
        for i, name in enumerate(names, start=1):
            gray = read_gray(Path(images) / name)
            if gray.shape != (NATIVE, NATIVE):
                raise ValueError(f"expected a {NATIVE}x{NATIVE} photograph, got {gray.shape}")
            futures = []
            for k, mirror in views:
                view = np.ascontiguousarray(np.rot90(np.fliplr(gray) if mirror else gray, k)) if d4 else gray
                futures.append(encoders.submit(_encode_view, predict(model, view, device, bf16, 0.02), k, mirror))
            pending.append((name, futures))
            while len(pending) > 2:                     # bounded lag: dense masks wait in memory
                done_name, done = pending.pop(0)
                out[done_name] = _finish_photo(done, d4)
            if worker == 0 and i % 10 == 0:
                print(f"  worker 0: {i}/{len(names)} {time.time() - started:.0f}s", flush=True)
        for done_name, done in pending:
            out[done_name] = _finish_photo(done, d4)
    return out


def infer(args) -> None:
    import multiprocessing as mp
    if args.split == "test":
        images, names = TEST_IMAGES, sorted(p.name for p in TEST_IMAGES.glob("*.jpeg"))
        if len(names) != 180:
            raise SystemExit(f"expected 180 test photographs, found {len(names)}")
    else:
        images, names = TRAIN_IMAGES, sorted(val_records(int(args.split.removeprefix("val"))))
    if args.limit:
        names = names[:args.limit]
    jobs = [(args.checkpoint, str(images), names[w::args.workers], args.d4, args.bf16, args.threads, w)
            for w in range(args.workers)]
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_infer_shard, jobs)
    cache = {name: part[name] for part in parts for name in part}
    cache = {name: cache[name] for name in sorted(cache)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cache))
    print(json.dumps(dict(event="INFER_DONE", checkpoint=args.checkpoint, split=args.split, d4=args.d4,
                          bf16=args.bf16, photographs=len(cache), candidates=sum(map(len, cache.values())),
                          workers=args.workers, seconds=round(time.time() - started, 1), out=args.out)), flush=True)


# ----------------------------------------------------------------------------------------------- fuse
def fuse_n(entries: list, link: float, vote: float, n: int) -> list[dict]:
    """src.inference.tta.fuse, operation for operation, with the agreement normaliser n as a parameter: candidates
    linked by IoU >= link, pixel vote, score = mean member score x (0.5 + 0.5 x members / n)."""
    import numpy as np
    from pycocotools import mask as mu
    from src.inference.tta import vote_rle
    order = sorted(range(len(entries)), key=lambda i: -entries[i][0])
    if not order:
        return []
    ious = mu.iou([e[1] for e in entries], [e[1] for e in entries], [0] * len(entries))
    used, output = set(), []
    for seed in order:
        if seed in used:
            continue
        members = [seed]
        used.add(seed)
        for i in order:
            if i not in used and ious[seed, i] >= link:
                members.append(i)
                used.add(i)
        binary = vote_rle([entries[i][1] for i in members], max(1, round(vote * len(members))))
        if mu.area(binary) < 40:
            continue
        score = float(np.mean([entries[i][0] for i in members]))
        output.append(dict(score=score * (.5 + .5 * len(members) / n), rle=binary))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, help="test, or val<k> for fold k's held-out photographs")
    parser.add_argument("--d4", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--workers", type=int, default=2, help="processes sharing the visible GPU")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads per process")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    infer(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
