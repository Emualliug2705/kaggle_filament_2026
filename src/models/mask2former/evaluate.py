"""Mask2Former validation: candidate caches of a checkpoint on the held-out photos (single view or 8 views) and the PQ
sweep of the emission rule over them, written next to the checkpoint during training."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.evaluation.sweep import evaluate_thresholds, records_for
from src.utils.masks import paint_panoptic
from src.models.mask2former.data import read_gray
from src.models.mask2former.model import build_model, predict_photograph

BASELINE_PQ = 0.4558638687292126          # parity check: PQ of a stored reference pool at conf 0.35, erosion 1, area 300
REFERENCE_PQ = 0.46814553                 # an earlier best validation PQ, reported next to each sweep
CONFS = tuple(round(0.05 * i, 2) for i in range(1, 20))
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]
PRED_SIZE = 256                          # side of the prediction masks written for WeightsLab


def emitted_masks(cache: dict, names: list[str], conf: float, grow: int, min_area: int) -> np.ndarray:
    """(N, PRED_SIZE, PRED_SIZE) uint8: the filaments the rule emits for each photo (1) over background (0).

    Same emission as evaluate_thresholds: candidates at or above conf, eroded when grow < 0, painted without overlap,
    pieces under min_area dropped. Reduced to PRED_SIZE by block maximum, so thin filaments stay visible."""
    import cv2
    from pycocotools import mask as mu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = np.zeros((len(names), PRED_SIZE, PRED_SIZE), np.uint8)
    for i, name in enumerate(names):
        candidates = []
        for entry in cache[name]:
            if entry["score"] < conf:
                continue
            mask = mu.decode(entry["rle"])
            if grow < 0:
                mask = cv2.erode(mask, kernel, iterations=-grow)
            candidates.append((entry["score"], mask))
        union = None
        for _, mask, _ in paint_panoptic(candidates, min_area=min_area):
            union = mask if union is None else union | mask
        if union is not None:
            f = union.shape[0] // PRED_SIZE
            out[i] = union[:f * PRED_SIZE, :f * PRED_SIZE].reshape(PRED_SIZE, f, PRED_SIZE, f).max((1, 3))
    return out


def event(tag: str, log: Path | None = None, **payload) -> None:
    line = f"{tag} {json.dumps(payload, sort_keys=True, default=str)}"
    print(line, flush=True)
    if log is not None:
        with Path(log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config=state["config"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state


def cache_validation(model, records: dict, images: Path, device: torch.device, bf16: bool,
                     d4: bool = False, limit: int = 0, floor: float = 0.02) -> tuple[dict, float]:
    names = sorted(records)[:limit] if limit else sorted(records)
    started = time.time()
    cache = {}
    for name in names:
        cache[name] = predict_photograph(model, read_gray(images / name), device, bf16, d4=d4, floor=floor)
    return cache, time.time() - started


def _sweep_one(job: tuple) -> list[dict]:
    cache_path, annotations, confs, grow, area, names = job
    records, _ = records_for(annotations)
    if names:
        records = {name: records[name] for name in names}
    cache = json.loads(Path(cache_path).read_text())
    return evaluate_thresholds(cache, records, confs, grow=grow, min_area=area)


def sweep(cache_path: Path, annotations: Path, processes: int = 4, confs=CONFS,
          names: list[str] | None = None) -> list[dict]:
    jobs = [(str(cache_path), str(annotations), tuple(confs), grow, area, names)
            for grow, area in EMISSION]
    with ProcessPoolExecutor(max_workers=max(1, min(processes, len(jobs)))) as pool:
        rows = [row for part in pool.map(_sweep_one, jobs) for row in part]
    return sorted(rows, key=lambda row: -row["pq"])


def parity(pool_path: Path, annotations: Path) -> dict:
    records, _ = records_for(annotations)
    pool = json.loads(Path(pool_path).read_text())
    row = evaluate_thresholds(pool, records, (0.35,), grow=-1, min_area=300)[0]
    if abs(row["pq"] - BASELINE_PQ) > 1e-9:
        raise SystemExit(f"evaluator does not reproduce the reference pool's PQ: {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None, help="sweep an existing cache")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--d4", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="pytorch")
    parser.add_argument("--msda-revision", default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N photographs, smoke tests")
    parser.add_argument("--parity-pool", type=Path, default=None)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.parity_pool is not None:
        row = parity(args.parity_pool, args.annotations)
        event("PARITY", args.log, pq=row["pq"], tp=row["tp"], fp=row["fp"], fn=row["fn"])
        if args.cache is None and args.checkpoint is None:
            return

    records, split = records_for(args.annotations)
    names = sorted(records)[:args.limit] if args.limit else None
    cache_path = args.cache
    report = dict(tag=args.tag, split=split.summary(), d4=args.d4)
    if cache_path is None:
        if args.checkpoint is None or args.images is None:
            raise SystemExit("give --cache, or --checkpoint with --images")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.msda != "pytorch":
            from src.models.mask2former.fast_ops import (install_deformable_kernel,
                                                                      verify_deformable_kernel)
            kernel = install_deformable_kernel(args.msda, args.msda_revision)
            rows = verify_deformable_kernel(device)
            event("KERNEL", args.log, **kernel, verification=rows)
            if not all(row["passed"] for row in rows):
                raise SystemExit("deformable attention kernel differs from the reference")
        model, state = load_checkpoint(args.checkpoint, device)
        cache, seconds = cache_validation(model, records, args.images, device, args.bf16,
                                          args.d4, args.limit)
        cache_path = args.output / f"{args.tag}_cache.json"
        cache_path.write_text(json.dumps(cache))
        report.update(checkpoint=str(args.checkpoint), epoch=state.get("epoch"),
                      step=state.get("step"), inference_seconds=round(seconds, 1),
                      candidates=sum(len(entries) for entries in cache.values()))
    started = time.time()
    rows = sweep(cache_path, args.annotations, args.processes, names=names)
    report.update(sweep_seconds=round(time.time() - started, 1), best=rows[0], rows=rows,
                  photographs=len(names) if names else len(records),
                  reference=dict(earlier_best=REFERENCE_PQ))
    # Per-photo counts of the best rule (read by the trainer for WeightsLab). Written before the sweep report, whose
    # appearance tells the trainer the evaluation is complete.
    best = rows[0]
    cache = json.loads(Path(cache_path).read_text())
    photos = {name: evaluate_thresholds({name: cache[name]}, {name: records[name]}, (best["conf"],),
                                        grow=best["grow"], min_area=best["min_area"])[0]
              for name in (names or sorted(records))}
    (args.output / f"{args.tag}_photos.json").write_text(json.dumps(photos))
    order = names or sorted(records)
    np.savez_compressed(args.output / f"{args.tag}_pred{PRED_SIZE}.npz", names=np.array(order),
                        masks=emitted_masks(cache, order, best["conf"], best["grow"], best["min_area"]))
    (args.output / f"{args.tag}_sweep.json").write_text(json.dumps(report, indent=2))
    event("EVAL", args.log, run=args.tag, d4=args.d4, best=rows[0],
          photographs=report["photographs"], sweep_seconds=report["sweep_seconds"])


if __name__ == "__main__":
    main()
