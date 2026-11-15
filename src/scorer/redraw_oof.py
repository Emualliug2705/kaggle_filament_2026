"""Out-of-fold evaluation of the boundary redraw (CPU, 707 photos): the final scorer's out-of-fold selection, as
emitted and with each boundary redrawn (src.scorer.redraw) from the U-Net maps of the fold models that did not train on
the photo; paired bootstrap against the masks as emitted. Writes oof_counts_asemitted.npy, oof_counts_final.npy and
redraw_report.md in work/scorer/final.

    python -m src.scorer.redraw_oof        (radius: FILAMENT_REDRAW_RADIUS, default 2)
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("FILAMENT_CPUS", "8")
import numpy as np  # noqa: E402

from src import paths  # noqa: E402
from src.evaluation.oof import counts, native, pq  # noqa: E402
from src.scorer import selector as S  # noqa: E402
from src.scorer.oof_jobs import base_masks, boot, load_jobs  # noqa: E402
from src.scorer.paint import paint  # noqa: E402
from src.scorer.redraw import redraw  # noqa: E402

MAPS = Path(os.environ.get("FILAMENT_REDRAW_MAPS", str(paths.SCRATCH / "maps")))
ARCHES = tuple(os.environ.get("FILAMENT_REDRAW_ARCHES", "cnxL,r34").split(","))   # maps averaged for the redraw
RADIUS = int(os.environ.get("FILAMENT_REDRAW_RADIUS", "2"))
EMITTED, FINAL = "as emitted", f"boundary redrawn within {RADIUS} px (0.25 px shift, area blend)"


def photo(job):
    records, entries, q, ovl, tau, theta, upaths, j = job
    gts = [[native(r) for r in rec] for rec in records]
    base, masks = base_masks(entries, q, ovl, tau, theta)
    U = np.mean([np.asarray(np.load(p, mmap_mode="r")[j, 0]).astype(np.float32) for p in upaths], 0)

    def total(pred):
        t = np.zeros(4)
        for g in gts:
            t += counts(pred, g)[0]
        return t

    return {EMITTED: total(base), FINAL: total(paint([redraw(m, U, RADIUS) for m in masks]))}


def main():
    names, fold, jobs = load_jobs()
    full = []
    for i, job in enumerate(jobs):
        k = int(fold[i])
        up = [str(MAPS / f"train_f{k}" / f"test{a}.npy") for a in ARCHES]
        files = json.loads(Path(up[0] + ".json").read_text())["files"]
        full.append(job + (up, files.index(names[i])))
    with ProcessPoolExecutor(S.PROCS) as ex:
        res = list(ex.map(photo, full, chunksize=2))
    T = {k: np.array([r[k] for r in res]) for k in (EMITTED, FINAL)}
    np.save(S.out_dir() / "oof_counts_asemitted.npy", T[EMITTED])
    np.save(S.out_dir() / "oof_counts_final.npy", T[FINAL])
    sq = lambda t: t[3] / t[0]                                                # noqa: E731
    rq = lambda t: t[0] / (t[0] + t[1] / 2 + t[2] / 2)                        # noqa: E731
    B = T[EMITTED]
    L = ["", "## Full-resolution boundaries (707 out-of-fold photos; each mask keeps its extent, its boundary is "
         "redrawn from the U-Net probability maps)", "",
         "| masks | PQ | change | 95% interval | P(better) | matched IoU | match rate |", "|---|---|---|---|---|---|---|"]
    t = B.sum(0)
    L.append(f"| {EMITTED} | {pq(t):.4f} | - | - | - | {sq(t):.4f} | {rq(t):.4f} |")
    t = T[FINAL].sum(0); lo, hi, p = boot(T[FINAL], B)
    L.append(f"| {FINAL} | {pq(t):.4f} | {pq(t) - pq(B.sum(0)):+.4f} | {lo:+.4f} to {hi:+.4f} | {p:.2f} | {sq(t):.4f} | {rq(t):.4f} |")
    folds = [pq(T[FINAL][np.asarray(fold) == k].sum(0)) for k in range(5)]
    L += ["", "by fold: " + ", ".join(f"{v:.4f}" for v in folds)]
    with open(S.out_dir() / "redraw_report.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
