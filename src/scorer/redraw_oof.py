"""Out-of-fold evaluation of the boundary redraw (CPU, 707 photos): the final scorer's out-of-fold selection, as
emitted and with each boundary redrawn within r px from the U-Net maps of the fold models that did not train on the
photo; paired bootstrap against the masks as emitted. Writes oof_counts_*.npy and redraw_report.md in work/scorer/final.

    python -m src.scorer.redraw_oof        (radii: FILAMENT_REDRAW_RADII, default 3)
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
RADII = tuple(int(r) for r in os.environ.get("FILAMENT_REDRAW_RADII", "3").split(","))


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

    out = {"as emitted": total(base)}
    for r in RADII:
        out[f"boundary within {r} px from the full-resolution U-Nets"] = total(paint([redraw(m, U, r) for m in masks]))
    return out


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
    keys = list(res[0])
    T = {k: np.array([r[k] for r in res]) for k in keys}
    for r_ in RADII:
        k_ = f"boundary within {r_} px from the full-resolution U-Nets"
        if k_ in T:
            np.save(S.out_dir() / f"oof_counts_fullres{r_}.npy", T[k_])
    np.save(S.out_dir() / "oof_counts_asemitted.npy", T["as emitted"])
    sq = lambda t: t[3] / t[0]                                                # noqa: E731
    rq = lambda t: t[0] / (t[0] + t[1] / 2 + t[2] / 2)                        # noqa: E731
    B = T["as emitted"]
    L = ["", "## Full-resolution boundaries (707 out-of-fold photos; each mask keeps its extent, its boundary is "
         "redrawn from the U-Net probability maps)", "",
         "| masks | PQ | change | 95% interval | P(better) | matched IoU | match rate |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        t = T[k].sum(0)
        if k == "as emitted":
            L.append(f"| {k} | {pq(t):.4f} | - | - | - | {sq(t):.4f} | {rq(t):.4f} |")
        else:
            lo, hi, p = boot(T[k], B)
            L.append(f"| {k} | {pq(t):.4f} | {pq(t) - pq(B.sum(0)):+.4f} | {lo:+.4f} to {hi:+.4f} | {p:.2f} | {sq(t):.4f} | {rq(t):.4f} |")
    with open(S.out_dir() / "redraw_report.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
