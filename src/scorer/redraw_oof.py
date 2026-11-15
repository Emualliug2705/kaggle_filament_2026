"""Out-of-fold evaluation of the boundary redraw (CPU, 707 photos): the final scorer's out-of-fold selection, as
emitted and with each boundary redrawn from the U-Net maps of the fold models that did not train on the photo, with the
three rules of src.scorer.redraw: the previous threshold rule (3 px), the same rule on the shifted map (2 px) and the
final rule (shift + area blend, 2 px). Paired bootstraps over photos against the masks as emitted and against the
previous rule. Writes oof_counts_asemitted.npy, oof_counts_threshold3.npy, oof_counts_final.npy and redraw_report.md
in work/scorer/final.

    python -m src.scorer.redraw_oof
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
from src.scorer.redraw import redraw, redraw_blend, shifted  # noqa: E402

MAPS = Path(os.environ.get("FILAMENT_REDRAW_MAPS", str(paths.SCRATCH / "maps")))
ARCHES = tuple(os.environ.get("FILAMENT_REDRAW_ARCHES", "cnxL,r34").split(","))   # maps averaged for the redraw
EMITTED = "as emitted"
PREVIOUS = "threshold rule, 3 px (previous submission)"
SHIFTED = "threshold rule on the map shifted 0.25 px up, 2 px"
FINAL = "final: shift + area blend, 2 px"
RULES = {PREVIOUS: lambda m, U: redraw(m, U, 3), SHIFTED: lambda m, U: redraw(m, shifted(U), 2),
         FINAL: lambda m, U: redraw_blend(m, U, 2)}
FILES = {EMITTED: "oof_counts_asemitted.npy", PREVIOUS: "oof_counts_threshold3.npy", FINAL: "oof_counts_final.npy"}


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

    out = {EMITTED: total(base)}
    for key, rule in RULES.items():
        out[key] = total(paint([rule(m, U) for m in masks]))
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
    for k, f in FILES.items():
        np.save(S.out_dir() / f, T[k])
    sq = lambda t: t[3] / t[0]                                                # noqa: E731
    rq = lambda t: t[0] / (t[0] + t[1] / 2 + t[2] / 2)                        # noqa: E731
    L = ["", "## Full-resolution boundaries (707 out-of-fold photos; each mask keeps its extent, its boundary is "
         "redrawn from the U-Net probability maps)", "",
         "| masks | PQ | vs as emitted [95% interval] | vs previous rule [95% interval] | matched IoU | match rate |",
         "|---|---|---|---|---|---|"]

    def against(k, ref):
        if k == ref or (ref == PREVIOUS and k == EMITTED):
            return "-"
        lo, hi, p = boot(T[k], T[ref])
        return f"{pq(T[k].sum(0)) - pq(T[ref].sum(0)):+.4f} [{lo:+.4f}, {hi:+.4f}], P(better) {p:.2f}"

    for k in keys:
        t = T[k].sum(0)
        L.append(f"| {k} | {pq(t):.4f} | {against(k, EMITTED)} | {against(k, PREVIOUS)} | {sq(t):.4f} | {rq(t):.4f} |")
    folds = [pq(T[FINAL][np.asarray(fold) == k].sum(0)) for k in range(5)]
    L += ["", "final rule, by fold: " + ", ".join(f"{v:.4f}" for v in folds)]
    with open(S.out_dir() / "redraw_report.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
