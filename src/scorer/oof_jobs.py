"""Out-of-fold jobs of the final scorer (photo records, candidates, q, overlaps, tau, theta), the selected masks of one
photo, and the paired bootstrap of a PQ difference over photos."""
import json
import os

import numpy as np

from src.evaluation.oof import ANN, native, pq
from src.scorer import selector as S


def base_masks(entries, q, ovl, tau, theta):
    from pycocotools import mask as mu
    from src.models.mask2former.submit import emit
    keep = S.choose(q, ovl, tau, theta)
    base = [native(r) for r in emit([dict(score=float(q[i]) + 1.0, rle=native(entries[i]["rle"])) for i in keep], 0.0, 0, 100)]
    return base, [mu.decode(b) for b in base]


def load_jobs():
    from src.evaluation.sweep import records_for
    z = np.load(S.OUT / "oof_rows.npz", allow_pickle=True)
    q = np.load(S.out_dir() / "oof_q.npy")
    ovl = np.load(S.OUT / "oof_ovl.npy", allow_pickle=True)
    rep = json.loads((S.out_dir() / "oof_report.json").read_text())
    tau, theta = rep["best"]["tau"], rep["best"]["theta"]
    names, photo, fold = list(z["names"]), z["photo"], z["fold"]
    off = np.concatenate([[0], np.cumsum(np.bincount(photo, minlength=len(names)))])
    jobs = {}
    for k in range(5):
        os.environ["FILAMENT_VAL_FOLD"] = str(k)
        records, _ = records_for(ANN)
        d4 = json.loads((S.d4_path(k, "val")).read_text())
        for i, n in enumerate(names):
            if fold[i] == k:
                jobs[i] = ([r for _, r in records[n]], d4[n], q[off[i]:off[i + 1]], ovl[i], tau, theta)
    return names, fold, [jobs[i] for i in range(len(names))]


def boot(A, B, n=4000):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(A), (n, len(A)))
    d = np.array([pq(A[i].sum(0)) - pq(B[i].sum(0)) for i in idx])
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()
