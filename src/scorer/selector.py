"""Learned candidate scorer, exact selection and the 5-fold test fusion (CPU only).

1. Candidates: the two-family pool of a fold on a photo (work/pools), 30 numbers each: the fused score, agreement with
   the single-view pool, shape, position, image contrast, competitors (overlapping candidates and their scores).
   Evidence blocks add 46 more: Mask R-CNN agreement (maskrcnn, 5), each U-Net's spine map (spine_cnx / spine_cnxL /
   spine_r34, 12 each) and the ROI detector + refiner agreement (roi, 5).
2. Target: the mean over the photo's annotator records of the candidate's IoU with the filament it overlaps most,
   counted only above 0.5. LightGBM regression (5 seeds), cross-fitted over the user's 5 folds: a photo is scored by a
   scorer that never saw its fold, on candidates from models that never saw it.
3. Selection: the non-overlapping set maximising sum(q - tau) (exact, src.scorer.select), then the emission (painting
   in descending q, minimum area 100). tau and the overlap tolerance theta are chosen on out-of-fold PQ.
4. Test: the scorer trained on all out-of-fold rows scores each fold model's test candidates; the 5 fold models'
   candidates are fused (IoU link 0.5, pixel vote 0.5, q = mean of each model's best member) and selected with the
   out-of-fold tau and theta.

    python -m src.scorer.selector features | evidence | roi_evidence | fit | test | ensemble   (in this order)
    -> work/scorer/ (tables) and work/scorer/final/ (out-of-fold report, test files)
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from src import paths  # noqa: E402
from src.evaluation.oof import ANN, counts, native, pq  # noqa: E402

os.environ.setdefault("FILAMENT_FOLDS_JSON", str(paths.FOLDS_JSON))

OUT = paths.WORK / "scorer"
POOLS = os.environ.get("FILAMENT_POOLS", "pools/fold{k}_{split}.json")   # candidate pools, relative to work/


def d4_path(k, split):
    return paths.WORK / POOLS.format(k=k, split=split)


def sv_path(k, split):
    """Single-view candidates of mask2former fold k: its held-out photos (written by training) or the test photos."""
    return paths.model_dir("mask2former", k) / "e050_cache.json" if split == "val" else paths.WORK / "mask2former" / f"fold{k}_test_single.json"


PROCS = int(os.environ.get("FILAMENT_CPUS", "24"))
TRAIN_IMAGES = paths.TRAIN_IMAGES
TEST_IMAGES = paths.TEST_IMAGES
H = W = 2048
FEATURES = ["s_d4", "s_sv", "iou_sv", "n_sv50", "s_sv_max50", "ratio", "log_area", "bw", "bh", "aspect", "fill",
            "major", "minor", "elong", "radius", "i_mean", "i_std", "ring_contrast", "rank", "n_cand", "n_strong",
            "ovl_hi", "s_hi", "n_ovl", "s_ovl_sum", "iou_max", "s_iou_max", "s_rel", "area_rank", "sv_area_ratio"]
SEEDS = [10, 11, 12, 13, 14]
LGBM = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)
ROUNDS = 300


def disk_stats(img):
    """Mean and standard deviation of the pixels inside the solar disk (radius 900 px around the centre)."""
    yy, xx = np.ogrid[:H, :W]
    disk = (yy - 1024) ** 2 + (xx - 1024) ** 2 < 900 ** 2
    v = img[disk].astype(np.float32)
    return float(v.mean()), float(v.std()) + 1e-6


def areas(rles) -> np.ndarray:
    """pycocotools mask.area in chunks (with numpy 2 it fails on more than 255 masks at once)."""
    from pycocotools import mask as mu
    out = [np.asarray(mu.area(rles[i:i + 200]), np.float64).reshape(-1) for i in range(0, len(rles), 200)]
    return np.concatenate(out) if out else np.zeros(0)


def describe(d4, sv, img):
    """Feature matrix [n, len(FEATURES)] of the D4 candidates of one photo; also their overlap-fraction matrix."""
    import cv2
    from pycocotools import mask as mu
    n = len(d4)
    X = np.zeros((n, len(FEATURES)), np.float32)
    if not n:
        return X, np.zeros((0, 0), np.float32)
    cand = [native(e["rle"]) for e in d4]
    s = np.array([e["score"] for e in d4], np.float32)
    svr = [native(e["rle"]) for e in sv]
    ss = np.array([e["score"] for e in sv], np.float32)
    area = areas(cand)
    if len(sv):
        m = np.asarray(mu.iou(cand, svr, [0] * len(svr)))            # n x n_sv
        b = m.argmax(1)
        X[:, 1] = ss[b]; X[:, 2] = m.max(1); X[:, 3] = (m > 0.5).sum(1)
        X[:, 4] = np.where(m > 0.5, ss[None], 0).max(1)
        X[:, 29] = areas(svr)[b] / area
    X[:, 0] = s
    X[:, 5] = s / np.maximum(X[:, 1], 1e-3)
    mean, std = disk_stats(img)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    for i, c in enumerate(cand):
        x, y, w, h = (int(v) for v in mu.toBbox(c))
        pad = 8
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
        mask = mu.decode(c)[y0:y1, x0:x1].astype(bool)
        ys, xs = np.nonzero(mask)
        if len(ys) > 2:
            ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64))))[::-1]
        else:
            ev = np.array([1.0, 1.0])
        major, minor = 4 * np.sqrt(max(ev[0], 1e-6)), 4 * np.sqrt(max(ev[1], 1e-6))
        crop = (img[y0:y1, x0:x1].astype(np.float32) - mean) / std
        ring = cv2.dilate(mask.astype(np.uint8), ring_k).astype(bool) & ~mask
        inside = crop[mask]
        X[i, 6:18] = [np.log(max(area[i], 1)), w, h, max(w, h) / max(min(w, h), 1), area[i] / max(w * h, 1), major,
                      minor, major / max(minor, 1e-3), np.hypot(xs.mean() + x0 - 1024, ys.mean() + y0 - 1024) / 965
                      if len(ys) else 0.0, inside.mean() if inside.size else 0.0, inside.std() if inside.size else 0.0,
                      (crop[ring].mean() - inside.mean()) if ring.any() and inside.size else 0.0]
    order = np.argsort(-s)
    rank = np.empty(n); rank[order] = np.arange(n) / max(n - 1, 1)
    X[:, 18] = rank; X[:, 19] = n; X[:, 20] = (s > 0.5).sum()
    frac = np.asarray(mu.iou(cand, cand, [1] * n))                    # [i, j] = intersection / area(i)
    np.fill_diagonal(frac, 0)
    ovl = np.maximum(frac, frac.T)                                    # relative to the smaller of the two
    iou = np.asarray(mu.iou(cand, cand, [0] * n)); np.fill_diagonal(iou, 0)
    higher = s[None] > s[:, None]                                     # [i, j]: j scores higher than i
    oh = np.where(higher, ovl, 0)
    X[:, 21] = oh.max(1); X[:, 22] = s[oh.argmax(1)] * (oh.max(1) > 0)
    X[:, 23] = (ovl > 0.1).sum(1); X[:, 24] = (np.where(ovl > 0.1, s[None], 0)).sum(1)
    X[:, 25] = iou.max(1); X[:, 26] = s[iou.argmax(1)] * (iou.max(1) > 0)
    X[:, 27] = s / max(float(s.max()), 1e-3)
    ar = np.argsort(-area); arank = np.empty(n); arank[ar] = np.arange(n) / max(n - 1, 1)
    X[:, 28] = arank
    return X, ovl.astype(np.float32)


def target(cand, records):
    """Mean over the records of the IoU with the best-overlapping filament, counted only above 0.5."""
    from pycocotools import mask as mu
    y = np.zeros(len(cand))
    for g in records:
        if g and cand:
            m = np.asarray(mu.iou(cand, g, [0] * len(g)))            # n x n_gt
            b = m.max(1)
            y += np.where(b > 0.5, b, 0.0)
    return y / max(len(records), 1)


def objarr(xs):
    a = np.empty(len(xs), object)
    for i, x in enumerate(xs):
        a[i] = x
    return a


def photo_features(job):
    import cv2
    name, image_dir, d4, sv, records = job
    img = cv2.imread(str(Path(image_dir) / name), cv2.IMREAD_GRAYSCALE)
    X, ovl = describe(d4, sv, img)
    y = None
    if records is not None:
        y = target([native(e["rle"]) for e in d4], [[native(r) for r in rec] for rec in records])
    return name, X, ovl, y


def build_features():
    from src.evaluation.sweep import records_for
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, meta = [], []
    for k in range(5):
        os.environ["FILAMENT_VAL_FOLD"] = str(k)
        records, _ = records_for(ANN)
        d4 = json.loads((d4_path(k, "val")).read_text())
        sv = json.loads(sv_path(k, "val").read_text())
        for name, pairs in sorted(records.items()):
            jobs.append((name, str(TRAIN_IMAGES), d4[name], sv[name], [r for _, r in pairs]))
            meta.append((name, k, len(pairs)))
    with ProcessPoolExecutor(PROCS) as ex:
        res = list(ex.map(photo_features, jobs, chunksize=2))
    np.savez(OUT / "oof_rows.npz", X=np.concatenate([r[1] for r in res]), y=np.concatenate([r[3] for r in res]),
             photo=np.concatenate([np.full(len(r[1]), i, np.int32) for i, r in enumerate(res)]),
             names=np.array([m[0] for m in meta]), fold=np.array([m[1] for m in meta]),
             n_records=np.array([m[2] for m in meta]))
    np.save(OUT / "oof_ovl.npy", objarr([r[2] for r in res]), allow_pickle=True)
    print(f"OOF rows: {sum(len(r[1]) for r in res)} candidates on {len(res)} photos", flush=True)
    for k in range(5):
        d4 = json.loads((d4_path(k, "test")).read_text())
        sv = json.loads(sv_path(k, "test").read_text())
        names = sorted(d4)
        with ProcessPoolExecutor(PROCS) as ex:
            res = list(ex.map(photo_features, [(n, str(TEST_IMAGES), d4[n], sv[n], None) for n in names], chunksize=2))
        np.savez(OUT / f"test_rows_f{k}.npz", X=np.concatenate([r[1] for r in res]),
                 photo=np.concatenate([np.full(len(r[1]), i, np.int32) for i, r in enumerate(res)]), names=np.array(names))
        np.save(OUT / f"test_ovl_f{k}.npy", objarr([r[2] for r in res]), allow_pickle=True)
        print(f"test rows fold model {k}: {sum(len(r[1]) for r in res)} candidates", flush=True)


BLOCKS = "maskrcnn,spine_cnx,spine_cnxL,spine_r34,roi"                     # the evidence blocks of the final scorer, in order
EXTRA = [e for e in os.environ.get("FILAMENT_SCORER_BLOCKS", BLOCKS).split(",") if e]
TAG = "final"                                                               # outputs go to work/scorer/final/


def out_dir() -> Path:
    d = OUT / TAG if TAG else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def oof_X(z, extra=None) -> np.ndarray:
    blocks = [z["X"]] + [np.load(OUT / f"extra_{e}_oof.npy") for e in (EXTRA if extra is None else extra)]
    assert all(len(b) == len(z["X"]) for b in blocks)
    return np.concatenate(blocks, 1)


def test_X(t, k, extra=None) -> np.ndarray:
    blocks = [t["X"]] + [np.load(OUT / f"extra_{e}_test_f{k}.npy") for e in (EXTRA if extra is None else extra)]
    assert all(len(b) == len(t["X"]) for b in blocks)
    return np.concatenate(blocks, 1)


def fit_predict(Xtr, ytr, wtr, Xte):
    import lightgbm as lgb
    out = np.zeros(len(Xte))
    for s in SEEDS:
        m = lgb.train(dict(LGBM, seed=s), lgb.Dataset(Xtr, ytr, weight=wtr), ROUNDS)
        out += m.predict(Xte)
    out /= len(SEEDS)
    return out


def choose(q, ovl, tau, theta):
    """Indices of the set maximising sum(q - tau) with no two members overlapping by more than theta of the smaller
    one (exact; scipy MILP per connected component, src.scorer.select)."""
    from src.scorer.select import select
    n = len(q)
    if not n:
        return np.zeros(0, np.int64)
    a, b = np.nonzero(np.triu(ovl > theta, 1))
    return np.asarray(select(dict(conf=np.stack([a, b], 1).astype(np.int64)), np.asarray(q, np.float64), tau), np.int64)


def emit_selected(entries, q, keep):
    """The validated emission of the chosen candidates, painted in descending q."""
    from src.models.mask2former.submit import emit
    return [native(r) for r in emit([dict(score=float(q[i]) + 1.0, rle=native(entries[i]["rle"])) for i in keep], 0.0, 0, 100)]


def _oof_photo(job):
    from src.models.mask2former.submit import emit
    entries, q, ovl, records, grid, single = job
    gts = [[native(r) for r in rec] for rec in records]
    out = {}
    for key, pool, conf in (("base_d4", entries, 0.75), ("base_single", single, 0.85)):   # the current rules
        pred = [native(r) for r in emit([dict(score=e["score"], rle=native(e["rle"])) for e in pool], conf, 0, 100)]
        tot = np.zeros(4)
        for g in gts:
            tot += counts(pred, g)[0]
        out[key] = tot
    for tau, theta in grid:
        keep = choose(q, ovl, tau, theta) if theta is not None else np.flatnonzero(q > tau)
        pred = emit_selected(entries, q, keep)
        tot = np.zeros(4)
        for g in gts:
            tot += counts(pred, g)[0]
        out[(tau, theta)] = tot
    return out


def fit():
    """Cross-fitted scorer and out-of-fold selection (tau, theta), with paired-bootstrap comparisons against the 8-view
    and single-view pools emitted with their own rules."""
    from src.evaluation.sweep import records_for
    z = np.load(OUT / "oof_rows.npz", allow_pickle=True)
    ovl = np.load(OUT / "oof_ovl.npy", allow_pickle=True)
    y, photo, names, fold, nrec = z["y"], z["photo"], z["names"], z["fold"], z["n_records"]
    w = nrec[photo].astype(np.float64)
    ref = list(EXTRA)                                                    # the blocks of the scorer
    ref_label = "base" if not ref else "+" + "+".join(ref)
    sets = {ref_label: oof_X(z, ref)}
    if EXTRA and EXTRA != ref:
        sets["+" + "+".join(EXTRA)] = oof_X(z)
    Q = {}
    for label, X in sets.items():
        q = np.zeros(len(y))
        for f in range(5):
            te = fold[photo] == f
            q[te] = fit_predict(X[~te], y[~te], w[~te], X[te])
        Q[label] = q
        print(f"{label}: {X.shape[1]} features, correlation of q with the target {np.corrcoef(q, y)[0, 1]:.4f} "
              f"(the model's own fused score {np.corrcoef(X[:, 0], y)[0, 1]:.4f})", flush=True)
    last = list(sets)[-1]
    np.save(out_dir() / "oof_q.npy", Q[last])
    entries, recs, single = {}, {}, {}
    for k in range(5):
        os.environ["FILAMENT_VAL_FOLD"] = str(k)
        records, _ = records_for(ANN)
        d4 = json.loads((d4_path(k, "val")).read_text())
        sv = json.loads(sv_path(k, "val").read_text())
        for name, pairs in records.items():
            entries[name] = d4[name]; recs[name] = [r for _, r in pairs]; single[name] = sv[name]
    grid = [(round(t, 3), 0.1) for t in np.arange(0.175, 0.351, 0.025)]
    off = np.concatenate([[0], np.cumsum(np.bincount(photo, minlength=len(names)))])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(names), (4000, len(names)))
    bq = lambda A: np.array([pq(A[i].sum(0)) for i in idx])                       # noqa: E731
    report, nested_of, B = {}, {}, None
    for label in sets:
        q = Q[label]
        jobs = [(entries[n], q[off[i]:off[i + 1]], ovl[i], recs[n], grid, single[n]) for i, n in enumerate(names)]
        with ProcessPoolExecutor(PROCS) as ex:
            res = list(ex.map(_oof_photo, jobs, chunksize=2))
        T = {g: np.array([r[g] for r in res]) for g in grid}
        best = max(grid, key=lambda g: pq(T[g].sum(0)))
        nested = np.zeros((len(names), 4))                                        # tau chosen on 4 folds, counted on the 5th
        for f in range(5):
            inn = fold != f
            g = max(grid, key=lambda g: pq(T[g][inn].sum(0)))
            nested[~inn] = T[g][~inn]
        nested_of[label] = nested
        B = {k: np.array([r[k] for r in res]) for k in ("base_d4", "base_single")}
        report[label] = dict(best=dict(tau=best[0], theta=best[1], pq=pq(T[best].sum(0))), nested_pq=pq(nested.sum(0)),
                             nested_counts=nested.sum(0).tolist(), grid={str(g[0]): pq(T[g].sum(0)) for g in grid})
        print(f"{label}: OOF PQ {pq(nested.sum(0)):.4f} (tau chosen on 4 folds); best tau on all {best[0]} -> "
              f"{pq(T[best].sum(0)):.4f}; counts {nested.sum(0)[:3].astype(int).tolist()}", flush=True)
        np.save(out_dir() / f"oof_nested_counts_{label.replace('+', 'plus_')}.npy", nested)
    cmp = {}
    refs = dict(B, **({f"scorer {ref_label}": nested_of[ref_label]} if len(sets) > 1 else {}))
    bl = bq(nested_of[last])
    for k, A in refs.items():
        d = bl - bq(A)
        cmp[k] = dict(pq=pq(A.sum(0)), diff=pq(nested_of[last].sum(0)) - pq(A.sum(0)), lo=float(np.percentile(d, 2.5)),
                      hi=float(np.percentile(d, 97.5)), p_better=float((d > 0).mean()))
        print(f"{last} against {k} ({cmp[k]['pq']:.4f}): {cmp[k]['diff']:+.4f}, 95% {cmp[k]['lo']:+.4f} to "
              f"{cmp[k]['hi']:+.4f}, P(better) {cmp[k]['p_better']:.3f}", flush=True)
    out = dict(report[last], sets=report, against=cmp, extra=EXTRA)
    (out_dir() / "oof_report.json").write_text(json.dumps(out, indent=1))


def _test_photo(job):
    name, entries, q, ovl, tau, theta = job
    keep = choose(q, ovl, tau, theta) if theta is not None else np.flatnonzero(q > tau)
    pred = emit_selected(entries, q, keep)
    return name, [p["counts"].decode() if isinstance(p["counts"], bytes) else p["counts"] for p in pred]


def test():
    import csv
    z = np.load(OUT / "oof_rows.npz", allow_pickle=True)
    rep = json.loads((out_dir() / "oof_report.json").read_text())
    tau, theta = rep["best"]["tau"], rep["best"]["theta"]
    w = z["n_records"][z["photo"]].astype(np.float64)
    for k in range(5):
        t = np.load(OUT / f"test_rows_f{k}.npz", allow_pickle=True)
        ovl = np.load(OUT / f"test_ovl_f{k}.npy", allow_pickle=True)
        q = fit_predict(oof_X(z), z["y"], w, test_X(t, k))
        d4 = json.loads((d4_path(k, "test")).read_text())
        names = list(t["names"])
        off = np.concatenate([[0], np.cumsum(np.bincount(t["photo"], minlength=len(names)))])
        jobs = [(n, d4[n], q[off[i]:off[i + 1]], ovl[i], tau, theta) for i, n in enumerate(names)]
        with ProcessPoolExecutor(PROCS) as ex:
            res = dict(ex.map(_test_photo, jobs, chunksize=2))
        path = out_dir() / f"submission_selector_fold{k}.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["filament_id", "segmentation_rle"])
            for n in sorted(res):
                for i, c in enumerate(res[n], start=1):
                    wr.writerow([f"{Path(n).stem}_{i}", c])
        print(f"fold model {k}: {sum(map(len, res.values()))} instances -> {path}", flush=True)




# ------------------------------------------------------------------------------------------------ 5-fold ensemble
def _fuse_photo(job):
    """One test photo: the 5 fold models' candidates with their predicted q, grouped as the view fusion does (IoU link
    0.5, pixel vote 0.5). A group's q is the mean over the 5 models of the model's best member (0 for a model with
    no member), so q keeps its meaning and a candidate only some models propose is marked down."""
    from pycocotools import mask as mu
    from src.inference.tta import vote_rle
    name, pools, qs = job                               # pools[k]: entries of fold model k; qs[k]: their q
    cand = [native(e["rle"]) for p in pools for e in p]
    model = np.concatenate([np.full(len(p), k) for k, p in enumerate(pools)])
    q = np.concatenate(qs) if cand else np.zeros(0)
    if not cand:
        return name, [], np.zeros(0), np.zeros((0, 0), np.float32)
    iou = np.asarray(mu.iou(cand, cand, [0] * len(cand)))
    used, out, qf = set(), [], []
    for seed in np.argsort(-q):
        if seed in used:
            continue
        members = [seed] + [i for i in np.argsort(-q) if i not in used and i != seed and iou[seed, i] >= 0.5]
        used.update(members)
        rle = vote_rle([cand[i] for i in members], max(1, round(0.5 * len(members))))
        if mu.area(native(rle)) < 40:
            continue
        best = np.zeros(len(pools))
        for i in members:
            best[model[i]] = max(best[model[i]], q[i])
        out.append(dict(score=float(best.mean()), rle=rle)); qf.append(float(best.mean()))
    fused = [native(e["rle"]) for e in out]
    frac = np.asarray(mu.iou(fused, fused, [1] * len(fused))) if fused else np.zeros((0, 0))
    np.fill_diagonal(frac, 0)
    return name, out, np.asarray(qf), np.maximum(frac, frac.T).astype(np.float32)


def _ens_photo(job):
    name, entries, q, ovl, taus, theta = job
    res = {}
    for tau in taus:
        keep = choose(q, ovl, tau, theta)
        pred = emit_selected(entries, q, keep)
        res[tau] = [p["counts"].decode() if isinstance(p["counts"], bytes) else p["counts"] for p in pred]
    return name, res


def ensemble():
    """Selector on the 5-fold ensemble of the test candidates. No out-of-fold validation is possible (every training
    photo was seen by 4 of the 5 models), so nothing is tuned: primary = the tau and theta chosen on out-of-fold;
    secondary = the tau whose instance count matches the mean count of the 5 single-model selector files."""
    import csv
    import lightgbm as lgb
    z = np.load(OUT / "oof_rows.npz", allow_pickle=True)
    rep = json.loads((out_dir() / "oof_report.json").read_text())
    tau0, theta = rep["best"]["tau"], rep["best"]["theta"]
    w = z["n_records"][z["photo"]].astype(np.float64)
    models = [lgb.train(dict(LGBM, seed=s), lgb.Dataset(oof_X(z), z["y"], weight=w), ROUNDS) for s in SEEDS]
    pools, qs, names = [], [], None
    for k in range(5):
        t = np.load(OUT / f"test_rows_f{k}.npz", allow_pickle=True)
        names = list(t["names"])
        q = np.mean([m.predict(test_X(t, k)) for m in models], 0)
        off = np.concatenate([[0], np.cumsum(np.bincount(t["photo"], minlength=len(names)))])
        d4 = json.loads((d4_path(k, "test")).read_text())
        pools.append([d4[n] for n in names]); qs.append([q[off[i]:off[i + 1]] for i in range(len(names))])
    with ProcessPoolExecutor(PROCS) as ex:
        fused = list(ex.map(_fuse_photo, [(n, [pools[k][i] for k in range(5)], [qs[k][i] for k in range(5)])
                                          for i, n in enumerate(names)], chunksize=2))
    taus = [round(0.10 + 0.01 * i, 2) for i in range(31)]
    with ProcessPoolExecutor(PROCS) as ex:
        res = dict(ex.map(_ens_photo, [(n, e, q, o, taus, theta) for n, e, q, o in fused], chunksize=2))
    count = {t: sum(len(res[n][t]) for n in res) for t in taus}
    single = []
    for k in range(5):
        with open(out_dir() / f"submission_selector_fold{k}.csv") as fh:
            single.append(sum(1 for _ in fh) - 1)
    target = float(np.mean(single))
    tm = min(taus, key=lambda t: (abs(count[t] - target), t))
    info = dict(primary_tau=tau0, theta=theta, primary_count=count[min(taus, key=lambda t: abs(t - tau0))],
                matched_tau=tm, matched_count=count[tm], target=target, counts={str(t): c for t, c in count.items()})
    (out_dir() / "ensemble_info.json").write_text(json.dumps(info, indent=1))
    for tag, t in (("ens5", min(taus, key=lambda t: abs(t - tau0))), ("ens5_matched", tm)):
        with open(out_dir() / f"submission_selector_{tag}.csv", "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["filament_id", "segmentation_rle"])
            for n in sorted(res):
                for i, c in enumerate(res[n][t], start=1):
                    wr.writerow([f"{Path(n).stem}_{i}", c])
    print(json.dumps({k: v for k, v in info.items() if k != "counts"}), flush=True)


# ------------------------------------------------------------------------------- extra block: probability map
PB_NAMES = ["pb_mean", "pb_std", "pb_p10", "pb_frac50", "pb_frac25", "ring_mean", "ring_max", "comp50_ratio",
            "comp50_n", "comp50_cover", "comp25_ratio", "comp25_cover"]


def map_features(job):
    """What a probability map (uint8, 0-255) says about each candidate of one photo: its level inside the mask and
    in a ring around it, and how the mask compares with the map's own blobs at 0.5 and 0.25 (total size of the
    blobs it touches relative to the mask, and the share of those blobs it covers)."""
    import cv2
    from pycocotools import mask as mu
    map_path, j, entries, ch = job
    arr = np.load(map_path, mmap_mode="r")
    P = np.asarray(arr[j] if ch is None else arr[j, ch])
    out = np.zeros((len(entries), len(PB_NAMES)), np.float32)
    if not len(entries):
        return out
    labs = {}
    for key, thr in (("50", 127), ("25", 63)):
        n, lab, st, _ = cv2.connectedComponentsWithStats((P > thr).astype(np.uint8), connectivity=8)
        labs[key] = (lab, st[:, cv2.CC_STAT_AREA])
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i, e in enumerate(entries):
        c = native(e["rle"])
        x, y, w, h = (int(v) for v in mu.toBbox(c))
        pad = 10
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
        m = mu.decode(c)[y0:y1, x0:x1].astype(bool)
        if not m.any():
            continue
        p = P[y0:y1, x0:x1].astype(np.float32) / 255
        v = p[m]
        ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~m
        row = [v.mean(), v.std(), np.percentile(v, 10), (v > 0.5).mean(), (v > 0.25).mean(),
               p[ring].mean() if ring.any() else 0.0, p[ring].max() if ring.any() else 0.0]
        area = float(m.sum())
        for key in ("50", "25"):
            lab, areas = labs[key]
            ids, cnt = np.unique(lab[y0:y1, x0:x1][m], return_counts=True)
            keep = ids > 0
            tot = float(areas[ids[keep]].sum())
            row += [tot / area, float(keep.sum()), float(cnt[keep].sum()) / tot if tot else 0.0] if key == "50" else \
                   [tot / area, float(cnt[keep].sum()) / tot if tot else 0.0]
        out[i] = row
    return out


def extra_maps(block: str, oof_map, test_map):
    """Write work/scorer/extra_<block>_{oof,test_f<k>}.npy: map features aligned with the candidate rows.
    oof_map(k) -> (npy path, file list, channel or None) of the out-of-fold maps of user fold k; test_map(k) -> the
    same for the test photos as seen with fold model k."""
    z = np.load(OUT / "oof_rows.npz", allow_pickle=True)
    names, fold = list(z["names"]), z["fold"]
    jobs = []
    for k in range(5):
        path, files, ch = oof_map(k)
        pos = {n: j for j, n in enumerate(files)}
        d4 = json.loads((d4_path(k, "val")).read_text())
        jobs += [(str(path), pos[n], d4[n], ch) for n, f in zip(names, fold) if f == k]
    with ProcessPoolExecutor(PROCS) as ex:
        res = list(ex.map(map_features, jobs, chunksize=2))
    X = np.concatenate(res)
    assert len(X) == len(z["X"]), (len(X), len(z["X"]))
    np.save(OUT / f"extra_{block}_oof.npy", X)
    for k in range(5):
        path, files, ch = test_map(k)
        pos = {n: j for j, n in enumerate(files)}
        t = np.load(OUT / f"test_rows_f{k}.npz", allow_pickle=True)
        d4 = json.loads((d4_path(k, "test")).read_text())
        with ProcessPoolExecutor(PROCS) as ex:
            res = list(ex.map(map_features, [(str(path), pos[n], d4[n], ch) for n in t["names"]], chunksize=2))
        X = np.concatenate(res)
        assert len(X) == len(t["X"])
        np.save(OUT / f"extra_{block}_test_f{k}.npy", X)
    print(f"extra block {block}: {len(PB_NAMES)} features written", flush=True)


def _assembled(root: Path, arch: str, ch: int):
    """(oof_map, test_map) for the assembled U-Net maps <root>/{train_f<k>,test}/test<arch>.npy, channel ch.
    Test photos: the map averaged over the fold models, the same for every fold model's candidates."""
    files = lambda p: json.loads(Path(str(p) + ".json").read_text())["files"]      # noqa: E731
    oof = lambda k: (root / f"train_f{k}" / f"test{arch}.npy", files(root / f"train_f{k}" / f"test{arch}.npy"), ch)  # noqa: E731
    tst = lambda k: (root / "test" / f"test{arch}.npy", files(root / "test" / f"test{arch}.npy"), ch)               # noqa: E731
    return oof, tst


def det_features(job):
    """Agreement of each candidate with a detector's candidate bank, averaged over the given record files:
    best IoU, the score of that detection, the best score among detections with IoU > 0.5, the number with
    IoU > 0.3, and the best IoU x score."""
    from pycocotools import mask as mu
    entries, paths = job
    out = np.zeros((len(entries), 5), np.float32)
    if not len(entries):
        return out
    cand = [native(e["rle"]) for e in entries]
    for path in paths:
        rows = json.loads(Path(path).read_text())["views"]["0.5"]
        if not rows:
            continue
        ext = [dict(size=r["rle"]["size"], counts=r["rle"]["counts"].encode()) for r in rows]
        sc = np.array([r["score"] for r in rows], np.float32)
        m = np.asarray(mu.iou(cand, ext, [0] * len(ext)))
        out[:, 0] += m.max(1); out[:, 1] += sc[m.argmax(1)] * (m.max(1) > 0)
        out[:, 2] += np.where(m > 0.5, sc[None], 0).max(1); out[:, 3] += (m > 0.3).sum(1); out[:, 4] += (m * sc[None]).max(1)
    return out / max(len(paths), 1)


BANKS = paths.WORK / "banks"                      # Mask R-CNN and ROI candidate banks
MAPS = paths.SCRATCH / "maps"                     # U-Net maps


def evidence():
    """Evidence of the Mask R-CNN (block maskrcnn: a training photo of fold k uses maskrcnn-f<k>, a test photo the mean
    over the 5) and of the spine channel of the 3 U-Nets (blocks spine_<arch>: out of fold, test = mean of the 5 fold
    models)."""
    folds = json.loads(Path(os.environ["FILAMENT_FOLDS_JSON"]).read_text())["folds"]
    uf = {n: k for k, names in enumerate(folds) for n in names}
    rec = {}
    for k in range(5):
        for part in ("train", "test"):
            man = json.loads((BANKS / f"maskrcnn-f{k}" / part / "manifest.json").read_text())
            rec[(k, part)] = {r["file_name"]: str(BANKS / f"maskrcnn-f{k}" / part / r["file"]) for r in man["records"]}
    z = np.load(OUT / "oof_rows.npz", allow_pickle=True)
    if not (OUT / "extra_maskrcnn_oof.npy").is_file():
        jobs = []
        for k in range(5):
            d4 = json.loads((d4_path(k, "val")).read_text())
            jobs += [(d4[n], [rec[(uf[n], "train")][n]]) for n, f in zip(z["names"], z["fold"]) if f == k]
        with ProcessPoolExecutor(PROCS) as ex:
            X = np.concatenate(list(ex.map(det_features, jobs, chunksize=2)))
        assert len(X) == len(z["X"])
        np.save(OUT / "extra_maskrcnn_oof.npy", X)
        for k in range(5):
            t = np.load(OUT / f"test_rows_f{k}.npz", allow_pickle=True)
            d4 = json.loads((d4_path(k, "test")).read_text())
            with ProcessPoolExecutor(PROCS) as ex:
                X = np.concatenate(list(ex.map(det_features, [(d4[n], [rec[(v, "test")][n] for v in range(5)]) for n in t["names"]],
                                               chunksize=2)))
            np.save(OUT / f"extra_maskrcnn_test_f{k}.npy", X)
        print("extra block maskrcnn: 5 features written", flush=True)
    for arch in ("cnx", "cnxL", "r34"):
        if not (OUT / f"extra_spine_{arch}_oof.npy").is_file():
            extra_maps(f"spine_{arch}", *_assembled(MAPS, arch, 1))


def roi_evidence():
    """Evidence of the ROI detector + refiner (block roi, the 5 features of det_features). Training photos of fold k: the
    bank of pair f<k>, which did not train on them. Test: the mean over the 5 fold pairs' banks."""
    folds = json.loads(Path(os.environ["FILAMENT_FOLDS_JSON"]).read_text())["folds"]
    uf = {n: k for k, names in enumerate(folds) for n in names}

    def records(tag, part):
        man = json.loads((BANKS / f"roi-{tag}" / part / "manifest.json").read_text())
        return {r["file_name"]: str(BANKS / f"roi-{tag}" / part / r["file"]) for r in man["records"]}
    tr = {k: records(f"f{k}", "train") for k in range(5)}
    z = np.load(OUT / "oof_rows.npz", allow_pickle=True)
    if not (OUT / "extra_roi_oof.npy").is_file():
        jobs = []
        for k in range(5):
            d4 = json.loads((d4_path(k, "val")).read_text())
            jobs += [(d4[n], [tr[uf[n]][n]]) for n, f in zip(z["names"], z["fold"]) if f == k]
        with ProcessPoolExecutor(PROCS) as ex:
            X = np.concatenate(list(ex.map(det_features, jobs, chunksize=2)))
        assert len(X) == len(z["X"])
        np.save(OUT / "extra_roi_oof.npy", X)
    te = {k: records(f"f{k}", "test") for k in range(5)}
    for k in range(5):
        t = np.load(OUT / f"test_rows_f{k}.npz", allow_pickle=True)
        d4 = json.loads((d4_path(k, "test")).read_text())
        with ProcessPoolExecutor(PROCS) as ex:
            X = np.concatenate(list(ex.map(det_features, [(d4[n], [te[v][n] for v in range(5)]) for n in t["names"]], chunksize=2)))
        np.save(OUT / f"extra_roi_test_f{k}.npy", X)
    print("extra block roi: 5 features written", flush=True)


if __name__ == "__main__":
    {"features": build_features, "evidence": evidence, "roi_evidence": roi_evidence, "fit": fit, "test": test,
     "ensemble": ensemble}[sys.argv[1]]()
