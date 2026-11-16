"""The evidence models: Mask R-CNN and 3 U-Nets (cnx: ConvNeXt-T 24k steps, cnxL: ConvNeXt-T 48k, r34: ResNet34 30k) x 5
folds, trained on the user's folds, and their predictions.

    python -m src.training.evidence_models train [--gpus 0,1,2,3]   training cache, then the 20 trainings (done ones skipped)
    python -m src.training.evidence_models infer [--gpus 0,1,2,3]   Mask R-CNN candidate banks (work/banks/maskrcnn-f<k>) and
                                                                    U-Net maps (scratch/maps/<unit>/test<src>.zip, compressed:
                                                                    train_f<k> out of fold, test = mean of 5)

Each fold k gets its own cache folder: the same arrays (links) and a meta.json whose fold is k for the user's fold-k photos
and 9 for all others, so the trainers hold out exactly fold k.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from src import paths  # noqa: E402

REPO = paths.REPO
MODELS = paths.MODELS                               # trained models: maskrcnn-f<k>, unet_<arch>-f<k>
BANKS = paths.WORK / "banks"                         # Mask R-CNN candidate banks
SHM = paths.SCRATCH                                  # training cache, U-Net maps
LOGS = paths.OUTPUTS / "logs" / "evidence"
PY = sys.executable
DATA = paths.DATA
FOLDS_JSON = paths.FOLDS_JSON
DENSE = {"cnx": ("unet", "tu-convnext_tiny", 24000), "cnxL": ("unet", "tu-convnext_tiny", 48000),
         "r34": ("unet", "resnet34", 30000)}
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p),
           PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1", OMP_NUM_THREADS="4", HF_HUB_ENABLE_HF_TRANSFER="0")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def user_fold_of() -> dict:
    folds = json.loads(FOLDS_JSON.read_text())["folds"]
    return {n: k for k, names in enumerate(folds) for n in names}


def build_cache(workers: int = 24) -> Path:
    """The training cache (images, soft mask target, spine target, centres) of the 707 photos (src.training.prep_cache)."""
    cache = SHM / "evidence_cache"
    if (cache / "centers.json").is_file():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "-m", "src.training.prep_cache", "--cache", str(cache), "--data", str(DATA),
                    "--split", str(paths.CONFIGS / "base_split.json"), "--workers", str(workers)],
                   cwd=REPO, env=ENV, check=True)
    return cache


def fold_caches(cache: Path) -> dict:
    """{k: cache folder whose meta selects the user's folds other than k}."""
    meta = json.loads((cache / "meta.json").read_text())
    uf = user_fold_of()
    out = {}
    for k in range(5):
        d = SHM / f"evidence_cache_f{k}"
        d.mkdir(parents=True, exist_ok=True)
        for f in ("train_img.npy", "train_soft.npy", "train_spine.npy", "centers.json"):
            if not (d / f).exists():
                (d / f).symlink_to(cache / f)
        m = {"train": [dict(r, fold=(k if uf[r["file_name"]] == k else 9)) for r in meta["train"]]}
        assert sum(r["fold"] == k for r in m["train"]) == sum(v == k for v in uf.values())
        (d / "meta.json").write_text(json.dumps(m))
        out[k] = d
    return out


def jobs(caches: dict, steps: int | None = None, iters: int | None = None, workers: int = 6) -> list:
    """(name, done flag, units of a GPU (2 = alone), command), longest first."""
    runs = MODELS
    out = []
    for k in range(5):
        it = f" --iters {iters}" if iters else ""
        out.append((f"maskrcnn-f{k}", runs / f"maskrcnn-f{k}" / "done.json", 1,
                    f"{PY} -m src.training.train_maskrcnn --name maskrcnn-f{k} --fold {k} --cache {caches[k]} --runs {runs} "
                    f"--workers 5 --ann {DATA / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'}{it}"))
    for src in ("cnxL", "cnx", "r34"):
        arch, enc, st = DENSE[src]
        for k in range(5):
            out.append((f"unet_{src}-f{k}", runs / f"unet_{src}-f{k}" / "done.json", 1,
                        f"{PY} -m src.training.train_unet --name unet_{src}-f{k} --fold {k} --arch {arch} --encoder {enc} "
                        f"--steps {steps or st} --workers {workers} --cache {caches[k]} --runs {runs}"))
    order = {"unet_cnxL": 0, "maskrcnn": 1, "unet_cnx": 2, "unet_r34": 3}
    return sorted(out, key=lambda j: order[j[0].rsplit("-f", 1)[0]])


def run_queue(job_list, gpus=(0, 1, 2, 3), capacity: int = 2) -> None:
    """Each GPU holds `capacity` units; a job uses 1 (U-Net, 4 GB) or 2 (Mask R-CNN, alone). Finished jobs are skipped."""
    todo = [j for j in job_list if not Path(j[1]).is_file()]
    LOGS.mkdir(parents=True, exist_ok=True)
    free = {g: capacity for g in gpus}
    running, failed = {}, []
    log(f"queue: {len(todo)} of {len(job_list)} jobs")
    while todo or running:
        started = True
        while started:
            started = False
            for j in list(todo):
                g = next((g for g in gpus if free[g] >= j[2]), None)
                if g is None:
                    continue
                todo.remove(j); free[g] -= j[2]
                env = dict(ENV, CUDA_VISIBLE_DEVICES=str(g))
                p = subprocess.Popen(shlex.split(j[3]), cwd=REPO, env=env,
                                     stdout=open(LOGS / f"{j[0]}.log", "a"), stderr=subprocess.STDOUT)
                running[p] = (j, g, time.time())
                log(f"  start {j[0]} on GPU {g}")
                started = True
                break
        time.sleep(20)
        for p in [p for p in running if p.poll() is not None]:
            j, g, t0 = running.pop(p)
            free[g] += j[2]
            ok = p.returncode == 0 and Path(j[1]).is_file()
            log(f"  {'done' if ok else 'FAILED'} {j[0]} ({(time.time() - t0) / 60:.0f} min)")
            if not ok:
                failed.append(j[0])
    if failed:
        raise RuntimeError(f"failed: {failed} (logs in {LOGS})")




def fold_major(job_list) -> list:
    """Fold by fold: every model of fold 0, then fold 1, ...; inside a fold the longest first."""
    order = {"unet_cnxL": 0, "maskrcnn": 1, "unet_cnx": 2, "unet_r34": 3}
    return sorted(job_list, key=lambda j: (int(j[0].rsplit("-f", 1)[1]), order[j[0].rsplit("-f", 1)[0]]))


# ================================================================================ inference with the retrained models
TEST_IMAGES = DATA / "test" / "test_images"
TRAIN_IMAGES = DATA / "train" / "train_images"


def names_of(fold: int | None) -> list:
    if fold is None:
        return sorted(p.name for p in TEST_IMAGES.glob("*.jpeg"))
    return sorted(n for n, k in user_fold_of().items() if k == fold)


def dense_job(src: str, fold: int) -> None:
    """Both channels of U-Net <src>-f<fold> (their D4 mean, bf16): its held-out photos and the test photos."""
    import numpy as np
    import torch
    from src.data.photos import disk_stats
    from src.models.unet import build_model, normalize, predict_full
    import cv2
    arch, enc, _ = DENSE[src]
    m = build_model(arch, enc, weights=None)
    m.load_state_dict(torch.load(MODELS / f"unet_{src}-f{fold}" / "ema.pt", map_location="cpu"))
    m = m.cuda().eval().to(memory_format=torch.channels_last)
    from src.utils.mapstore import MapWriter, map_path
    for unit, names, images in ((f"train_f{fold}", names_of(fold), TRAIN_IMAGES), (f"test_part_f{fold}", names_of(None), TEST_IMAGES)):
        out = map_path(SHM / "maps" / unit, f"test{src}")             # compressed, lossless (src/utils/mapstore.py)
        if out.is_file():
            continue
        with MapWriter(out, names) as w, torch.no_grad():
            for j, n in enumerate(names):
                g = cv2.imread(str(images / n), cv2.IMREAD_GRAYSCALE)
                mean, std, _ = disk_stats(g)
                p = predict_full(m, torch.from_numpy(normalize(g, mean, std))[None, None].cuda())
                w.write(j, (p.clamp(0, 1) * 255).round().byte().cpu().numpy())


def average_test(src: str) -> None:
    """test/test<src>.zip = round(mean of the 5 fold models' test maps); the per-fold parts are deleted."""
    import numpy as np
    from src.utils.mapstore import MapWriter, map_path, open_maps
    out = map_path(SHM / "maps" / "test", f"test{src}")
    if out.is_file():
        return
    parts = [map_path(SHM / "maps" / f"test_part_f{k}", f"test{src}") for k in range(5)]
    sets = [open_maps(p) for p in parts]
    names = sets[0].files
    assert all(m.files == names for m in sets)
    with MapWriter(out, names) as w:
        for j in range(len(names)):
            w.write(j, np.round(np.mean([m[j].astype(np.float32) for m in sets], 0)).astype(np.uint8))
    for m, p in zip(sets, parts):
        m.close(); p.unlink(missing_ok=True)


def det_job(fold: int) -> None:
    """Candidates of Mask R-CNN maskrcnn-f<fold> (full 2048 px, fp32): its held-out photos (train) and the test photos."""
    import numpy as np
    import cv2
    from src.models import maskrcnn
    for split, names, images in (("train", names_of(fold), TRAIN_IMAGES), ("test", names_of(None), TEST_IMAGES)):
        out = BANKS / f"maskrcnn-f{fold}" / split
        if (out / "manifest.json").is_file():
            continue
        imgs = np.stack([cv2.imread(str(images / n), cv2.IMREAD_GRAYSCALE) for n in names])
        maskrcnn.infer(f"maskrcnn-f{fold}", imgs, [dict(file_name=n) for n in names], out, device="cuda", log=log)


def infer_all(gpus=(0, 1, 2, 3)) -> None:
    import shutil
    if (SHM / "evidence_cache").exists():
        shutil.rmtree(SHM / "evidence_cache")              # training is over: frees ~8 GB of scratch
        for k in range(5):
            shutil.rmtree(SHM / f"evidence_cache_f{k}", ignore_errors=True)
    me = f"{PY} -m src.training.evidence_models"
    # one prediction job per GPU (units 2): two Mask R-CNN jobs pasting 2048 px masks do not fit in 12 GB
    jl = [(f"det_infer_f{k}", BANKS / f"maskrcnn-f{k}" / "test" / "manifest.json", 2, f"{me} det_job {k}") for k in range(5)]
    run_queue(jl, gpus)
    for src in ("cnxL", "r34", "cnx"):
        jl = [(f"dense_infer_{src}_f{k}", SHM / "maps" / f"test_part_f{k}" / f"test{src}.zip", 2,
               f"{me} dense_job {src} {k}") for k in range(5)
              if not (SHM / "maps" / "test" / f"test{src}.zip").is_file()]
        run_queue(jl, gpus)
        average_test(src)
        log(f"maps of {src} done")


def _gpus() -> tuple:
    if "--gpus" in sys.argv:
        return tuple(int(g) for g in sys.argv[sys.argv.index("--gpus") + 1].split(","))
    return tuple(int(g) for g in os.environ.get("FILAMENT_GPUS", "0,1,2,3").split(","))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        c = build_cache()
        run_queue(fold_major(jobs(fold_caches(c))), gpus=_gpus())
        log("all trainings done")
    elif cmd == "infer":
        infer_all(gpus=_gpus())
    elif cmd == "dense_job":
        dense_job(sys.argv[2], int(sys.argv[3]))
    elif cmd == "det_job":
        det_job(int(sys.argv[2]))
